"""
Evaluation runner — Stage 3 of the data pipeline.

Executes TestCases against the LNL runtime, judges outcomes with an LLM, and
reports correctness and cost metrics.

Usage:
    python -m src.data.evaluate \\
        -i outputs/data/zapier/20260322_120000/test_cases.jsonl \\
        --runs 3 \\
        --model gpt-4o --judge-model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.data.schema import (
    EvalSummary,
    EventResult,
    ModificationResult,
    TestCase,
    TestCaseResult,
    to_lnl_definition,
)
from src.data.utils import (
    add_common_args,
    infer_provider,
    load_jsonl,
    print_run_info,
)


# ── Timestamp parsing ──────────────────────────────────────────────────────────

def parse_when(when: str) -> int:
    """Convert 'W02-1T10:30' → ordinal minutes for sorting."""
    week_part, time_part = when.split("T")
    w, d = week_part.lstrip("W").split("-")
    h, m = time_part.split(":")
    return (int(w) * 7 + int(d)) * 1440 + int(h) * 60 + int(m)


# ── Evidence gathering ─────────────────────────────────────────────────────────

def gather_evidence(rt, results, recipient: str) -> str:
    """Collect observable evidence after an event for the LLM judge."""
    parts: list[str] = []

    # Replies from the chain triggered by this event
    replies = [r for r in results if r.reply and str(r.reply).strip()]
    if replies:
        parts.append("Replies:\n" + "\n".join(f"  [{r.object_id}]: {r.reply}" for r in replies))

    # External actions declared by objects (Slack, Email, Jira, etc.)
    ext_actions = [ea for r in results for ea in r.external_actions]
    if ext_actions:
        action_lines = []
        for ea in ext_actions:
            params_str = ", ".join(f"{k}={v}" for k, v in ea.params.items()) if ea.params else ""
            line = f"  [{ea.system}.{ea.action}]"
            if params_str:
                line += f" ({params_str})"
            line += f": {ea.content}"
            action_lines.append(line)
        parts.append("External actions:\n" + "\n".join(action_lines))

    # State of all objects (captures write-service audit trails)
    for obj_id, obj in rt._bus.objects.items():
        state = obj.state
        if isinstance(state, dict):
            state_str = json.dumps(state, indent=2) if state else "(empty)"
        else:
            state_str = str(state).strip() or "(empty)"
        parts.append(f"State of [{obj_id}]:\n{state_str}")

    return "\n\n".join(parts) if parts else "(no observable state)"


# ── Core execution ─────────────────────────────────────────────────────────────

def _print_message(msg) -> None:
    """Print a message exchange between LLM-objects."""
    arrow = "↩" if msg.type.value == "reply" else "→"
    content = msg.content[:120].replace("\n", " ")
    print(f"      {msg.sender} {arrow} {msg.recipient} ({msg.type.value}): {content}")


def _execute_test_case_inner(
    tc: TestCase,
    brain,
    harness,
    debug_messages: bool = False,
    timeout_s: Optional[float] = None,
) -> tuple[list[EventResult], list[ModificationResult]]:
    """Run a single TestCase and return event + modification results."""
    from src.lnl.gateway import EventGateway
    from src.lnl.runtime import Runtime
    from src.lnl.tools import CodeExecutor, ToolRegistry

    # 1. Create Runtime, EventGateway, and start the live environment
    tool_registry = ToolRegistry()
    tool_registry.register("execute_code", CodeExecutor())

    rt = Runtime(brain, strict_peers=False, tool_registry=tool_registry)
    if debug_messages:
        rt.set_message_listener(_print_message)
    gw = EventGateway(rt)

    for obj_def in tc.objects:
        rt.create_object(to_lnl_definition(obj_def))

    # Start the runtime — objects are now live instances
    rt.start()

    try:
        return _run_test_case_timeline(tc, rt, gw, harness, timeout_s=timeout_s)
    finally:
        rt.stop()


def _run_with_timeout(fn, timeout_s: Optional[float]):
    """Run fn() with an optional per-step timeout. Returns (result, timed_out)."""
    if timeout_s is None:
        return fn(), False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_s), False
        except concurrent.futures.TimeoutError:
            future.cancel()
            return [], True


def _run_test_case_timeline(
    tc: TestCase,
    rt,
    gw,
    harness,
    timeout_s: Optional[float] = None,
) -> tuple[list[EventResult], list[ModificationResult]]:
    """Execute steps and timeline events against a live runtime."""
    event_results: list[EventResult] = []
    mod_results: list[ModificationResult] = []

    # 2. Run steps — initialize state and assert default (no-modification) behavior
    for i, step in enumerate(tc.steps):
        t0 = time.monotonic()
        results, timed_out = _run_with_timeout(
            lambda s=step: gw.dispatch(s.target, s.text), timeout_s,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if step.expect is not None:
            if timed_out:
                event_results.append(EventResult(
                    event_id=f"S{i+1:03d}",
                    passed=False,
                    reasoning=f"Timeout after {timeout_s}s",
                    expected=step.expect.action,
                    latency_ms=latency_ms,
                ))
            else:
                in_tok = sum(r.metrics.input_tokens for r in results if r.metrics)
                out_tok = sum(r.metrics.output_tokens for r in results if r.metrics)
                evidence = gather_evidence(rt, results, step.target)
                condition = step.expect.action
                passed, reasoning = harness.evaluate_assertion(condition, evidence)
                event_results.append(EventResult(
                    event_id=f"S{i+1:03d}",
                    passed=passed,
                    reasoning=reasoning,
                    expected=condition,
                    evidence=evidence,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                ))

    # 3. Build sorted timeline: tag each item with its type and when-ordinal
    timeline: list[tuple[int, str, object]] = []
    for mod in tc.modifications:
        timeline.append((parse_when(mod.when), "mod", mod))
    for evt in tc.events:
        timeline.append((parse_when(evt.when), "event", evt))
    timeline.sort(key=lambda x: x[0])

    for _, kind, item in timeline:
        if kind == "mod":
            t0 = time.monotonic()
            results, timed_out = _run_with_timeout(
                lambda it=item: rt.send(it.target, it.intent, sender=it.source),
                timeout_s,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            if timed_out:
                mod_results.append(ModificationResult(mod_id=item.id, latency_ms=latency_ms))
            else:
                in_tok = sum(r.metrics.input_tokens for r in results if r.metrics)
                out_tok = sum(r.metrics.output_tokens for r in results if r.metrics)
                mod_results.append(ModificationResult(
                    mod_id=item.id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                ))

        else:  # event
            t0 = time.monotonic()
            if item.call_type == "send_event":
                # Wrap as structured JSON envelope so the object receives a typed external event
                payload = json.dumps({"system": item.source, "content": item.input})
                results, timed_out = _run_with_timeout(
                    lambda it=item, p=payload: gw.dispatch(it.recipient, p, source=it.source),
                    timeout_s,
                )
            else:
                results, timed_out = _run_with_timeout(
                    lambda it=item: rt.send(it.recipient, it.input, sender=it.source),
                    timeout_s,
                )
            latency_ms = (time.monotonic() - t0) * 1000

            if timed_out:
                event_results.append(EventResult(
                    event_id=item.id,
                    passed=False,
                    reasoning=f"Timeout after {timeout_s}s",
                    expected=item.expect.action,
                    latency_ms=latency_ms,
                ))
            else:
                in_tok = sum(r.metrics.input_tokens for r in results if r.metrics)
                out_tok = sum(r.metrics.output_tokens for r in results if r.metrics)

                evidence = gather_evidence(rt, results, item.recipient)
                condition = item.expect.action
                passed, reasoning = harness.evaluate_assertion(condition, evidence)

                event_results.append(EventResult(
                    event_id=item.id,
                    passed=passed,
                    reasoning=reasoning,
                    expected=condition,
                    evidence=evidence,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                ))

    return event_results, mod_results


def execute_test_case(
    tc: TestCase,
    brain,
    harness,
    timeout_s: Optional[float] = None,
    debug_messages: bool = False,
) -> tuple[list[EventResult], list[ModificationResult]]:
    """Run a single TestCase with a per-event timeout (seconds).

    Each step, modification, and event gets its own timeout. If a single
    step times out, it is marked as failed and execution continues.
    """
    return _execute_test_case_inner(
        tc, brain, harness,
        debug_messages=debug_messages,
        timeout_s=timeout_s,
    )


# ── Output path ────────────────────────────────────────────────────────────────

def default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_eval.jsonl"


# ── Verbose output ────────────────────────────────────────────────────────────

def _print_verbose(tc_result: TestCaseResult) -> None:
    """Print detailed per-event breakdown to console."""
    for ev in tc_result.events:
        status = "PASS" if ev.passed else "FAIL"
        print(f"    [{status}] {ev.event_id}")
        print(f"      Expected: {ev.expected}")
        if ev.evidence:
            # Indent evidence lines for readability
            indented = ev.evidence.replace("\n", "\n        ")
            print(f"      Evidence: {indented}")
        print(f"      Judge:    {ev.reasoning}")
        print()


# ── Main runner ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> Path:
    """Run evaluation. Returns the output path."""
    logging.basicConfig(level=logging.WARNING)

    if args.output is None:
        args.output = default_output_path(args.input)

    if args.provider is None:
        args.provider = infer_provider(args.model)

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    test_cases = load_jsonl(args.input, TestCase)

    if args.limit:
        test_cases = test_cases[: args.limit]

    timeout_s: Optional[float] = getattr(args, "timeout", None)

    print(f"Loaded {len(test_cases)} test cases from {args.input}")
    judge_model = args.judge_model or args.model
    judge_provider = args.judge_provider or infer_provider(judge_model)
    extra_info = {
        "Runs per test case": str(args.runs),
        "Timeout per event": f"{timeout_s}s" if timeout_s else "none",
    }
    if args.judge_model:
        extra_info["Judge"] = f"{judge_provider}/{judge_model}"
    print_run_info(
        args.provider,
        args.model,
        getattr(args, "seed", None),
        extra_info,
    )

    # Build LNL brain (for objects) and judge brain (for assertions)
    def _make_brain(provider, model):
        if provider == "openai":
            from src.lnl.brain import OpenAIBrain
            return OpenAIBrain(model=model)
        else:
            from src.lnl.brain import AnthropicBrain
            return AnthropicBrain(model=model)

    brain = _make_brain(args.provider, args.model)
    judge_brain = _make_brain(judge_provider, judge_model) if args.judge_model else None

    from src.lnl.benchmark import BenchmarkHarness
    harness = BenchmarkHarness(brain=brain, judge=judge_brain)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_tc_results: list[TestCaseResult] = []

    with open(args.output, "w") as f:
        for tc in test_cases:
            for run_idx in range(args.runs):
                label = f"{tc.id} run={run_idx}"
                print(f"  Evaluating {label} ...", end=" ", flush=True)
                try:
                    event_results, mod_results = execute_test_case(
                        tc, brain, harness, timeout_s,
                        debug_messages=getattr(args, "debug_messages", False),
                    )
                    pass_rate = (
                        sum(1 for e in event_results if e.passed) / len(event_results)
                        if event_results else 1.0
                    )
                    tc_result = TestCaseResult(
                        tc_id=tc.id,
                        name=tc.name,
                        domain=tc.domain,
                        run_index=run_idx,
                        events=event_results,
                        modifications=mod_results,
                        pass_rate=pass_rate,
                    )
                    f.write(tc_result.model_dump_json() + "\n")
                    f.flush()
                    all_tc_results.append(tc_result)
                    print(f"pass_rate={pass_rate:.2f}")
                    if args.verbose:
                        _print_verbose(tc_result)
                except Exception as e:
                    print(f"FAILED: {e}", file=sys.stderr)

    # Write summary
    summary = _compute_summary(all_tc_results)
    with open(args.output, "a") as f:
        f.write(summary.model_dump_json() + "\n")

    print()
    print(f"Complete. Output: {args.output}")
    print(f"Mean pass rate: {summary.mean_pass_rate:.3f}  std: {summary.pass_rate_std:.3f}")
    return args.output


def _compute_summary(results: list[TestCaseResult]) -> EvalSummary:
    """Compute aggregate metrics across all test case results."""
    all_events = [e for r in results for e in r.events]
    all_mods = [m for r in results for m in r.modifications]
    total_runs = len(results)
    total_test_cases = len({r.tc_id for r in results})

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # Mean pass rate: average across all (tc, run) results
    pass_rates = [r.pass_rate for r in results]
    mean_pass_rate = mean(pass_rates)

    # Behavioral consistency: mean of per-TC std devs across runs.
    # Groups results by tc_id, computes std dev within each group, then averages.
    # Requires --runs > 1; returns 0.0 when each TC has only one run.
    by_tc: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_tc[r.tc_id].append(r.pass_rate)
    per_tc_stds = [
        statistics.stdev(rates) for rates in by_tc.values() if len(rates) > 1
    ]
    pass_rate_std = mean(per_tc_stds)

    return EvalSummary(
        total_test_cases=total_test_cases,
        total_runs=total_runs,
        total_events=len(all_events),
        mean_pass_rate=mean_pass_rate,
        pass_rate_std=pass_rate_std,
        mean_event_input_tokens=mean([e.input_tokens for e in all_events]),
        mean_event_output_tokens=mean([e.output_tokens for e in all_events]),
        mean_event_latency_ms=mean([e.latency_ms for e in all_events]),
        mean_mod_input_tokens=mean([m.input_tokens for m in all_mods]),
        mean_mod_output_tokens=mean([m.output_tokens for m in all_mods]),
        mean_mod_latency_ms=mean([m.latency_ms for m in all_mods]),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate test cases against the LNL runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.evaluate -i outputs/data/zapier/20260322_120000/test_cases.jsonl
  python -m src.data.evaluate -i test_cases.jsonl --runs 3 --model claude-sonnet-4-6
  python -m src.data.evaluate -i test_cases.jsonl --model gpt-4o --judge-model claude-sonnet-4-6
""",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to test cases JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSONL path (default: {stem}_eval.jsonl next to input)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per test case for behavioral consistency (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Wall-clock timeout per step/event (not per test case); timed-out steps are marked failed and execution continues (default: 60)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print detailed per-event evidence, expected conditions, and judge reasoning",
    )
    parser.add_argument(
        "--debug-messages",
        action="store_true",
        default=False,
        help="Print messages exchanged between LLM-objects during evaluation",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for LLM-as-judge (default: same as --model). Provider is inferred from model name.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="Provider for judge model (inferred from judge-model if not specified)",
    )
    add_common_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
