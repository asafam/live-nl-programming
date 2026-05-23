"""Stage 1e validator: grade the base event sequence of each Workflow.

For each Workflow.events (role='base'), run deterministic health checks and
call an LLM judge to detect sequence-level issues (paradoxes, causal orphans,
expect leakage, incompleteness). Output: one EventSequenceValidation per workflow.

Issue codes detected:
  sequential_paradox   — event N assumes state contradicted by event N-1's output
  causal_orphan        — actor takes action without having received a prior prerequisite
  expect_leak          — expect.action includes outcomes from a later event's input
  expect_incomplete    — expect.action misses obvious externally-observable outputs
  expect_null_invalid  — expect is null but steps describe a clear output for this trigger
  redundant            — near-duplicate event (same trigger class, no meaningful variation)

Usage:
    python -m src.data.validate_workflow_events \\
        --input outputs/.../workflows.jsonl \\
        --provider azure --judge-model gpt-5.4 \\
        --workers 4 \\
        [--output workflows__event_validation.jsonl] [--no-fail]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.data.llm import create_llm
from src.data.schema import (
    Event,
    EventSequenceJudgement,
    EventSequenceValidation,
    EventVerdict,
    EventVerdictOutput,
    Workflow,
)
from src.data.utils import generate_with_retries, load_jsonl

load_dotenv()

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent
    / "config" / "prompts" / "data-gen" / "validate_workflow_events.yaml"
)


def _load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["prompt"]


def _format_events(events: list[Event]) -> str:
    lines = []
    for e in events:
        expect_str = e.expect.action if e.expect else "null"
        lines.append(
            f"{e.id}:\n"
            f"  source: {e.source}\n"
            f"  input:  {e.input}\n"
            f"  expect: {expect_str}"
        )
    return "\n\n".join(lines)


def _format_steps(workflow: Workflow) -> str:
    if not workflow.steps:
        return "(no grounded steps)"
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(workflow.steps))


def _format_mock_data(workflow: Workflow) -> str:
    read_tools = [t for t in workflow.tools if "_data" in t.tool_name.lower()
                  or any(k in t.tool_name.lower() for k in ("get", "fetch", "query", "list", "read", "directory"))]
    if not read_tools:
        return "(none)"
    return "\n\n".join(
        f"Tool: {t.tool_name}\n{t.response_template[:1500]}"
        for t in read_tools
    )


# ── Deterministic health checks ───────────────────────────────────────────────

def _health_check_event(
    event: Event,
    all_base_events: list[Event],
    workflow: Workflow,
) -> list[str]:
    issues: list[str]  = []
    object_ids = {o.object_id for o in workflow.objects}

    if not event.input.strip():
        issues.append("input is empty")
    if not event.source.strip():
        issues.append("source is empty")
    if event.recipient not in object_ids:
        issues.append(f"recipient '{event.recipient}' not found in workflow.objects")
    if event.expect is not None and not (event.expect.action or "").strip():
        issues.append("expect.action is empty string (use null instead)")

    # Exact-duplicate input check
    others = [e for e in all_base_events if e.id != event.id]
    if any(e.input.strip() == event.input.strip() for e in others):
        issues.append("input text is identical to another base event")

    return issues


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _validate_workflow_events(
    llm,
    workflow: Workflow,
    prompt_template: str,
) -> EventSequenceValidation:
    base_events = [e for e in workflow.events if e.role == "base"]

    if not base_events:
        return EventSequenceValidation(
            workflow_id=workflow.id,
            n_base_events=0,
            event_verdicts=[],
            sequence_verdict="INCOMPLETE",
            sequence_issues=["workflow has no base events"],
            aggregate_health="ISSUES",
            aggregate_quality="POOR",
        )

    # Deterministic health pass
    health_map: dict[str, list[str]] = {
        e.id: _health_check_event(e, base_events, workflow)
        for e in base_events
    }

    # LLM judge
    prompt = (
        prompt_template
        .replace("{WORKFLOW_ID}", workflow.id)
        .replace("{WORKFLOW_NAME}", workflow.name)
        .replace("{GROUNDED_STEPS}", _format_steps(workflow))
        .replace("{MOCK_DATA}", _format_mock_data(workflow))
        .replace("{BASE_EVENTS}", _format_events(base_events))
    )
    expected_ids = {e.id for e in base_events}
    judgement: Optional[EventSequenceJudgement] = generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=EventSequenceJudgement,
        item_id=f"{workflow.id}-events",
        validator=lambda r: (
            r.sequence_verdict in ("CLEAN", "MILD_ISSUES", "PARADOX", "INCOMPLETE")
            and all(v.event_id in expected_ids for v in r.event_verdicts)
        ),
    )

    if judgement is None:
        event_verdicts = [
            EventVerdict(
                workflow_id=workflow.id,
                event_id=e.id,
                event_input_preview=e.input[:100],
                issues=health_map[e.id],
                issue_descriptions=[],
                quality="POOR",
            )
            for e in base_events
        ]
        return EventSequenceValidation(
            workflow_id=workflow.id,
            n_base_events=len(base_events),
            event_verdicts=event_verdicts,
            sequence_verdict="INCOMPLETE",
            sequence_issues=["(judge failed)"],
            sequence_reasoning="LLM judge failed to produce a verdict.",
            aggregate_health="ISSUES" if any(health_map.values()) else "OK",
            aggregate_quality="POOR",
        )

    verdict_by_id = {v.event_id: v for v in judgement.event_verdicts}
    event_verdicts: list[EventVerdict] = []
    for e in base_events:
        llm_v: Optional[EventVerdictOutput] = verdict_by_id.get(e.id)
        merged_issues = list(health_map[e.id])
        merged_descs: list[str] = []
        if llm_v:
            merged_issues += llm_v.issues
            merged_descs += llm_v.issue_descriptions
        event_verdicts.append(EventVerdict(
            workflow_id=workflow.id,
            event_id=e.id,
            event_input_preview=e.input[:100],
            issues=merged_issues,
            issue_descriptions=merged_descs,
            quality=(llm_v.quality if llm_v else "POOR"),
        ))

    agg_health = "ISSUES" if any(health_map.values()) else "OK"
    qualities = [v.quality for v in event_verdicts]
    if any(q == "POOR" for q in qualities):
        agg_quality = "POOR"
    elif any(q == "ADEQUATE" for q in qualities):
        agg_quality = "ADEQUATE"
    else:
        agg_quality = "GOOD"

    return EventSequenceValidation(
        workflow_id=workflow.id,
        n_base_events=len(base_events),
        event_verdicts=event_verdicts,
        sequence_verdict=judgement.sequence_verdict,
        sequence_issues=list(judgement.sequence_issues or []),
        sequence_reasoning=judgement.reasoning,
        aggregate_health=agg_health,
        aggregate_quality=agg_quality,
        judge_input_tokens=getattr(judgement, "_input_tokens", 0),
        judge_output_tokens=getattr(judgement, "_output_tokens", 0),
    )


# ── Summary + CLI ─────────────────────────────────────────────────────────────

def _print_summary(results: list[EventSequenceValidation]) -> None:
    verdict_counts = Counter(r.sequence_verdict for r in results)
    health_counts = Counter(r.aggregate_health for r in results)
    quality_counts = Counter(r.aggregate_quality for r in results)
    issue_counts: Counter = Counter()
    for r in results:
        for v in r.event_verdicts:
            for issue in v.issues:
                issue_counts[issue] += 1

    print("\n" + "=" * 70)
    print(f"Event sequence validation — {len(results)} workflows")
    print("=" * 70)
    print("Sequence verdict (LLM):")
    for verdict in ("CLEAN", "MILD_ISSUES", "PARADOX", "INCOMPLETE"):
        print(f"  {verdict:<15} {verdict_counts.get(verdict, 0):3d}")
    print("Aggregate health / quality:")
    print(f"  Health OK / ISSUES:  {health_counts.get('OK', 0)} / {health_counts.get('ISSUES', 0)}")
    for q in ("GOOD", "ADEQUATE", "POOR"):
        print(f"  Quality {q:<8}:  {quality_counts.get(q, 0):3d}")
    if issue_counts:
        print("Issue codes (across all events):")
        for code, n in issue_counts.most_common():
            print(f"  {code:<30} {n:3d}")
    print()

    flagged = [r for r in results if r.sequence_verdict in ("PARADOX", "INCOMPLETE")]
    if flagged:
        print(f"PARADOX / INCOMPLETE workflows ({len(flagged)}):")
        for r in flagged:
            paradox_evts = [v.event_id for v in r.event_verdicts if "sequential_paradox" in v.issues]
            print(f"  {r.workflow_id:<55} verdict={r.sequence_verdict}  paradox_on={paradox_evts or '-'}")
        print()


def main_with_args(args: argparse.Namespace) -> int:
    workflows: list[Workflow] = load_jsonl(args.input, Workflow)
    prompt_template = _load_prompt()

    if getattr(args, "filter", None):
        ids = set(args.filter)
        workflows = [w for w in workflows if w.id in ids]
    if getattr(args, "limit", None):
        workflows = workflows[: args.limit]

    if not workflows:
        print("No workflows to validate.", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if getattr(args, "output", None)
        else Path(args.input).with_name(Path(args.input).stem + "__event_validation.jsonl")
    )

    provider = getattr(args, "provider", None) or os.environ.get("LLM_PROVIDER", "azure")
    judge_model = getattr(args, "judge_model", None) or "gpt-5.4"
    workers = getattr(args, "workers", 4)
    llm = create_llm(provider=provider, model=judge_model, temperature=0.0)

    results: list[EventSequenceValidation] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_validate_workflow_events, llm, w, prompt_template): w
            for w in workflows
        }
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  ✗ {w.id}: judge crashed — {exc}", file=sys.stderr)
                continue
            results.append(result)
            marker = {"CLEAN": "✓", "MILD_ISSUES": "·", "PARADOX": "✗", "INCOMPLETE": "!"}.get(
                result.sequence_verdict, "?"
            )
            print(f"  {marker} {result.workflow_id:<55} {result.sequence_verdict}")

    results.sort(key=lambda r: r.workflow_id)

    with open(output_path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    _print_summary(results)
    print(f"Wrote {len(results)} validation records to {output_path}")

    no_fail = getattr(args, "no_fail", False)
    if any(r.sequence_verdict in ("PARADOX", "INCOMPLETE") for r in results) and not no_fail:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to workflows JSONL (Workflow records).")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "azure"))
    parser.add_argument("--judge-model", dest="judge_model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Workflow IDs to process (default: all).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-fail", dest="no_fail", action="store_true",
                        help="Exit 0 even if PARADOX/INCOMPLETE workflows are found.")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
