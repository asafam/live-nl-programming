"""Stage 2'' validator: grade the full event sequence of each Sample.

Reuses the validate_workflow_events approach but extends it to Samples: checks
the FULL event sequence (base + pre_mod + post_mod + irrelevant) for paradoxes,
expect leakage, causal orphans, AND modification-effect consistency.

Issue codes:
  Base codes (same as validate_workflow_events):
    sequential_paradox, causal_orphan, expect_leak, expect_incomplete,
    expect_null_invalid, redundant
  Mod-specific codes:
    mod_effect_not_reflected — post_mod event's expect doesn't show mod's effect
    mod_unsuppression_invalid — [suppressed by Mxxx] used incorrectly

Usage:
    python -m src.data.validate_sample_events \\
        --samples outputs/.../workflows-mods.jsonl \\
        --provider azure --judge-model gpt-5.4 \\
        --workers 4 \\
        [--filter <sample_id> ...] [--output verdicts.jsonl] [--no-fail]
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
    EventVerdict,
    EventVerdictOutput,
    Sample,
    SampleEventSequenceValidation,
)
from src.data.utils import generate_with_retries, load_jsonl

load_dotenv()

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent
    / "config" / "prompts" / "data-gen" / "validate_sample_events.yaml"
)


def _load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["prompt"]


_WHEN_RE = None
def _when_sort_key(when: str) -> tuple:
    """Sort key for `Wnn-NTHH:MM` timestamps."""
    try:
        w, rest = when.split("-", 1)
        d, hm = rest.split("T", 1)
        h, m = hm.split(":", 1)
        return (int(w[1:]), int(d), int(h), int(m))
    except Exception:
        return (0, 0, 0, 0)


def _format_events(events: list[Event]) -> str:
    sorted_events = sorted(events, key=lambda e: _when_sort_key(e.when))
    lines = []
    for e in sorted_events:
        expect_str = e.expect.action if e.expect else "null"
        after = ",".join(e.after_mod_ids) if e.after_mod_ids else "-"
        lines.append(
            f"{e.id}  role={e.role}  when={e.when}  after_mod={after}\n"
            f"  source: {e.source}\n"
            f"  recipient: {e.recipient}\n"
            f"  input:  {e.input}\n"
            f"  expect: {expect_str}"
        )
    return "\n\n".join(lines)


def _format_modifications(sample: Sample) -> str:
    if not sample.modifications:
        return "(no modifications)"
    lines = []
    for m in sample.modifications:
        lines.append(
            f"{m.id}  when={m.when}  target={m.target}  "
            f"type={m.mod_type}  ambiguity={m.ambiguity}\n"
            f"  intent: {m.intent}"
        )
    return "\n\n".join(lines)


def _format_steps(sample: Sample) -> str:
    if not sample.steps:
        return "(no grounded steps)"
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sample.steps))


def _format_mock_data(sample: Sample) -> str:
    read_tools = [t for t in sample.tools if "_data" in t.tool_name.lower()
                  or any(k in t.tool_name.lower() for k in ("get", "fetch", "query", "list", "read", "directory"))]
    if not read_tools:
        return "(none)"
    return "\n\n".join(
        f"Tool: {t.tool_name}\n{t.response_template[:1500]}"
        for t in read_tools
    )


# ── Deterministic health checks ───────────────────────────────────────────────

def _health_check_event(event: Event, all_events: list[Event], sample: Sample) -> list[str]:
    issues: list[str] = []
    object_ids = {o.object_id for o in sample.objects}
    mod_ids = {m.id for m in sample.modifications}

    if not event.input.strip():
        issues.append("input is empty")
    if not event.source.strip():
        issues.append("source is empty")
    if event.recipient not in object_ids:
        issues.append(f"recipient '{event.recipient}' not in sample.objects")
    if event.expect is not None and not (event.expect.action or "").strip():
        issues.append("expect.action is empty string (use null instead)")
    if event.after_mod_ids:
        bad_refs = [m for m in event.after_mod_ids if m not in mod_ids]
        if bad_refs:
            issues.append(f"after_mod_ids references unknown mods: {bad_refs}")

    # Exact-duplicate input check
    others = [e for e in all_events if e.id != event.id]
    if any(e.input.strip() == event.input.strip() for e in others):
        issues.append("input text is identical to another event")

    return issues


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _validate_sample_events(
    llm,
    sample: Sample,
    prompt_template: str,
) -> SampleEventSequenceValidation:
    events = list(sample.events)

    if not events:
        return SampleEventSequenceValidation(
            sample_id=sample.id,
            workflow_id=sample.sample_id or "",
            n_events=0,
            event_verdicts=[],
            sequence_verdict="INCOMPLETE",
            sequence_issues=["sample has no events"],
            aggregate_health="ISSUES",
            aggregate_quality="POOR",
        )

    health_map: dict[str, list[str]] = {
        e.id: _health_check_event(e, events, sample) for e in events
    }

    prompt = (
        prompt_template
        .replace("{SAMPLE_ID}", sample.id)
        .replace("{SAMPLE_NAME}", sample.name)
        .replace("{GROUNDED_STEPS}", _format_steps(sample))
        .replace("{MOCK_DATA}", _format_mock_data(sample))
        .replace("{MODIFICATIONS}", _format_modifications(sample))
        .replace("{EVENTS}", _format_events(events))
    )
    expected_ids = {e.id for e in events}
    judgement: Optional[EventSequenceJudgement] = generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=EventSequenceJudgement,
        item_id=f"{sample.id}-events",
        validator=lambda r: (
            r.sequence_verdict in ("CLEAN", "MILD_ISSUES", "PARADOX", "INCOMPLETE")
            and all(v.event_id in expected_ids for v in r.event_verdicts)
        ),
    )

    if judgement is None:
        event_verdicts = [
            EventVerdict(
                workflow_id=sample.id,
                event_id=e.id,
                event_input_preview=e.input[:100],
                issues=health_map[e.id],
                issue_descriptions=[],
                quality="POOR",
            )
            for e in events
        ]
        return SampleEventSequenceValidation(
            sample_id=sample.id,
            workflow_id=sample.sample_id or "",
            n_events=len(events),
            event_verdicts=event_verdicts,
            sequence_verdict="INCOMPLETE",
            sequence_issues=["(judge failed)"],
            sequence_reasoning="LLM judge failed to produce a verdict.",
            aggregate_health="ISSUES" if any(health_map.values()) else "OK",
            aggregate_quality="POOR",
        )

    verdict_by_id = {v.event_id: v for v in judgement.event_verdicts}
    event_verdicts: list[EventVerdict] = []
    for e in events:
        llm_v: Optional[EventVerdictOutput] = verdict_by_id.get(e.id)
        merged_issues = list(health_map[e.id])
        merged_descs: list[str] = []
        if llm_v:
            merged_issues += llm_v.issues
            merged_descs += llm_v.issue_descriptions
        event_verdicts.append(EventVerdict(
            workflow_id=sample.id,
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

    return SampleEventSequenceValidation(
        sample_id=sample.id,
        workflow_id=sample.sample_id or "",
        n_events=len(events),
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

def _print_summary(results: list[SampleEventSequenceValidation]) -> None:
    verdict_counts = Counter(r.sequence_verdict for r in results)
    health_counts = Counter(r.aggregate_health for r in results)
    quality_counts = Counter(r.aggregate_quality for r in results)
    issue_counts: Counter = Counter()
    for r in results:
        for v in r.event_verdicts:
            for issue in v.issues:
                issue_counts[issue] += 1

    print("\n" + "=" * 70)
    print(f"Sample event sequence validation — {len(results)} samples")
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
        print(f"PARADOX / INCOMPLETE samples ({len(flagged)}):")
        for r in flagged:
            print(f"  {r.sample_id:<70} verdict={r.sequence_verdict}")
        print()


def main_with_args(args: argparse.Namespace) -> int:
    samples: list[Sample] = load_jsonl(args.samples, Sample)
    prompt_template = _load_prompt()

    if getattr(args, "filter", None):
        ids = set(args.filter)
        samples = [s for s in samples if s.id in ids]
    if getattr(args, "limit", None):
        samples = samples[: args.limit]

    if not samples:
        print("No samples to validate.", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if getattr(args, "output", None)
        else Path(args.samples).with_name(Path(args.samples).stem + "__sample_event_validation.jsonl")
    )

    provider = getattr(args, "provider", None) or os.environ.get("LLM_PROVIDER", "azure")
    judge_model = getattr(args, "judge_model", None) or "gpt-5.4"
    workers = getattr(args, "workers", 4)
    llm = create_llm(provider=provider, model=judge_model, temperature=0.0)

    results: list[SampleEventSequenceValidation] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_validate_sample_events, llm, s, prompt_template): s
            for s in samples
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  ✗ {s.id}: judge crashed — {exc}", file=sys.stderr)
                continue
            results.append(result)
            marker = {"CLEAN": "✓", "MILD_ISSUES": "·", "PARADOX": "✗", "INCOMPLETE": "!"}.get(
                result.sequence_verdict, "?"
            )
            print(f"  {marker} {result.sample_id:<70} {result.sequence_verdict}")

    results.sort(key=lambda r: r.sample_id)

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
    parser.add_argument("--samples", required=True, type=Path,
                        help="Path to samples JSONL (Sample records).")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "azure"))
    parser.add_argument("--judge-model", dest="judge_model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Sample IDs to process (default: all).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-fail", dest="no_fail", action="store_true",
                        help="Exit 0 even if PARADOX/INCOMPLETE samples are found.")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
