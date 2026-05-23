"""Cross-check named entities in events against mock-tool response data.

For each Sample, extract identifier-like tokens from event inputs and expects
(emails, Slack thread/user IDs, channel names, ALL-CAPS+number IDs), then verify
each token appears somewhere in the workflow's mock tool data. Reports tokens
that don't appear — those are likely out-of-sync between events and mocks.

Usage:
    python -m src.data.check_mock_event_sync \\
        --samples outputs/.../workflows-mods.jsonl \\
        --workflows outputs/.../workflows.jsonl \\
        [--filter <sample_id> ...] [--summary]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.data.schema import Sample, Workflow
from src.data.utils import load_jsonl


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SLACK_USER_RE = re.compile(r"\bU[0-9A-Z]{6,15}\b")
SLACK_THREAD_RE = re.compile(r"\bTS-[A-Za-z0-9]{4,}\b")
CHANNEL_RE = re.compile(r"#[a-z0-9_-]{2,}\b")
RECORD_ID_RE = re.compile(r"\b[A-Z]{1,5}-\d{2,6}\b")


def _extract_entities(text: str) -> set[str]:
    if not text:
        return set()
    found: set[str] = set()
    found.update(EMAIL_RE.findall(text))
    found.update(SLACK_USER_RE.findall(text))
    found.update(SLACK_THREAD_RE.findall(text))
    found.update(CHANNEL_RE.findall(text))
    found.update(RECORD_ID_RE.findall(text))
    return found


def _event_entities(sample: Sample) -> set[str]:
    bag: set[str] = set()
    for e in sample.events:
        bag |= _extract_entities(e.input)
        if e.expect:
            bag |= _extract_entities(e.expect.action or "")
    return bag


def _mock_haystack(workflow: Workflow) -> str:
    parts = [workflow.name]
    for t in workflow.tools:
        parts.append(t.response_template or "")
        for trig in t.triggers or []:
            parts.append(trig.message_template or "")
    for o in workflow.objects:
        parts.append(o.behavior or "")
        parts.append(o.state_description or "")
    return "\n".join(parts)


def _people_names_in_event(event_text: str) -> set[str]:
    """Heuristic: extract proper-noun pairs (FirstName LastName), but be cautious."""
    pattern = re.compile(r"\b([A-Z][a-z]{2,}) ([A-Z][a-z]{2,})\b")
    return {f"{a} {b}" for a, b in pattern.findall(event_text or "")}


def _sample_people(sample: Sample) -> set[str]:
    bag: set[str] = set()
    for e in sample.events:
        bag |= _people_names_in_event(e.input or "")
        if e.expect:
            bag |= _people_names_in_event(e.expect.action or "")
    # Filter common-word false-positives
    BAD = {"Account Executive", "Quote Submitted", "Slack Channel", "Status Set"}
    return {n for n in bag if n not in BAD}


def _scan_sample(sample: Sample, wf_by_id: dict[str, Workflow]) -> dict:
    wf = wf_by_id.get(sample.sample_id)
    if not wf:
        return {"sample_id": sample.id, "error": "parent workflow not found"}
    haystack = _mock_haystack(wf)

    event_entities = _event_entities(sample)
    event_people = _sample_people(sample)
    missing_entities = sorted(e for e in event_entities if e not in haystack)
    missing_people = sorted(p for p in event_people if p not in haystack)
    return {
        "sample_id": sample.id,
        "workflow_id": sample.sample_id,
        "n_event_entities": len(event_entities),
        "n_event_people": len(event_people),
        "missing_entities": missing_entities,
        "missing_people": missing_people,
    }


def main_with_args(args: argparse.Namespace) -> int:
    samples: list[Sample] = load_jsonl(args.samples, Sample)
    workflows: list[Workflow] = load_jsonl(args.workflows, Workflow)
    wf_by_id = {w.id: w for w in workflows}

    if args.filter:
        ids = set(args.filter)
        samples = [s for s in samples if s.id in ids]

    results = [_scan_sample(s, wf_by_id) for s in samples]
    issues = [r for r in results if r.get("missing_entities") or r.get("missing_people")]

    print(f"Sync-check across {len(samples)} samples — {len(issues)} have mismatches.")
    print()

    if args.summary:
        # roll up by entity
        entity_count: dict[str, int] = defaultdict(int)
        person_count: dict[str, int] = defaultdict(int)
        for r in issues:
            for e in r["missing_entities"]:
                entity_count[e] += 1
            for p in r["missing_people"]:
                person_count[p] += 1
        if entity_count:
            print("Missing IDs/emails/channels by frequency:")
            for e, n in sorted(entity_count.items(), key=lambda x: -x[1])[:30]:
                print(f"  {n:3d}  {e}")
            print()
        if person_count:
            print("Missing people-name candidates by frequency:")
            for p, n in sorted(person_count.items(), key=lambda x: -x[1])[:30]:
                print(f"  {n:3d}  {p}")
            print()

    if not args.summary:
        for r in issues[:50]:
            print(f"\n[{r['sample_id']}] workflow={r['workflow_id']}")
            if r["missing_entities"]:
                print(f"  missing entities: {r['missing_entities'][:10]}")
            if r["missing_people"]:
                print(f"  missing people:   {r['missing_people'][:10]}")
        if len(issues) > 50:
            print(f"\n... and {len(issues) - 50} more samples with mismatches")

    return 1 if issues else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--workflows", required=True, type=Path)
    parser.add_argument("--filter", nargs="+", default=None)
    parser.add_argument("--summary", action="store_true", help="Aggregate by entity instead of per-sample dump")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
