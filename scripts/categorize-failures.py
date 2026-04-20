#!/usr/bin/env python3
"""Categorize eval failures by reading the judge's `reasoning` field and
bucketing each failed event into one of several common LLM failure modes.

Usage:
  python scripts/categorize-failures.py <results.jsonl> [--examples N]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Ordered categories. First matching regex wins. Patterns are applied to the
# lowercased reasoning string. Order matters — most specific first.
CATEGORIES: list[tuple[str, list[str]]] = [
    # Write-service silent: data reached a store/table/db but the write isn't
    # evidenced in state (no row appended, no record created).
    ("write_service_silent", [
        r"no confirmation that .*(row|record|entry|document).* (was )?(created|appended|stored|written|added)",
        r"no (recorded|new)? ?(row|record|entry|write) (created|appended|stored|written|recorded)",
        r"no .*(airtable|google sheets|zapier tables?|zapier table|database|supabase|notion|google drive).*(record|row|entry)",
        r"not (stored|written|appended|recorded|saved) (in|to) (airtable|google sheets|tables?|drive|database|supabase|notion)",
        r"(airtable|google sheets|tables?|drive|database|supabase|notion).*\b(no (row|record|entry|write|append|upload)|not (recorded|stored|written))\b",
        r"no recorded (write|append|creation)",
        r"write[- ]service .* (empty|did not|no record)",
    ]),
    # Notification silent: Slack/email/message wasn't actually sent or posted.
    ("notification_silent", [
        r"no (slack|email|notification|message).*(sent|posted|delivered|triggered)",
        r"(slack|email|notification).*(not (sent|posted|delivered|triggered))",
        r"(was|is) not (posted|sent|delivered|triggered) (to|in)",
        r"no .*(slack dm|direct message|email) (to|was)",
        r"no recorded (email|slack|notification)",
        r"did not (send|post|deliver) (the|a|an)? ?(email|slack|notification|message)",
    ]),
    # Fan-out incomplete: some peers acted, not all expected.
    ("fan_out_incomplete", [
        r"only shows .* (and|,)? ?(no|with no|but no)",
        r"only .* (and no|, no) ",
        r"with no recorded actions? (updating|to|for)",
        r"no evidence (that|of) (the )?(other|remaining|additional|second|third)",
        r"but no (message|record|write|notification|action|entry) (to|for|in|was)",
    ]),
    # Content incomplete: message sent but missing required fields/details.
    ("content_incomplete", [
        r"does not (include|contain|show|specify|reflect|match) the (required|specified|requested|exact|full)",
        r"(does|did) not include .* (date|link|field|name|value|amount|id|url|subject|body|detail)",
        r"missing (the )?(required|specified|requested) (date|link|field|name|value|detail|information)",
        r"(subject|body|title|description|content|message) .* (do(es)? not|did not) (fully )?match",
        r"only partially (reflect|include|match)",
        r"does not show the required",
        r"do not match the requested",
    ]),
    # Conditional branch not triggered (escalation, routing condition).
    ("conditional_not_triggered", [
        r"(escalation|alert|branch|condition) (was )?not (triggered|sent|taken|fired)",
        r"should have (escalated|triggered|alerted|routed|notified)",
        r"required .* (escalation|escalation message) .* (missing|not sent)",
        r"manager (escalation|message) .* (missing|not sent)",
    ]),
    # Wrong value: data is there but incorrect.
    ("wrong_content", [
        r"incorrectly set",
        r"instead of (the )?(required|expected|specified)",
        r"rather than (the )?(required|expected|specified)",
        r"(does|did) not match the (condition|expected|required)",
        r"mismatch",
        r"(wrong|incorrect) (value|name|score|id|subject|field|data|record|format)",
        r"status was updated to .* rather than",
    ]),
    # Downstream service returned an explicit error/refusal.
    ("service_error", [
        r"service explicitly (said|reported|stated)",
        r"explicitly (said|reported|stated) (it )?(could not|failed|missing)",
        r"explicitly failed",
        r"could not complete",
        r"failed due to",
        r"explicit(ly)? marked .* not (triggered|completed|done)",
    ]),
    # Action never taken (more generic than write-service_silent).
    ("action_never_taken", [
        r"no (write|update|upload|creation|post|action)",
        r"did not (act|update|create|write|upload|post)",
        r"no (recorded )?action",
        r"took no action",
        r"without (any )?(write|update|notification|action|recorded)",
    ]),
    # Pending workflow without follow-through.
    ("pending_no_completion", [
        r"workflow state is pending",
        r"is pending",
        r"only .* pending",
        r"pending (draft|approval|deduplication|request).* (no|without)",
    ]),
    # State-level hallucination or stale.
    ("state_mismatch", [
        r"was in fact .* contradict",
        r"contradicts the (requested|expected) ",
        r"stored record .* not the required",
        r"(updated|written) to .* rather than",
    ]),
]

# Pre-compile
COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (cat, [re.compile(p) for p in pats]) for cat, pats in CATEGORIES
]


def categorize(reasoning: str) -> str:
    if not reasoning:
        return "no_reasoning"
    r = reasoning.lower()
    for cat, pats in COMPILED:
        for p in pats:
            if p.search(r):
                return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="Path to eval results JSONL")
    ap.add_argument("--examples", type=int, default=3,
                    help="Number of example reasonings to print per bucket")
    ap.add_argument("--show-tc-ids", action="store_true",
                    help="Include tc_id with each example")
    ap.add_argument("--dump-other", action="store_true",
                    help="Dump ALL 'other'-bucket reasonings for pattern-tuning")
    args = ap.parse_args()

    buckets: dict[str, list[dict]] = defaultdict(list)
    total_events = 0
    failed_events = 0

    with args.results.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("record_type") is not None:
                continue
            tc_id = d.get("tc_id", "")
            for e in d.get("events", []):
                total_events += 1
                if e.get("passed") is False:
                    failed_events += 1
                    reasoning = (e.get("reasoning") or "").strip()
                    cat = categorize(reasoning)
                    buckets[cat].append({
                        "tc_id": tc_id,
                        "event_id": e.get("event_id", ""),
                        "reasoning": reasoning,
                    })

    print(f"Total events:   {total_events}")
    print(f"Failed events:  {failed_events}")
    if total_events:
        print(f"Pass rate:      {(total_events - failed_events) / total_events:.3f}")
    print()
    print("Failure distribution:")
    print(f"  {'bucket':<28} {'count':>6}  {'%':>6}")
    ordered = sorted(buckets.items(), key=lambda x: -len(x[1]))
    for cat, items in ordered:
        pct = 100 * len(items) / failed_events if failed_events else 0
        print(f"  {cat:<28} {len(items):>6}  {pct:>5.1f}%")
    print()

    if args.dump_other and "other" in buckets:
        print("--- All 'other' bucket reasonings ---")
        for it in buckets["other"]:
            print(f"- {it['reasoning']}")
            print()
        return

    top_n = min(6, len(ordered))
    print(f"--- Examples (top {top_n} buckets, up to {args.examples} per bucket) ---")
    for cat, items in ordered[:top_n]:
        print(f"\n### {cat}  ({len(items)})")
        for i, it in enumerate(items[: args.examples]):
            tag = f"[{it['tc_id']}] " if args.show_tc_ids else ""
            print(f"  {i+1}. {tag}{it['reasoning']}")


if __name__ == "__main__":
    main()
