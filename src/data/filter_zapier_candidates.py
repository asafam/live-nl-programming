"""
Heuristic pre-filter for Zapier templates.

Reads the raw scraped JSONL and emits candidate records likely to represent
complex, multi-actor automations suitable for the LNL data generation pipeline.

Usage:
    python -m src.data.filter_zapier_candidates \\
        --input outputs/zapier_templates_raw.jsonl \\
        --output outputs/zapier_filtered_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

# ── Heuristic configuration ───────────────────────────────────────────────────

# Multi-step action verbs that suggest actor-to-actor interactions
MULTI_STEP_VERBS = [
    r"\broute\b",
    r"\bassign\b",
    r"\bnotif(?:y|ies|ied)\b",
    r"\bescalat\b",
    r"\bupdate\b",
    r"\bschedul\b",
    r"\bforward\b",
    r"\bsync\b",
    r"\bapprove\b",
    r"\breview\b",
    r"\btrack\b",
    r"\bmonitor\b",
    r"\bautomat\b",
    r"\btrigger\b",
    r"\brendering\b",
    r"\bprocess\b",
    r"\bcoordinat\b",
    r"\borgani[sz]\b",
    r"\bmanage\b",
]

# Patterns that indicate trivial single-step "trigger → action" automations to exclude
TRIVIAL_PATTERNS = [
    # "Send X to Y" / "Add X to Y" — simple copy operations
    r"^(?:send|add|post|save|copy|log|record)\s+.{0,40}\s+to\s+",
    # "Create X from Y" without further complexity
    r"^create\s+\w+\s+from\s+\w+\s*$",
    # Very short titles that are obviously one-step
]

# Known Zapier app/service keywords — presence of ≥ 2 indicates multi-app workflow
APP_KEYWORDS = [
    "gmail", "slack", "hubspot", "notion", "airtable", "clickup", "salesforce",
    "jira", "zendesk", "trello", "asana", "monday", "linear", "github", "gitlab",
    "google sheets", "google calendar", "google drive", "dropbox", "onedrive",
    "microsoft", "outlook", "teams", "zoom", "calendly", "stripe", "shopify",
    "typeform", "formstack", "gravity forms", "webflow", "mailchimp", "sendgrid",
    "twilio", "intercom", "freshdesk", "servicenow", "pagerduty", "opsgenie",
    "datadog", "segment", "mixpanel", "amplitude", "snowflake", "bigquery",
    "postgres", "mysql", "mongodb", "redis", "s3", "aws", "gcp", "azure",
    "anthropic", "openai", "chatgpt", "claude", "gong", "chorus", "salesloft",
    "outreach", "pipedrive", "copper", "close", "apollo", "zoominfo",
    "bamboohr", "workday", "gusto", "rippling", "lever", "greenhouse", "ashby",
    "docusign", "hellosign", "pandadoc", "notion", "coda", "confluence",
    "fathom", "otter", "fireflies", "loom", "zapier tables", "webhooks",
    "discord", "telegram", "whatsapp", "sms", "twilio",
]

_APP_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in APP_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_TRIVIAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRIVIAL_PATTERNS]
_MULTI_STEP = re.compile("|".join(MULTI_STEP_VERBS), re.IGNORECASE)


def count_apps(text: str) -> int:
    """Count distinct app mentions in text."""
    matches = _APP_PATTERN.findall(text.lower())
    return len(set(matches))


def is_trivial(title: str) -> bool:
    """Return True if the title looks like a trivial single-step automation."""
    for pat in _TRIVIAL_PATTERNS:
        if pat.search(title):
            return True
    return False


def has_multi_step_verbs(text: str) -> bool:
    """Return True if the text contains multi-step action verbs."""
    return bool(_MULTI_STEP.search(text))


def has_multi_actor_signal(title: str, description: str) -> bool:
    """Return True if title or description mention ≥ 2 distinct apps."""
    combined = f"{title} {description}"
    return count_apps(combined) >= 2


def is_complex_candidate(record: dict) -> bool:
    """Return True if a record passes all heuristic filters for complexity."""
    # Skip failed fetches
    if record.get("error"):
        return False

    title = (record.get("title") or "").strip()
    description = (record.get("description") or "").strip()

    # Require a meaningful description
    if len(description) < 100:
        return False

    # Skip trivial single-step patterns
    if is_trivial(title):
        return False

    # Must mention ≥ 2 distinct services/apps
    if not has_multi_actor_signal(title, description):
        return False

    # Must contain multi-step action verbs
    combined = f"{title} {description}"
    if not has_multi_step_verbs(combined):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic filter for Zapier template candidates")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input JSONL (raw scraped templates)")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output JSONL (filtered candidates)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Stop after emitting N candidates")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0

    # Count lines for progress bar
    with open(args.input, encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=num_lines, desc="Filtering", unit="rec"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            if is_complex_candidate(record):
                fout.write(json.dumps(record) + "\n")
                passed += 1

                if args.limit and passed >= args.limit:
                    break

    print(f"\nProcessed {total:,} records → {passed:,} candidates ({100 * passed / max(total, 1):.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
