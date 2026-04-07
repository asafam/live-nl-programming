"""
Validate templates.yaml entries against their source Zapier pages.

For each template (skipping hand-curated ones without a Zapier slug URL),
fetches the live page, extracts all relevant text content, then uses a
strong LLM to compare the YAML raw_steps against the page content.

Outputs a report showing which templates accurately represent their source.

Usage:
    python -m src.data.validate_templates \\
        --input data/zapier/raw/templates.yaml \\
        --model claude-sonnet-4-6 \\
        --skip N          # skip first N templates (e.g. 13 to skip hand-curated)
        --fix             # rewrite inaccurate templates in-place
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

from src.data.llm import create_llm, user_message
from src.data.utils import generate_with_retries, infer_provider

USER_AGENT = "ZapierTemplateValidator/1.0 (Educational/Research purposes)"

VALIDATION_PROMPT = """\
You are validating whether a template's `raw_steps` accurately represents the workflow described on a Zapier template page.

## YAML raw_steps (what we have)

{YAML_STEPS}

## Zapier page content (ground truth)

{PAGE_CONTENT}

## Task

Compare the YAML steps against the page content. Assess:
1. **Accuracy**: Do the steps faithfully represent the workflow on the page? Are any steps fabricated, misleading, or significantly wrong?
2. **Completeness**: Are major workflow stages missing from the YAML steps?
3. **Complexity**: Based on the page, does this automation actually involve ≥2 distinct actors with multi-step interactions?

Output:
- `valid`: true if the steps accurately represent the page content (minor wording differences are fine; outright fabrication or wrong workflow is not)
- `complex`: true if the page confirms ≥2 actors with multi-step interactions
- `issues`: list of specific problems found (empty list if none)
- `corrected_steps`: if `valid` is false, provide corrected steps based on the page content; otherwise empty list
- `summary`: one-sentence assessment
"""


class ValidationResult(BaseModel):
    valid: bool
    complex: bool
    issues: list[str]
    corrected_steps: list[str]
    summary: str


def fetch_page_content(url: str) -> Optional[str]:
    """Fetch and extract text content from a Zapier template page."""
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text
    except requests.RequestException as e:
        return None

    # Extract all meaningful text: title, meta description, headings, paragraphs, list items
    parts = []

    # Title
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        parts.append(f"Title: {m.group(1).strip()}")

    # Meta description
    m = re.search(
        r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if not m:
        m = re.search(
            r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']description["\']',
            html, re.IGNORECASE
        )
    if m:
        parts.append(f"Description: {m.group(1).strip()}")

    # All headings h1-h4
    for h in re.findall(r"<h[1-4][^>]*>(.*?)</h[1-4]>", html, re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", "", h).strip()
        if text and len(text) > 3:
            parts.append(f"Heading: {text}")

    # Paragraphs
    for p in re.findall(r"<p[^>]*>(.*?)</p>", html, re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", p)
        text = re.sub(r"&#x27;", "'", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 20:
            parts.append(text)

    # List items
    for li in re.findall(r"<li[^>]*>(.*?)</li>", html, re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", li)
        text = re.sub(r"&#x27;", "'", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 20:
            parts.append(f"• {text}")

    content = "\n".join(parts)
    # Truncate to avoid token limits
    return content[:6000] if content else None


def _rewrite_template(template: dict, corrected_steps: list[str]) -> dict:
    """Return a copy of the template with corrected raw_steps."""
    updated = dict(template)
    updated["raw_steps"] = corrected_steps
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate templates.yaml against Zapier source pages")
    parser.add_argument("--input", "-i", default="data/zapier/raw/templates.yaml", type=Path)
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6")
    parser.add_argument("--provider", "-p", default=None)
    parser.add_argument("--skip", type=int, default=13, help="Skip first N templates (default: 13 hand-curated)")
    parser.add_argument("--fix", action="store_true", help="Rewrite invalid templates in-place with corrected steps")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Validate only first N templates after skip")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        all_templates = yaml.safe_load(f)

    to_validate = all_templates[args.skip:]
    if args.limit:
        to_validate = to_validate[:args.limit]

    print(f"Validating {len(to_validate)} templates (skipping first {args.skip})")
    provider = args.provider or infer_provider(args.model)
    print(f"Model: {args.model} ({provider})\n")

    llm = create_llm(provider=provider, model=args.model, temperature=0.1)

    results = []
    fixes_needed = []

    for template in tqdm(to_validate, desc="Validating", unit="template"):
        tid = template["id"]
        url = template["link"]

        # Fetch page
        page_content = fetch_page_content(url)
        if not page_content:
            tqdm.write(f"  ⚠ FETCH_FAILED  {tid}")
            results.append({"id": tid, "status": "fetch_failed"})
            continue

        # Format prompt
        yaml_steps = "\n".join(f"- {s}" for s in template.get("raw_steps", []))
        prompt = (
            VALIDATION_PROMPT
            .replace("{YAML_STEPS}", yaml_steps)
            .replace("{PAGE_CONTENT}", page_content)
        )

        result = generate_with_retries(
            llm=llm,
            prompt=prompt,
            response_model=ValidationResult,
            item_id=tid,
            validator=lambda r: isinstance(r.valid, bool),
        )

        if result is None:
            tqdm.write(f"  ✗ LLM_FAILED    {tid}")
            results.append({"id": tid, "status": "llm_failed"})
            continue

        status_icon = "✓" if (result.valid and result.complex) else ("⚠" if result.valid else "✗")
        complexity_tag = "" if result.complex else " [NOT COMPLEX]"
        tqdm.write(f"  {status_icon} {'VALID' if result.valid else 'INVALID'}{complexity_tag:14}  {tid}")
        tqdm.write(f"    {result.summary}")
        if result.issues:
            for issue in result.issues:
                tqdm.write(f"    • {issue}")

        results.append({
            "id": tid,
            "status": "valid" if result.valid else "invalid",
            "complex": result.complex,
            "issues": result.issues,
            "summary": result.summary,
        })

        if not result.valid and result.corrected_steps:
            fixes_needed.append((template, result.corrected_steps))

    # Summary
    print("\n" + "=" * 60)
    valid = sum(1 for r in results if r.get("status") == "valid")
    invalid = sum(1 for r in results if r.get("status") == "invalid")
    not_complex = sum(1 for r in results if r.get("status") == "valid" and not r.get("complex", True))
    failed = sum(1 for r in results if r.get("status") in ("fetch_failed", "llm_failed"))

    print(f"Results: {valid} valid, {invalid} invalid, {not_complex} not complex, {failed} failed")
    print(f"  Valid + complex: {valid - not_complex}/{len(results)}")

    if invalid > 0:
        print(f"\nInvalid templates:")
        for r in results:
            if r.get("status") == "invalid":
                print(f"  - {r['id']}: {r.get('summary', '')}")

    if not_complex > 0:
        print(f"\nValid but not complex (consider removing):")
        for r in results:
            if r.get("status") == "valid" and not r.get("complex", True):
                print(f"  - {r['id']}: {r.get('summary', '')}")

    # Apply fixes if requested
    if args.fix and fixes_needed:
        print(f"\nApplying {len(fixes_needed)} fixes to {args.input}...")
        fix_map = {t["id"]: steps for t, steps in fixes_needed}
        updated = []
        for t in all_templates:
            if t["id"] in fix_map:
                t = _rewrite_template(t, fix_map[t["id"]])
            updated.append(t)
        with open(args.input, "w", encoding="utf-8") as f:
            yaml.dump(updated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print("Done.")


if __name__ == "__main__":
    main()
