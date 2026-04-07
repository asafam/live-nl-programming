"""
LLM-powered template classifier and templates.yaml builder.

Reads enriched candidates (with raw_steps from re-scraping), classifies each
for complexity using an LLM, and appends qualifying entries to templates.yaml
in the same schema as the existing hand-curated entries.

Usage:
    python -m src.data.build_templates_yaml \\
        --input outputs/zapier_enriched_candidates.jsonl \\
        --output data/zapier/raw/templates.yaml \\
        --target 100 \\
        --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

from src.data.llm import create_llm, user_message
from src.data.utils import (
    add_common_args,
    generate_with_retries,
    infer_provider,
    load_jsonl,
    load_prompt_template,
)

_PROMPT_PATH = Path("config/prompts/data-gen/classify_complexity.yaml")


class ComplexityResult(BaseModel):
    complex: bool
    reason: str
    cleaned_steps: list[str]
    name: str


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _url_to_id(url: str) -> str:
    """Derive a template ID from its Zapier URL."""
    # e.g. https://zapier.com/templates/details/foo-bar → foo-bar
    path = url.rstrip("/").split("/")[-1]
    return _slugify(path) if path else _slugify(url)


def _classify(llm, record: dict, prompt_cfg: dict) -> Optional[ComplexityResult]:
    """Run LLM complexity classification on a single record."""
    steps_text = "\n".join(f"- {s}" for s in record.get("raw_steps", []))
    prompt = (
        prompt_cfg["prompt"]
        .replace("{TITLE}", record.get("title", ""))
        .replace("{DESCRIPTION}", record.get("description", ""))
        .replace("{STEPS}", steps_text)
    )
    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=ComplexityResult,
        item_id=record.get("url", "unknown"),
        validator=lambda r: isinstance(r.complex, bool),
    )


def _load_existing_ids(output_path: Path) -> set[str]:
    """Load IDs already present in the output YAML to avoid duplicates."""
    if not output_path.exists():
        return set()
    with open(output_path, encoding="utf-8") as f:
        existing = yaml.safe_load(f) or []
    return {entry.get("id", "") for entry in existing if isinstance(entry, dict)}


def _append_entry(output_path: Path, entry: dict) -> None:
    """Append a single template entry to the YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Format as clean YAML block
    lines = [
        f"\n- id: {entry['id']}",
        f"  name: \"{entry['name']}\"",
        f"  source_type: \"{entry['source_type']}\"",
        f"  link: {entry['link']}",
        "  raw_steps:",
    ]
    for step in entry["raw_steps"]:
        # Escape internal quotes
        escaped = step.replace('"', '\\"')
        lines.append(f'    - "{escaped}"')
    block = "\n".join(lines) + "\n"

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(block)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build templates.yaml from enriched Zapier candidates")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Enriched candidates JSONL")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output templates.yaml path")
    parser.add_argument("--target", "-t", type=int, default=100, help="Target number of new templates to add")
    add_common_args(parser)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not _PROMPT_PATH.exists():
        print(f"Error: prompt not found: {_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    provider = args.provider or infer_provider(args.model)
    llm = create_llm(provider=provider, model=args.model, temperature=args.temperature)
    prompt_cfg = load_prompt_template(_PROMPT_PATH)

    existing_ids = _load_existing_ids(args.output)
    print(f"Existing templates in output: {len(existing_ids)}")
    print(f"Target: {args.target} new templates")
    print(f"Model: {args.model} ({provider})\n")

    records = load_jsonl(args.input)
    added = 0
    skipped_duplicate = 0
    skipped_simple = 0
    failed = 0

    pbar = tqdm(records, desc="Classifying", unit="template")
    for record in pbar:
        if added >= args.target:
            break

        url = record.get("url", "")
        template_id = _url_to_id(url)

        if template_id in existing_ids:
            skipped_duplicate += 1
            continue

        result = _classify(llm, record, prompt_cfg)

        if result is None:
            failed += 1
            pbar.set_postfix(added=added, failed=failed, simple=skipped_simple)
            continue

        if not result.complex:
            skipped_simple += 1
            pbar.set_postfix(added=added, failed=failed, simple=skipped_simple)
            continue

        # Require at least 3 cleaned steps
        if len(result.cleaned_steps) < 3:
            skipped_simple += 1
            continue

        entry = {
            "id": template_id,
            "name": result.name or record.get("title", template_id),
            "source_type": "Zapier/Workflow Logic",
            "link": url,
            "raw_steps": result.cleaned_steps,
        }

        _append_entry(args.output, entry)
        existing_ids.add(template_id)
        added += 1
        pbar.set_postfix(added=added, failed=failed, simple=skipped_simple)

    pbar.close()
    print(f"\nDone: {added} templates added to {args.output}")
    print(f"  Skipped duplicates: {skipped_duplicate}")
    print(f"  Skipped (not complex): {skipped_simple}")
    print(f"  Failed (LLM error): {failed}")


if __name__ == "__main__":
    main()
