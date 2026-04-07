"""
Validate NoviCode personal assistant samples against their seed utterances.

For each sample, uses an LLM judge to evaluate 4 dimensions:
  1. Faithfulness  — steps trace back to the seed utterance
  2. Naturalness   — steps read like a real assistant (no API names, codes, IDs)
  3. Actor validity — declared actors are correct; no undeclared actors in steps
  4. Flow quality  — clear multi-step actor interaction (≥5 steps, coherent narrative)

Usage:
    python -m src.data.validate_novicode \\
        --input data/novicode/raw/personal_assistant_samples.yaml \\
        --model claude-opus-4-6 \\
        [--fix]           # rewrite invalid samples in-place with corrected_steps
        [--limit N]       # validate only first N samples
        [--ids ID ...]    # validate specific sample IDs only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

from src.data.llm import create_llm
from src.data.utils import generate_with_retries, infer_provider

VALIDATION_PROMPT = """\
You are evaluating a NoviCode personal assistant workflow sample.
The seed utterance is the original NL request from a real user (ground truth).
The raw_steps are an expanded workflow generated to illustrate how the assistant handles it.

## Seed Utterance (ground truth — original user request)
{SEED}

## Workflow Name
{NAME}

## Declared Actors
{ACTORS}

## Raw Steps
{STEPS}

---

Evaluate on 4 dimensions:

1. FAITHFULNESS: Do the steps trace back to what the seed actually says or strongly implies?
   - Flag if steps introduce concepts entirely absent from the seed (unsolicited features, wrong names, fabricated events, invented medical context).
   - Minor reasonable inferences (a specific coffee shop for a "lunch meeting" seed) are acceptable.

2. NATURALNESS: Do the steps read like a real personal assistant response (Siri, Alexa, Google Assistant)?
   - Flag API names ("OpenWeatherMap API", "Weather.gov"), numeric condition codes (600-622, SKC), GPS coordinates, zip codes, device IDs, VIN numbers, or other implementation mechanics.
   - Flag overly technical language the user would never see.
   - Flag robotic step descriptions that sound like software documentation.

3. ACTOR VALIDITY: Are the declared actors the right domain agents for this task?
   - Valid actors: WeatherAgent, ClockAgent, CalendarAgent, RemindersAgent, MessagingAgent, NavigationAgent, MusicAgent, ShoppingAgent, SmartHomeAgent, EventsAgent.
   - Flag if an actor appears in the steps but is NOT in the declared actors list.
   - Flag if an actor is declared but never used in any step.
   - Flag if ShoppingAgent is used for event/concert/movie tickets (should be EventsAgent).

4. FLOW QUALITY: Is there a clear multi-step interaction between actors?
   - Flag if fewer than 5 meaningful steps exist.
   - Flag if steps don't form a coherent narrative (disconnected actions, steps out of order).
   - Flag if there's no meaningful actor handoff (single agent doing everything).

Output:
- `valid`: true only if ALL 4 dimensions pass
- `faithfulness_ok`: true if faithfulness passes
- `naturalness_ok`: true if naturalness passes
- `actor_validity_ok`: true if actor validity passes
- `flow_quality_ok`: true if flow quality passes
- `issues`: list of specific problems found (empty if valid)
- `corrected_steps`: if NOT valid, provide corrected raw_steps that fix all issues; else empty list
- `summary`: one-sentence assessment
"""


class NoviCodeValidationResult(BaseModel):
    valid: bool
    faithfulness_ok: bool
    naturalness_ok: bool
    actor_validity_ok: bool
    flow_quality_ok: bool
    issues: list[str]
    corrected_steps: list[str]
    summary: str


def _format_actors(actors: list[dict]) -> str:
    return "\n".join(f"- {a['name']} (domain: {a.get('domain', '?')})" for a in actors)


def _format_steps(steps: list[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))


def validate_sample(llm, sample: dict) -> NoviCodeValidationResult | None:
    actors_text = _format_actors(sample.get("actors", []))
    steps_text = _format_steps(sample.get("raw_steps", []))
    prompt = (
        VALIDATION_PROMPT
        .replace("{SEED}", sample.get("seed_utterance", "(none)"))
        .replace("{NAME}", sample.get("name", ""))
        .replace("{ACTORS}", actors_text)
        .replace("{STEPS}", steps_text)
    )
    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=NoviCodeValidationResult,
        item_id=sample["id"],
        validator=lambda r: isinstance(r.valid, bool),
    )


def _dim_tag(ok: bool) -> str:
    return "✓" if ok else "✗"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NoviCode personal assistant samples against their seed utterances"
    )
    parser.add_argument("--input", "-i", default="data/novicode/raw/personal_assistant_samples.yaml", type=Path)
    parser.add_argument("--model", "-m", default="claude-opus-4-6")
    parser.add_argument("--provider", "-p", default=None)
    parser.add_argument("--fix", action="store_true", help="Rewrite invalid samples in-place with corrected_steps")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Validate only first N samples")
    parser.add_argument("--ids", nargs="+", default=None, help="Validate only samples with these IDs")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        all_samples = yaml.safe_load(f)

    to_validate = all_samples
    if args.ids:
        id_set = set(args.ids)
        to_validate = [s for s in all_samples if s["id"] in id_set]
    if args.limit:
        to_validate = to_validate[: args.limit]

    provider = args.provider or infer_provider(args.model)
    print(f"Validating {len(to_validate)} samples")
    print(f"Model: {args.model} ({provider})\n")

    llm = create_llm(provider=provider, model=args.model, temperature=0.1)

    results = []
    fixes_needed = []

    for sample in tqdm(to_validate, desc="Validating", unit="sample"):
        sid = sample["id"]
        result = validate_sample(llm, sample)

        if result is None:
            tqdm.write(f"  ✗ LLM_FAILED  {sid}")
            results.append({"id": sid, "status": "llm_failed"})
            continue

        status_icon = "✓" if result.valid else "✗"
        dims = (
            f"F:{_dim_tag(result.faithfulness_ok)} "
            f"N:{_dim_tag(result.naturalness_ok)} "
            f"A:{_dim_tag(result.actor_validity_ok)} "
            f"Q:{_dim_tag(result.flow_quality_ok)}"
        )
        tqdm.write(f"  {status_icon} [{dims}]  {sid}")
        tqdm.write(f"    {result.summary}")
        if result.issues:
            for issue in result.issues:
                tqdm.write(f"    • {issue}")

        results.append({
            "id": sid,
            "status": "valid" if result.valid else "invalid",
            "faithfulness_ok": result.faithfulness_ok,
            "naturalness_ok": result.naturalness_ok,
            "actor_validity_ok": result.actor_validity_ok,
            "flow_quality_ok": result.flow_quality_ok,
            "issues": result.issues,
            "summary": result.summary,
        })

        if not result.valid and result.corrected_steps:
            fixes_needed.append((sample, result.corrected_steps))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    valid_count = sum(1 for r in results if r.get("status") == "valid")
    invalid_count = sum(1 for r in results if r.get("status") == "invalid")
    failed_count = sum(1 for r in results if r.get("status") == "llm_failed")

    dim_fails = {
        "faithfulness": sum(1 for r in results if r.get("status") == "invalid" and not r.get("faithfulness_ok", True)),
        "naturalness": sum(1 for r in results if r.get("status") == "invalid" and not r.get("naturalness_ok", True)),
        "actor_validity": sum(1 for r in results if r.get("status") == "invalid" and not r.get("actor_validity_ok", True)),
        "flow_quality": sum(1 for r in results if r.get("status") == "invalid" and not r.get("flow_quality_ok", True)),
    }

    print(f"Results: {valid_count} valid, {invalid_count} invalid, {failed_count} failed  (total: {len(results)})")
    print(f"\nFailing dimension breakdown (among {invalid_count} invalid samples):")
    for dim, count in dim_fails.items():
        print(f"  {dim:20s}: {count}")

    if invalid_count > 0:
        print(f"\nInvalid samples:")
        for r in results:
            if r.get("status") == "invalid":
                print(f"  - {r['id']}: {r.get('summary', '')}")

    # ── Apply fixes ───────────────────────────────────────────────────────────
    if args.fix and fixes_needed:
        print(f"\nApplying {len(fixes_needed)} fixes to {args.input}...")
        fix_map = {s["id"]: steps for s, steps in fixes_needed}
        updated = []
        for s in all_samples:
            if s["id"] in fix_map:
                s = dict(s)
                s["raw_steps"] = fix_map[s["id"]]
            updated.append(s)
        with open(args.input, "w", encoding="utf-8") as f:
            yaml.dump(updated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print("Done.")


if __name__ == "__main__":
    main()
