"""
Data generation pipeline — runs both stages in sequence.

Usage:
    # Full pipeline into a target folder
    python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run

    # Continue an existing run (skips stage 1 if samples.jsonl already exists)
    python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run

    # Skip stage 1 explicitly with a specific samples file
    python -m src.data.pipeline --samples outputs/data/zapier/templates_samples_object.jsonl

    # With options
    python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run \\
        --samples-per-template 3 --scenario-count 2 --mod-type temporal
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.data import generate_samples, generate_test_cases

SAMPLES_FILENAME = "samples.jsonl"
TEST_CASES_FILENAME = "test_cases.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data generation pipeline (samples → test cases)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline into a target folder
  python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run

  # Continue an existing run (stage 1 is skipped if samples.jsonl already exists)
  python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run

  # Skip stage 1 with a specific samples file (no target-dir)
  python -m src.data.pipeline --samples outputs/data/zapier/templates_samples_object.jsonl

  # Full pipeline with custom options
  python -m src.data.pipeline -i data/zapier/raw/templates.yaml --target-dir outputs/my-run \\
      --samples-per-template 3 --scenario-count 2 --mod-type temporal --ambiguity precise
""",
    )

    # --- Output targeting ---
    parser.add_argument(
        "--target-dir", "-t",
        type=Path,
        default=None,
        help=(
            "Directory for all pipeline outputs. Stage 1 writes samples.jsonl here; "
            "stage 2 writes test_cases.jsonl here. If samples.jsonl already exists, "
            "stage 1 is skipped automatically (continuation)."
        ),
    )

    # --- Stage selection ---
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Skip stage 1 and use this specific samples JSONL file as input to stage 2",
    )

    # --- Stage 1 args ---
    stage1 = parser.add_argument_group("Stage 1: Generate Samples")
    stage1.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to raw templates YAML file (required for stage 1)",
    )
    stage1.add_argument(
        "--samples-per-template",
        type=int,
        default=1,
        help="Number of samples per template (default: 1)",
    )
    stage1.add_argument(
        "--id",
        dest="ids",
        metavar="ID",
        action="append",
        default=None,
        help="Only process template(s) with this ID (repeatable: --id foo --id bar)",
    )
    stage1.add_argument(
        "--samples-prompt-template",
        type=Path,
        default=Path("config/prompts/data-gen/generate_samples.yaml"),
        help="Prompt template for stage 1",
    )

    # --- Stage 2 args ---
    stage2 = parser.add_argument_group("Stage 2: Generate Test Cases")
    stage2.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for test cases JSONL (default: derived from samples path or target-dir)",
    )
    stage2.add_argument(
        "--scenario-count",
        type=int,
        default=1,
        help="Scenarios per modification type (default: 1)",
    )
    stage2.add_argument(
        "--events-before",
        type=int,
        default=1,
        help="Events before modification (default: 1)",
    )
    stage2.add_argument(
        "--events-after",
        type=int,
        default=2,
        help="Events after modification (default: 2)",
    )
    stage2.add_argument(
        "--events-unrelated",
        type=int,
        default=1,
        help="Events unaffected by modification (default: 1)",
    )
    stage2.add_argument(
        "--mod-type",
        type=str,
        choices=list(generate_test_cases.MODIFICATION_TYPES.keys()),
        default=None,
        help="Modification type (default: all types)",
    )
    stage2.add_argument(
        "--mods-per-scenario",
        type=int,
        default=1,
        help="Modifications per scenario (default: 1)",
    )
    stage2.add_argument(
        "--ambiguity",
        type=str,
        choices=list(generate_test_cases.AMBIGUITY_DESCRIPTIONS.keys()),
        default="random",
        help="Ambiguity level (default: random)",
    )
    stage2.add_argument(
        "--test-cases-prompt-template",
        type=Path,
        default=Path("config/prompts/data-gen/generate_test_cases.yaml"),
        help="Prompt template for stage 2",
    )

    # --- Shared args ---
    shared = parser.add_argument_group("Shared")
    shared.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider (inferred from model if not specified)",
    )
    shared.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-5-20250929",
        help="Model name (default: claude-sonnet-4-5-20250929)",
    )
    shared.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed",
    )
    shared.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    shared.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all items (both stages)",
    )
    shared.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only the first N items",
    )

    args = parser.parse_args()

    # --- Resolve target dir (auto-timestamp if not specified) ---
    if args.target_dir is None and args.samples is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.target_dir = Path("outputs/data/zapier") / timestamp
        print(f"Target directory: {args.target_dir}")

    # --- Resolve samples path and stage 1 continuation ---
    samples_path: Path | None = None
    skip_stage1 = False

    if args.samples is not None:
        # Explicit samples file provided — always skip stage 1
        samples_path = args.samples
        skip_stage1 = True
        if not samples_path.exists():
            print(f"Error: Samples file not found: {samples_path}", file=sys.stderr)
            sys.exit(1)
    elif args.target_dir is not None:
        # Target dir provided — check for existing samples (continuation)
        samples_path = args.target_dir / SAMPLES_FILENAME
        if samples_path.exists() and not args.force:
            skip_stage1 = True
            print(f"Found existing samples: {samples_path} (skipping stage 1)")
        else:
            skip_stage1 = False
    else:
        # No target dir, no explicit samples — derive path from input (original behaviour)
        if args.input is None:
            parser.error(
                "Either --input (for stage 1) or --samples / --target-dir (to skip stage 1) is required"
            )

    # Validate stage 1 inputs when stage 1 will run
    if not skip_stage1 and args.input is None:
        parser.error("--input is required to run stage 1")

    # Resolve stage 2 output path
    test_cases_output: Path | None = args.output
    if test_cases_output is None and args.target_dir is not None:
        test_cases_output = args.target_dir / TEST_CASES_FILENAME

    # --- Stage 1 ---
    if skip_stage1:
        print("=" * 60)
        print("STAGE 1: skipped (using existing samples)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("STAGE 1: Generate Samples")
        print("=" * 60)

        stage1_args = argparse.Namespace(
            input=args.input,
            output=samples_path,  # None → derived by generate_samples.run()
            prompt_template=args.samples_prompt_template,
            samples_per_template=args.samples_per_template,
            ids=args.ids,
            provider=args.provider,
            model=args.model,
            seed=args.seed,
            temperature=args.temperature,
            force=args.force,
            limit=args.limit,
        )
        samples_path = generate_samples.run(stage1_args)

    # --- Stage 2 ---
    print()
    print("=" * 60)
    print("STAGE 2: Generate Test Cases")
    print("=" * 60)

    stage2_args = argparse.Namespace(
        input=samples_path,
        output=test_cases_output,  # None → derived by generate_test_cases.run()
        prompt_template=args.test_cases_prompt_template,
        scenario_count=args.scenario_count,
        events_before=args.events_before,
        events_after=args.events_after,
        events_unrelated=args.events_unrelated,
        mod_type=args.mod_type,
        mods_per_scenario=args.mods_per_scenario,
        ambiguity=args.ambiguity,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        temperature=args.temperature,
        force=args.force,
        limit=args.limit,
    )
    output_path = generate_test_cases.run(stage2_args)

    print()
    print("=" * 60)
    print(f"Pipeline complete. Test cases: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
