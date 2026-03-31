"""
Stage 3: Generate seed data (mock tools) and expectations jointly.

Reads test cases (from Stage 2) and their matching samples (from Stage 1),
then for each test case:
  1. Generates mock tool data using all event inputs + step texts as context —
     covering every entity that will appear at eval time in one consistent pass.
  2. Writes event expectations using that exact mock data.

Because mock data and expectations are generated with full knowledge of all events
(not just the original sample steps), no refresh or augmentation is needed later.

Usage:
    python -m src.data.generate_seed \\
        --input outputs/my-run/test_cases.jsonl \\
        --samples outputs/my-run/samples.jsonl

    # Force regeneration of all test cases
    python -m src.data.generate_seed \\
        --input outputs/my-run/test_cases.jsonl \\
        --samples outputs/my-run/samples.jsonl --force

    # Write to a separate output file (non-destructive)
    python -m src.data.generate_seed \\
        --input outputs/my-run/test_cases.jsonl \\
        --samples outputs/my-run/samples.jsonl \\
        -o outputs/my-run/test_cases_seeded.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.data.schema import Sample, TestCase
from src.data.generate_samples import _generate_mock_tool_data, _DATA_TOOL_RE
from src.data.generate_test_cases import _rewrite_event_expectations
from src.data.llm import create_llm
from src.data.utils import (
    add_common_args,
    infer_provider,
    print_run_info,
)


def generate_seed(llm, test_case: TestCase, sample: Sample) -> None:
    """Generate mock_tools and expectations for a test case in one consistent pass.

    Collects all entities from event inputs AND step texts before generating mock data
    so no refresh is needed later. Expectations are written using that exact mock data.
    Mutates test_case in place.
    """
    step_texts = [s.text for s in sample.steps if s.text]
    event_texts = [e.input for e in test_case.events if getattr(e, "input", None)]
    all_texts = step_texts + event_texts

    mock_tools = []
    for obj in sample.objects:
        match = _DATA_TOOL_RE.search(obj.behavior or "")
        if not match:
            continue
        tool_name = match.group(1)
        description = (obj.state_description or "").strip() or obj.role
        tool = _generate_mock_tool_data(llm, tool_name, description, all_texts)
        if tool:
            mock_tools.append(tool)
    test_case.mock_tools = mock_tools

    _rewrite_event_expectations(llm, test_case, sample)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate seed data and expectations for test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.generate_seed \\
      --input outputs/my-run/test_cases.jsonl \\
      --samples outputs/my-run/samples.jsonl

  python -m src.data.generate_seed \\
      --input outputs/my-run/test_cases.jsonl \\
      --samples outputs/my-run/samples.jsonl --force
""",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to test cases JSONL file (output from Stage 2)",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to samples JSONL file (output from Stage 1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSONL path (default: overwrites input file in place)",
    )
    add_common_args(parser)
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.provider is None:
        args.provider = infer_provider(args.model)

    output_path = args.output if args.output else args.input

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.samples.exists():
        print(f"Error: samples file not found: {args.samples}", file=sys.stderr)
        sys.exit(1)

    # Load samples indexed by id
    samples_by_id: dict[str, Sample] = {}
    with open(args.samples) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = Sample.model_validate_json(line)
            samples_by_id[s.id] = s

    # Load test cases
    test_cases: list[TestCase] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_cases.append(TestCase.model_validate_json(line))

    if args.limit:
        test_cases = test_cases[: args.limit]

    # Determine which test cases need seeding
    def _needs_seed(tc: TestCase) -> bool:
        if args.force:
            return True
        has_mock = bool(tc.mock_tools)
        has_expect = all(e.expect is not None for e in tc.events if hasattr(e, "expect"))
        return not (has_mock and has_expect)

    pending = [tc for tc in test_cases if _needs_seed(tc)]
    already_done = len(test_cases) - len(pending)

    if not pending:
        print("All test cases already seeded. Use --force to regenerate.")
        return output_path

    if already_done:
        print(f"Resuming: {already_done} already seeded, {len(pending)} remaining")
    else:
        print(f"Seeding {len(pending)} test cases")

    print_run_info(args.provider, args.model, args.seed, {})

    llm = create_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Index pending by id for fast lookup
    pending_ids = {tc.id for tc in pending}

    # Process — mutate pending test cases in place, keep others unchanged
    tc_by_id = {tc.id: tc for tc in test_cases}
    success_count = 0
    fail_count = 0

    for tc in tqdm(pending, desc="Seeding test cases"):
        # Find matching sample — test case ID is "{sample_id}-TC{NNN}"
        sample_id = tc.id.rsplit("-TC", 1)[0]
        sample = samples_by_id.get(sample_id)
        if sample is None:
            print(f"  Warning: no sample found for {tc.id} (expected sample id: {sample_id})", file=sys.stderr)
            fail_count += 1
            continue

        generate_seed(llm, tc, sample)

        if tc.mock_tools:
            success_count += 1
        else:
            fail_count += 1

    # Write all test cases (seeded + unchanged) to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for tc in test_cases:
            f.write(tc.model_dump_json() + "\n")

    print()
    print(f"Complete. Output: {output_path}")
    print(f"Seeded: {success_count} (failed: {fail_count})")
    return output_path


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
