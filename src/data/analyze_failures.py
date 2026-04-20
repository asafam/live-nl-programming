"""CLI: post-hoc failure analysis of eval results files.

Usage:
  python -m src.data.analyze_failures -i path/to/results.jsonl
  python -m src.data.analyze_failures -i 'outputs/**/runs/*.jsonl'
  python -m src.data.analyze_failures -i path/to/runs/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.data.failure_analysis import analyze_file, format_report_stdout
from src.data.failure_classifier import make_classifier

load_dotenv()


def _expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in inputs:
        p = Path(item)
        candidates: list[Path] = []
        if p.is_dir():
            candidates = sorted(p.rglob("*.jsonl"))
        elif any(c in item for c in "*?["):
            candidates = [Path(x) for x in sorted(glob.glob(item, recursive=True))]
        elif p.exists():
            candidates = [p]
        else:
            print(f"warning: no files match {item!r}", file=sys.stderr)

        for c in candidates:
            s = str(c.resolve())
            if s in seen:
                continue
            # Skip our own outputs so re-running is idempotent
            if c.name.endswith(".analysis.jsonl"):
                continue
            seen.add(s)
            out.append(c)
    return out


def _is_results_file(path: Path) -> bool:
    """Cheap check: first non-empty line looks like an eval record (config, meta, or a TC result)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                return (
                    rec.get("record_type") == "run_config"
                    or rec.get("type") == "meta"
                    or ("tc_id" in rec and "events" in rec)
                )
    except Exception:
        return False
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-hoc failure analysis of eval results JSONL files.")
    parser.add_argument("-i", "--input", action="append", required=True,
                        help="Path, glob, or directory. Can be given multiple times.")
    parser.add_argument("--classifier-provider", default="openai", choices=["openai", "anthropic", "none"],
                        help="Provider for the LLM classifier. Use 'none' to skip classification (only metrics).")
    parser.add_argument("--classifier-model", default=None,
                        help="Model for the classifier (defaults: gpt-4o-mini / claude-haiku-4-5-20251001).")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--combined-report", default=None,
                        help="Optional path to write a combined report across all inputs.")
    parser.add_argument("--max-classify", type=int, default=None,
                        help="Cap the number of classifier LLM calls per file. Useful for dry runs / budget caps.")
    args = parser.parse_args(argv)

    paths = _expand_inputs(args.input)
    # Filter to plausible results files
    paths = [p for p in paths if _is_results_file(p)]
    if not paths:
        print("No eval results files found.", file=sys.stderr)
        return 1

    classifier = None
    if args.classifier_provider != "none":
        classifier = make_classifier(args.classifier_provider, args.classifier_model)

    combined: dict[str, dict] = {}
    for p in paths:
        print(f"\n=== {p} ===")
        try:
            report = analyze_file(
                p,
                classifier=classifier,
                progress=not args.no_progress,
                max_classify=args.max_classify,
            )
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(format_report_stdout(report))
        combined[str(p)] = report

    if args.combined_report:
        Path(args.combined_report).write_text(json.dumps(combined, indent=2))
        print(f"\nCombined report written: {args.combined_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
