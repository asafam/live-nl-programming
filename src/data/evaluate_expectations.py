"""
Semantic evaluation of test case event expectations using an LLM judge.

For each event with a stored expectation, the judge independently derives
the correct terminal output from the object system + mock data + modifications,
then compares it to the stored expectation.  Flags: wrong, uncertain, correct.

Usage:
    python -m src.data.evaluate_expectations \\
        --input outputs/data/zapier/ITER4/test_cases.jsonl \\
        --output outputs/data/zapier/ITER4/expectation_audit.jsonl \\
        --model claude-sonnet-4-6 \\
        --workers 8

    # Repair wrong expectations in-place after auditing:
    python -m src.data.evaluate_expectations \\
        --input outputs/data/zapier/ITER4/test_cases.jsonl \\
        --repair \\
        --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

import yaml
from src.data.schema import TestCase, EventExpect
from src.data.llm import create_llm
from src.data.llm.base import ChatMessage
from src.data.utils import infer_provider, add_common_args, generate_with_retries
from src.data.generate_test_cases import _ts_key, _active_mods_for  # noqa: F401 (re-exported)


# ── Schema ────────────────────────────────────────────────────────────────────

class ExpectationVerdict(BaseModel):
    event_id: str
    verdict: Literal["correct", "wrong", "uncertain"]
    issue: str = ""
    corrected_action: str = ""


class AuditResult(BaseModel):
    tc_id: str
    verdicts: list[ExpectationVerdict]

    @property
    def wrong(self) -> list[ExpectationVerdict]:
        return [v for v in self.verdicts if v.verdict == "wrong"]

    @property
    def uncertain(self) -> list[ExpectationVerdict]:
        return [v for v in self.verdicts if v.verdict == "uncertain"]


# ── Prompt building ───────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "data-gen" / "evaluate_expectations.yaml"


def _load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["prompt"]


def _build_context(tc: TestCase) -> tuple[str, str, str]:
    """Return (obj_lines, mock_lines, mod_lines) shared across events."""
    obj_lines = "\n\n".join(
        f"[{o.object_id}]\nRole: {o.role}\nBehavior: {o.behavior}"
        for o in tc.objects
    )
    mock_lines = "\n\n".join(
        f"Tool: {t.tool_name}\n{t.response_template[:3000]}"
        for t in tc.mock_tools
    ) or "(none)"
    mod_lines = "\n".join(
        f"{m.id} at {m.when} → {m.target}: {m.intent}"
        for m in tc.modifications
    ) or "(none)"
    return obj_lines, mock_lines, mod_lines


def _evaluate_step(llm, tc: TestCase, step_idx: int, prompt_template: str) -> ExpectationVerdict | None:
    """Evaluate a single step's expectation. Returns None if no expectation stored."""
    step = tc.steps[step_idx]
    if step.expect is None:
        return None

    obj_lines, mock_lines, mod_lines = _build_context(tc)

    # Steps run before any modifications — always baseline behavior
    event_line = f"S{step_idx+1:03d} | baseline (no modifications) | recipient={step.target} | input: {step.text}"

    prompt = (
        prompt_template
        .replace("{OBJECTS}", obj_lines)
        .replace("{MOCK_DATA}", mock_lines)
        .replace("{MODIFICATIONS}", "(none — steps run before modifications)")
        .replace("{EVENT}", event_line)
        .replace("{ACTION}", step.expect.action)
        .replace("{REASON}", step.expect.reason or "")
    )

    result = generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=ExpectationVerdict,
        item_id=f"{tc.id}-S{step_idx+1:03d}-audit",
        validator=lambda r: r.verdict in ("correct", "wrong", "uncertain"),
    )
    if result:
        result.event_id = f"S{step_idx+1:03d}"
    return result


# ── Per-TC audit ──────────────────────────────────────────────────────────────

def _audit_tc(llm, tc: TestCase, prompt_template: str) -> AuditResult:
    verdicts = []
    for i in range(len(tc.steps)):
        v = _evaluate_step(llm, tc, i, prompt_template)
        if v is not None:
            verdicts.append(v)
    return AuditResult(tc_id=tc.id, verdicts=verdicts)


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_test_cases(path: Path) -> list[tuple[int, TestCase]]:
    tcs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "record_type" in d or "tc_id" in d:
                continue
            try:
                tc = TestCase.model_validate(d)
                # Only audit TCs that have at least one step with an expectation
                if any(s.expect is not None for s in tc.steps):
                    tcs.append((i, tc))
            except Exception:
                pass
    return tcs


def _load_completed(audit_path: Path) -> set[str]:
    if not audit_path.exists():
        return set()
    completed = set()
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "tc_id" in d:
                    completed.add(d["tc_id"])
            except Exception:
                pass
    return completed


# ── Repair ────────────────────────────────────────────────────────────────────

def _repair_tc(tc: TestCase, audit: AuditResult) -> bool:
    """Apply corrected actions from audit verdicts to steps. Returns True if any change made."""
    changed = False
    verdict_map = {v.event_id: v for v in audit.wrong}
    for i, step in enumerate(tc.steps):
        step_id = f"S{i+1:03d}"
        v = verdict_map.get(step_id)
        if v and v.corrected_action and step.expect is not None:
            step.expect = EventExpect(
                action=v.corrected_action,
                reason=step.expect.reason,
            )
            changed = True
    return changed


# ── Main ──────────────────────────────────────────────────────────────────────

def audit(
    input_path: Path,
    output_path: Path,
    model: str,
    provider: str,
    workers: int,
    repair: bool,
) -> None:
    tcs = _load_test_cases(input_path)
    completed = _load_completed(output_path)
    remaining = [(i, tc) for i, tc in tcs if tc.id not in completed]

    total_steps = sum(
        sum(1 for s in tc.steps if s.expect is not None)
        for _, tc in tcs
    )
    print(f"TCs to audit: {len(remaining)} remaining of {len(tcs)} total ({total_steps} steps with expectations)")

    if not remaining:
        print("All TCs already audited.")
    else:
        llm = create_llm(model=model, provider=provider)
        prompt_template = _load_prompt()

        results: list[tuple[int, AuditResult]] = []

        def _do_one(args: tuple[int, TestCase]) -> tuple[int, AuditResult]:
            i, tc = args
            return i, _audit_tc(llm, tc, prompt_template)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_one, item): item for item in remaining}
            with open(output_path, "a") as out_f:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Auditing"):
                    try:
                        i, audit_result = fut.result()
                        out_f.write(json.dumps({
                            "tc_id": audit_result.tc_id,
                            "verdicts": [v.model_dump() for v in audit_result.verdicts],
                        }) + "\n")
                        out_f.flush()
                        results.append((i, audit_result))
                    except Exception as e:
                        orig_tc = futures[fut][1]
                        print(f"  WARN: {orig_tc.id} failed: {e}", file=sys.stderr)

    # --- Summary ---
    all_results: list[AuditResult] = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "tc_id" in d:
                    all_results.append(AuditResult(
                        tc_id=d["tc_id"],
                        verdicts=[ExpectationVerdict(**v) for v in d["verdicts"]],
                    ))
            except Exception:
                pass

    total = sum(len(r.verdicts) for r in all_results)
    n_wrong = sum(len(r.wrong) for r in all_results)
    n_uncertain = sum(len(r.uncertain) for r in all_results)
    n_correct = total - n_wrong - n_uncertain

    print(f"\n=== Audit Summary ===")
    print(f"  correct:   {n_correct:4d} ({100*n_correct//total if total else 0}%)")
    print(f"  wrong:     {n_wrong:4d} ({100*n_wrong//total if total else 0}%)")
    print(f"  uncertain: {n_uncertain:4d} ({100*n_uncertain//total if total else 0}%)")
    print(f"  total:     {total:4d}")

    if n_wrong > 0:
        print(f"\nSample wrong expectations:")
        shown = 0
        tc_map = {tc.id: tc for _, tc in tcs}
        for r in all_results:
            for v in r.wrong:
                print(f"  [{r.tc_id}] {v.event_id}: {v.issue}")
                tc = tc_map.get(r.tc_id)
                if tc and shown < 3:
                    step_num = int(v.event_id[1:]) - 1 if v.event_id.startswith("S") else -1
                    stored = tc.steps[step_num].expect.action if 0 <= step_num < len(tc.steps) and tc.steps[step_num].expect else "?"
                    print(f"    stored:    {stored[:120]}")
                    print(f"    corrected: {v.corrected_action[:120]}")
                shown += 1
                if shown >= 5:
                    break
            if shown >= 5:
                break

    # --- Repair ---
    if repair and n_wrong > 0:
        audit_map = {r.tc_id: r for r in all_results}
        raw_lines = input_path.read_text().splitlines()
        repaired = 0

        for i, tc in tcs:
            if tc.id not in audit_map:
                continue
            if _repair_tc(tc, audit_map[tc.id]):
                raw_lines[i] = tc.model_dump_json()
                repaired += 1

        input_path.write_text("\n".join(raw_lines) + "\n")
        print(f"\nRepaired {repaired} TCs in {input_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Path to test_cases.jsonl")
    p.add_argument("--output", default=None, type=Path,
                   help="Audit output JSONL (default: <input_dir>/expectation_audit.jsonl)")
    p.add_argument("--repair", action="store_true",
                   help="Apply corrected actions for 'wrong' verdicts back to test_cases.jsonl")
    p.add_argument("--workers", type=int, default=4)
    add_common_args(p)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    provider = args.provider or infer_provider(args.model)
    output = args.output or args.input.parent / "expectation_audit.jsonl"

    audit(
        input_path=Path(args.input),
        output_path=output,
        model=args.model,
        provider=provider,
        workers=args.workers,
        repair=args.repair,
    )


if __name__ == "__main__":
    main()
