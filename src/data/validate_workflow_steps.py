"""Stage 1d: Validate grounded Workflow steps against templates.yaml raw_steps.

For each generated Workflow in the input JSONL, compare each Step's grounded
text + expect.action against the corresponding abstract raw_step from
templates.yaml. An LLM judge classifies each pair as FAITHFUL / DRIFTED /
WRONG. Per-workflow aggregate rolls up to CLEAN / MILD_DRIFT / NOTABLE_DRIFT
/ WRONG.

Usage:
    python -m src.data.validate_workflow_steps \\
        --workflows outputs/data/zapier/20260521_multistep/workflows.jsonl \\
        --templates data/zapier/raw/templates.yaml \\
        --provider openai --judge-model gpt-5.4 \\
        --workers 4 \\
        --output validation_results.jsonl

Exit code is non-zero if any workflow scores WRONG (override with --no-fail).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.data.llm import create_llm

load_dotenv()
from src.data.schema import (
    MissedTrigger,
    RawStepClassification,
    StepVerdict,
    Workflow,
    WorkflowStepsJudgement,
    WorkflowValidation,
)
from src.data.utils import generate_with_retries, load_jsonl


_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "data-gen" / "validate_workflow_step.yaml"


def _load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["prompt"]


def _load_templates(path: Path) -> dict[str, dict]:
    with open(path) as f:
        templates = yaml.safe_load(f)
    return {t["id"]: t for t in templates}


def _format_all_raw_steps(raw_steps: list[str]) -> str:
    return "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(raw_steps))


def _health_check_step(workflow: Workflow, step_index: int) -> list[str]:
    """Deterministic per-step structural checks. Returns list of issues (empty = OK)."""
    issues: list[str] = []
    step = workflow.steps[step_index]
    if not (step.text or "").strip():
        issues.append("text is empty")
    if not (step.target or "").strip():
        issues.append("target is empty")
    else:
        object_map = {o.object_id: o for o in workflow.objects}
        target_obj = object_map.get(step.target)
        if target_obj is None:
            issues.append(f"target '{step.target}' does not exist in workflow.objects")
        elif not target_obj.event_sources:
            issues.append(
                f"target '{step.target}' has no event_sources (not an entry-point object)"
            )
    if step.expect is None or not (step.expect.action or "").strip():
        issues.append("expect.action is missing or empty")
    return issues


def _format_workflow_steps(workflow: Workflow) -> str:
    if not workflow.steps:
        return "(no workflow steps)"
    lines = []
    for i, st in enumerate(workflow.steps, 1):
        expect_action = st.expect.action if st.expect else "(none)"
        lines.append(f"  [{i}] target={st.target}")
        lines.append(f"      text: {st.text}")
        lines.append(f"      expect.action: {expect_action}")
    return "\n".join(lines)


def _judge_workflow_steps(
    llm,
    workflow: Workflow,
    template: dict,
    prompt_template: str,
) -> Optional[WorkflowStepsJudgement]:
    """Single LLM call that classifies raw_steps + judges each workflow Step + reports missed triggers."""
    prompt = (
        prompt_template
        .replace("{WORKFLOW_ID}", workflow.id)
        .replace("{WORKFLOW_NAME}", workflow.name)
        .replace("{LINK}", workflow.link or "")
        .replace("{RAW_STEPS}", _format_all_raw_steps(template["raw_steps"]))
        .replace("{WORKFLOW_STEPS}", _format_workflow_steps(workflow))
    )
    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=WorkflowStepsJudgement,
        item_id=f"{workflow.id}-steps",
        validator=lambda r: (
            len(r.step_judgements) == len(workflow.steps)
            and len(r.raw_step_classifications) == len(template["raw_steps"])
            and all(sj.verdict in ("FAITHFUL", "DRIFTED", "WRONG") for sj in r.step_judgements)
            and all(sj.quality in ("GOOD", "ADEQUATE", "POOR") for sj in r.step_judgements)
        ),
    )


def _aggregate_verdict(step_verdicts: list[StepVerdict], n_missed_triggers: int) -> str:
    """Roll fidelity up to a workflow-level aggregate (trigger-only framing)."""
    has_wrong = any(v.verdict == "WRONG" for v in step_verdicts)
    n_drifted = sum(1 for v in step_verdicts if v.verdict == "DRIFTED")
    if has_wrong:
        return "WRONG"
    n_issues = n_drifted + n_missed_triggers
    if n_issues >= 2:
        return "NOTABLE_DRIFT"
    if n_issues == 1:
        return "MILD_DRIFT"
    return "CLEAN"


def _aggregate_health(step_verdicts: list[StepVerdict]) -> str:
    return "OK" if all(not v.health_issues for v in step_verdicts) else "ISSUES"


def _aggregate_quality(step_verdicts: list[StepVerdict]) -> str:
    scores = [v.quality for v in step_verdicts]
    if any(q == "POOR" for q in scores):
        return "POOR"
    if any(q == "ADEQUATE" for q in scores):
        return "ADEQUATE"
    return "GOOD" if scores else "ADEQUATE"


def _validate_workflow(
    llm,
    workflow: Workflow,
    template: Optional[dict],
    prompt_template: str,
) -> WorkflowValidation:
    """Validate one workflow's Steps. Single LLM call + deterministic health checks."""
    if template is None:
        return WorkflowValidation(
            workflow_id=workflow.id,
            n_template_steps=0,
            n_workflow_steps=len(workflow.steps),
            step_verdicts=[],
            aggregate="WRONG",
            aggregate_health="ISSUES",
            aggregate_quality="POOR",
        )

    n_template = len(template["raw_steps"])
    n_workflow = len(workflow.steps)

    # Deterministic health pass per Step
    health_lists = [_health_check_step(workflow, i) for i in range(n_workflow)]

    # Single LLM call: classify raw_steps + judge each Step + report missed triggers
    judgement = _judge_workflow_steps(llm, workflow, template, prompt_template)

    if judgement is None:
        # Fall back: all WRONG, mark every TRIGGER raw_step as missed (we don't know which).
        step_verdicts = [
            StepVerdict(
                workflow_id=workflow.id,
                step_index=i,
                grounded_step=workflow.steps[i].text,
                expect_action=workflow.steps[i].expect.action if workflow.steps[i].expect else None,
                target=workflow.steps[i].target,
                verdict="WRONG",
                reasoning="(judge failed)",
                health_issues=health_lists[i],
                quality="POOR",
                quality_issues=["(judge failed; quality not assessed)"],
            )
            for i in range(n_workflow)
        ]
        return WorkflowValidation(
            workflow_id=workflow.id,
            n_template_steps=n_template,
            n_template_triggers=0,
            n_workflow_steps=n_workflow,
            step_verdicts=step_verdicts,
            missed_triggers=[],
            raw_step_classifications=[],
            aggregate="WRONG",
            aggregate_health=_aggregate_health(step_verdicts),
            aggregate_quality="POOR",
        )

    # Build per-Step verdicts using LLM output + deterministic health
    step_judgements_by_idx = {sj.workflow_step_index: sj for sj in judgement.step_judgements}
    step_verdicts: list[StepVerdict] = []
    for i in range(n_workflow):
        sj = step_judgements_by_idx.get(i + 1)
        step = workflow.steps[i]
        if sj is None:
            step_verdicts.append(StepVerdict(
                workflow_id=workflow.id,
                step_index=i,
                grounded_step=step.text,
                expect_action=step.expect.action if step.expect else None,
                target=step.target,
                verdict="WRONG",
                reasoning="(judge did not produce a verdict for this Step)",
                health_issues=health_lists[i],
                quality="POOR",
                quality_issues=["(no judge verdict)"],
            ))
        else:
            step_verdicts.append(StepVerdict(
                workflow_id=workflow.id,
                step_index=i,
                grounded_step=step.text,
                expect_action=step.expect.action if step.expect else None,
                target=step.target,
                aligned_to=sj.aligned_to,
                verdict=sj.verdict,
                reasoning=sj.reasoning,
                health_issues=health_lists[i],
                quality=sj.quality,
                quality_issues=list(sj.quality_issues or []),
            ))

    n_triggers = sum(1 for c in judgement.raw_step_classifications if c.classification == "TRIGGER")
    n_missed = len(judgement.missed_triggers)

    return WorkflowValidation(
        workflow_id=workflow.id,
        n_template_steps=n_template,
        n_template_triggers=n_triggers,
        n_workflow_steps=n_workflow,
        step_verdicts=step_verdicts,
        missed_triggers=list(judgement.missed_triggers),
        raw_step_classifications=list(judgement.raw_step_classifications),
        aggregate=_aggregate_verdict(step_verdicts, n_missed),
        aggregate_health=_aggregate_health(step_verdicts),
        aggregate_quality=_aggregate_quality(step_verdicts),
    )


def _print_summary(results: list[WorkflowValidation]) -> None:
    """Print fidelity / health / quality rollups + missed-trigger stats."""
    from collections import Counter
    counts = Counter(r.aggregate for r in results)
    health_counts = Counter(r.aggregate_health for r in results)
    quality_counts = Counter(r.aggregate_quality for r in results)
    total_missed_triggers = sum(len(r.missed_triggers) for r in results)
    workflows_with_missed = sum(1 for r in results if r.missed_triggers)

    print("\n" + "=" * 70)
    print(f"Workflow step validation — {len(results)} workflows")
    print("=" * 70)
    print(f"Missed triggers: {total_missed_triggers} across {workflows_with_missed} workflow(s)")
    print("Fidelity (workflow Step grounds a TRIGGER template raw_step):")
    print(f"  CLEAN:          {counts.get('CLEAN', 0):3d}")
    print(f"  MILD_DRIFT:     {counts.get('MILD_DRIFT', 0):3d}")
    print(f"  NOTABLE_DRIFT:  {counts.get('NOTABLE_DRIFT', 0):3d}")
    print(f"  WRONG:          {counts.get('WRONG', 0):3d}")
    print("Health (deterministic structural checks):")
    print(f"  OK:             {health_counts.get('OK', 0):3d}")
    print(f"  ISSUES:         {health_counts.get('ISSUES', 0):3d}")
    print("Quality (LLM grading of grounded text):")
    print(f"  GOOD:           {quality_counts.get('GOOD', 0):3d}")
    print(f"  ADEQUATE:       {quality_counts.get('ADEQUATE', 0):3d}")
    print(f"  POOR:           {quality_counts.get('POOR', 0):3d}")
    print()

    # Flag if EITHER fidelity is bad OR health has issues OR quality is poor
    flagged = [
        r for r in results
        if r.aggregate in ("NOTABLE_DRIFT", "WRONG")
        or r.aggregate_health == "ISSUES"
        or r.aggregate_quality == "POOR"
    ]
    if flagged:
        print(f"Flagged for review ({len(flagged)}):")
        for r in flagged:
            n_drift = sum(1 for v in r.step_verdicts if v.verdict == "DRIFTED")
            n_wrong = sum(1 for v in r.step_verdicts if v.verdict == "WRONG")
            n_health = sum(len(v.health_issues) for v in r.step_verdicts)
            n_poor = sum(1 for v in r.step_verdicts if v.quality == "POOR")
            n_missed = len(r.missed_triggers)
            print(
                f"  {r.workflow_id:<55} "
                f"fid={r.aggregate:<13} health={r.aggregate_health:<6} quality={r.aggregate_quality:<8} "
                f"(drift={n_drift} wrong={n_wrong} missed_triggers={n_missed} "
                f"health_issues={n_health} poor_steps={n_poor} "
                f"steps={r.n_workflow_steps}/{r.n_template_triggers}trig "
                f"of {r.n_template_steps}raw)"
            )
        print()


def main_with_args(args: argparse.Namespace) -> int:
    """Run the validator with a pre-built args namespace. Returns exit code.

    Used by src.data.pipeline (Stage 1d) so the validation can be invoked
    in-process without re-parsing argv. The CLI ``main()`` is a thin wrapper.
    """
    workflows: list[Workflow] = load_jsonl(args.workflows, Workflow)
    templates = _load_templates(args.templates)
    prompt_template = _load_prompt()

    if args.filter:
        workflows = [w for w in workflows if w.id in set(args.filter)]
    if args.limit:
        workflows = workflows[: args.limit]

    if not workflows:
        print("No workflows to validate.", file=sys.stderr)
        return 1

    output_path = args.output or args.workflows.with_name(
        args.workflows.stem + "__validation.jsonl"
    )

    llm = create_llm(provider=args.provider, model=args.judge_model, temperature=0.0)

    results: list[WorkflowValidation] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_validate_workflow, llm, w, templates.get(w.id), prompt_template): w
            for w in workflows
        }
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover — keep going
                print(f"  ✗ {w.id}: judge crashed — {exc}", file=sys.stderr)
                continue
            results.append(result)
            marker = {
                "CLEAN": "✓", "MILD_DRIFT": "·",
                "NOTABLE_DRIFT": "⚠", "WRONG": "✗",
            }.get(result.aggregate, "?")
            print(f"  {marker} {result.workflow_id:<55} {result.aggregate}")

    # Sort by workflow id for stable output
    results.sort(key=lambda r: r.workflow_id)

    with open(output_path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    _print_summary(results)
    print(f"Wrote {len(results)} validation records to {output_path}")

    has_wrong = any(r.aggregate == "WRONG" for r in results)
    if has_wrong and not args.no_fail:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workflows", required=True, type=Path,
                        help="Path to workflows JSONL (Workflow records).")
    parser.add_argument("--templates", type=Path,
                        default=Path("data/zapier/raw/templates.yaml"),
                        help="Path to templates.yaml.")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "azure"))
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workflows judged at once.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write WorkflowValidation JSONL here. Default: alongside workflows.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge the first N workflows (for quick checks).")
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Only judge these workflow ids.")
    parser.add_argument("--no-fail", action="store_true",
                        help="Exit 0 even when WRONG workflows are present.")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
