"""Stage 1c-b: regenerate per-Step expect.action / expect.reason for existing workflows.

Surgical re-write: takes an existing workflows.jsonl, keeps every Workflow's
objects + step text + targets unchanged, but rewrites each Step's `expect`
field with the latest write_step_expects.yaml prompt rules (single observable
outcome per Step, no bundled cascade, vendor-consistent naming).

Use this when:
  - the workflow structure is fine, but the expect fields drifted (e.g., bundled
    multiple outcomes, used a vendor name not in the template)
  - you don't want to re-pay for Stage 1a/1b grounding + object identification

Usage:
    python -m src.data.regen_expects \\
        --workflows outputs/.../workflows.jsonl \\
        --provider azure --model gpt-5.4 \\
        --workers 12 \\
        [--filter <id> ...] [--output new.jsonl]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.data.llm import create_llm
from src.data.schema import (
    EventExpect,
    Workflow,
    WorkflowExpectsRegenOutput,
)
from src.data.utils import generate_with_retries, load_jsonl

load_dotenv()


_PROMPT_PATH = (
    Path(__file__).parent.parent.parent
    / "config" / "prompts" / "data-gen" / "write_step_expects.yaml"
)


def _load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["prompt"]


def _format_grounded_steps(workflow: Workflow) -> str:
    if not workflow.raw_steps:
        return "(no grounded scenario steps)"
    return "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(workflow.raw_steps))


def _format_objects(workflow: Workflow) -> str:
    if not workflow.objects:
        return "(no objects)"
    lines = []
    for o in workflow.objects:
        lines.append(f"### {o.object_id}")
        lines.append(f"  role: {o.role}")
        lines.append(f"  behavior: {o.behavior}")
        if o.state_description:
            lines.append(f"  state: {o.state_description}")
        if o.event_sources:
            lines.append(f"  event_sources: {o.event_sources}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_steps_for_expects(workflow: Workflow) -> str:
    if not workflow.steps:
        return "(no steps)"
    lines = []
    for i, st in enumerate(workflow.steps, 1):
        lines.append(f"### Step {i}")
        lines.append(f"  target: {st.target}")
        lines.append(f"  source: {st.source}")
        lines.append(f"  text: {st.text}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _regen_workflow_expects(llm, workflow: Workflow, prompt_template: str) -> Workflow:
    """Call the LLM once per workflow to rewrite every Step's expect fields in-place."""
    if not workflow.steps:
        return workflow

    prompt = (
        prompt_template
        .replace("{NAME}", workflow.name)
        .replace("{GROUNDED_STEPS}", _format_grounded_steps(workflow))
        .replace("{OBJECTS}", _format_objects(workflow))
        .replace("{STEPS}", _format_steps_for_expects(workflow))
    )

    result = generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=WorkflowExpectsRegenOutput,
        item_id=f"{workflow.id}-expects",
        validator=lambda r: len(r.step_expects) == len(workflow.steps),
    )

    if result is None:
        # Leave the existing expects untouched on failure
        return workflow

    expects_by_idx = {e.step_index: e for e in result.step_expects}
    for i, step in enumerate(workflow.steps, 1):
        e = expects_by_idx.get(i)
        if e is None:
            continue
        if (e.action or "").strip():
            step.expect = EventExpect(action=e.action, reason=(e.reason or ""))
        else:
            # null action means: scheduled/heartbeat trigger — leave expect=None
            step.expect = None
    return workflow


def main_with_args(args: argparse.Namespace) -> int:
    workflows: list[Workflow] = load_jsonl(args.workflows, Workflow)
    prompt_template = _load_prompt()

    if args.filter:
        workflows = [w for w in workflows if w.id in set(args.filter)]
    if args.limit:
        workflows = workflows[: args.limit]

    if not workflows:
        print("No workflows to process.", file=sys.stderr)
        return 1

    output_path = args.output or args.workflows  # in-place by default
    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    # If writing to the same path, build a complete list in memory first, then
    # overwrite (avoid corrupting the input file mid-stream).
    in_place = output_path == args.workflows

    updated: dict[str, Workflow] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_regen_workflow_expects, llm, w, prompt_template): w for w in workflows}
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover
                print(f"  ✗ {w.id}: regen crashed — {exc}", file=sys.stderr)
                updated[w.id] = w
                continue
            updated[w.id] = result
            print(f"  ✓ {result.id}  ({len(result.steps)} step(s) refreshed)")

    if in_place:
        # Re-read full list (including any not filtered) to preserve untouched workflows
        all_workflows: list[Workflow] = load_jsonl(args.workflows, Workflow)
        out_list = [updated.get(w.id, w) for w in all_workflows]
    else:
        out_list = [updated[w.id] for w in workflows]

    with open(output_path, "w") as f:
        for w in out_list:
            f.write(w.model_dump_json() + "\n")

    print(f"\nWrote {len(out_list)} workflows to {output_path}")
    print(f"Regenerated expects for: {len(updated)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workflows", required=True, type=Path,
                        help="Path to workflows JSONL (Workflow records). Edited in-place unless --output is set.")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "azure"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Lower than generation default — expects benefit from determinism.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write to this path instead of editing in-place.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Only regenerate expects for these workflow ids.")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
