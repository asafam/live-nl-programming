"""Stage 1c surgical: regenerate base events on Workflow records in workflows.jsonl.

Re-runs `_write_steps()` (the same function generate_workflows.py uses) for each
selected workflow, replaces `workflow.events` (role='base') with the freshly
generated base events, and writes back in-place (with a .bak copy).

Use this after `write_steps.yaml` rule changes (e.g., the Sequential coherence
section) to fix workflows the new event validator flagged as PARADOX/POOR.

NOTE: Does NOT propagate changes into workflows-mods.jsonl Samples. Sync that
separately with sync_sample_steps or a follow-up tool once the new events are
verified clean.

Usage:
    python -m src.data.regen_workflow_events \\
        --workflows outputs/.../workflows.jsonl \\
        --filter deal-desk-manage-hubspot-quote-approvals-slack \\
        --provider azure --model gpt-5.4 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from src.data.generate_workflows import _write_steps, _STEPS_PROMPT
from src.data.llm import create_llm
from src.data.schema import (
    Event,
    GroundedTemplate,
    ObjectGraph,
    Workflow,
)
from src.data.utils import load_jsonl

load_dotenv()


def _regen_one(llm, workflow: Workflow, prompt_cfg: dict) -> Workflow | None:
    grounded = GroundedTemplate(
        name=workflow.name,
        domain=workflow.domain,
        grounded_steps=list(workflow.steps),
    )
    graph = ObjectGraph(objects=list(workflow.objects))
    template = {"id": workflow.id}

    result = _write_steps(llm, grounded, graph, template, prompt_cfg, tools=workflow.tools)
    if not result:
        return None

    new_events = [
        Event(
            id=f"S{i+1:03d}",
            call_type="send",
            source=s.source,
            recipient=s.target,
            input=s.text,
            when="W00-1T00:00",
            expect=s.expect,
            role="base",
        )
        for i, s in enumerate(result.steps)
    ]

    non_base = [e for e in workflow.events if e.role != "base"]
    workflow.events = new_events + non_base
    return workflow


def main_with_args(args: argparse.Namespace) -> int:
    workflows: list[Workflow] = load_jsonl(args.workflows, Workflow)

    if args.filter:
        ids = set(args.filter)
        targets = [w for w in workflows if w.id in ids]
    else:
        targets = workflows[: args.limit] if args.limit else workflows

    if not targets:
        print("No workflows match filter/limit.", file=sys.stderr)
        return 1

    with open(_STEPS_PROMPT) as f:
        prompt_cfg = yaml.safe_load(f)

    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    print(f"Regenerating base events for {len(targets)} workflow(s)...")

    updated: dict[str, Workflow] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_regen_one, llm, w, prompt_cfg): w for w in targets}
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  ✗ {w.id}: crashed — {exc}", file=sys.stderr)
                failed.append(w.id)
                continue
            if result is None:
                print(f"  ✗ {w.id}: _write_steps returned None", file=sys.stderr)
                failed.append(w.id)
                continue
            n_base = sum(1 for e in result.events if e.role == "base")
            updated[result.id] = result
            print(f"  ✓ {result.id:<55} ({n_base} base events)")

    if not updated:
        print("Nothing to write.")
        return 1

    if not args.no_backup:
        bak = args.workflows.with_suffix(args.workflows.suffix + ".bak_pre_event_regen")
        shutil.copy2(args.workflows, bak)
        print(f"  Backup → {bak}")

    out_path = args.output or args.workflows
    out_list = [updated.get(w.id, w) for w in workflows]
    with open(out_path, "w") as f:
        for w in out_list:
            f.write(w.model_dump_json() + "\n")

    print(f"\nWrote {len(out_list)} workflows to {out_path}")
    print(f"Regenerated base events for: {len(updated)}")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workflows", required=True, type=Path,
                        help="Path to workflows JSONL (Workflow records). In-place unless --output set.")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "azure"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Workflow IDs to regenerate (default: all).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing a .bak_pre_event_regen copy of the input.")
    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
