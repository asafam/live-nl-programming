"""
Sample generator for live NL programming.

Generates concrete samples from raw Zapier automation templates using a three-stage
LLM pipeline:
  1. Ground  — replace abstract placeholders with specific concrete values
  2. Objects — design the distributed LLM-object system from the grounded scenario
  3. Steps   — write the external trigger steps

Each stage is a focused LLM call, producing higher-quality output than attempting
all three tasks in a single prompt.

Usage:
    python -m src.data.generate_samples \\
        --input data/zapier/raw/examples.yaml \\
        --output outputs/data/zapier/generated/samples.jsonl \\
        --model claude-sonnet-4-6 \\
        --seed 42 \\
        --samples-per-template 3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.data.schema import (
    GroundedTemplate,
    MockToolDef,
    ObjectGraph,
    Sample,
    SampleSteps,
)
from src.data.llm import create_llm
from src.lnl.parser import slugify
from src.data.llm.base import ChatMessage
from src.data.utils import (
    infer_provider,
    load_prompt_template,
    load_yaml,
    load_completed_keys,
    generate_with_retries,
    add_common_args,
    validate_paths,
    setup_output,
    print_run_info,
)

# ── Prompt directories ────────────────────────────────────────────────────────

_PROMPTS_DIR = Path("config/prompts/data-gen")
_GROUND_PROMPT = _PROMPTS_DIR / "ground_template.yaml"
_OBJECTS_PROMPT = _PROMPTS_DIR / "identify_objects.yaml"
_STEPS_PROMPT = _PROMPTS_DIR / "write_steps.yaml"


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _format_template(template: dict) -> str:
    steps = "\n".join(f"- {s}" for s in template["raw_steps"])
    return (
        f"ID: {template['id']}\n"
        f"Name: {template['name']}\n"
        f"Domain: {template.get('domain', 'general')}\n"
        f"Source: {template['source_type']}\n"
        f"Link: {template['link']}\n\n"
        f"Raw Steps:\n{steps}"
    )


def _format_objects(graph: ObjectGraph) -> str:
    lines = []
    for obj in graph.objects:
        lines.append(f"- {obj.object_id} ({obj.role})")
        lines.append(f"  behavior: {obj.behavior[:200]}")
        if obj.peers:
            peer_ids = ", ".join(p.object_id for p in obj.peers)
            lines.append(f"  peers: {peer_ids}")
        if obj.event_sources:
            lines.append(f"  event_sources: {'; '.join(obj.event_sources)}")
    return "\n".join(lines)


def _ground_template(llm, template: dict, prompt_cfg: dict) -> GroundedTemplate | None:
    """Stage 1a: resolve abstract placeholders into specific concrete values."""
    prompt = (
        prompt_cfg["prompt"]
        .replace("{TEMPLATE}", _format_template(template))
    )
    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=GroundedTemplate,
        item_id=f"{template['id']}:ground",
        validator=lambda r: bool(r.grounded_steps),
    )


def _identify_objects(llm, grounded: GroundedTemplate, template: dict, prompt_cfg: dict) -> ObjectGraph | None:
    """Stage 1b: design the distributed LLM-object system from the grounded scenario."""
    steps_text = "\n".join(f"- {s}" for s in grounded.grounded_steps)
    prompt = (
        prompt_cfg["prompt"]
        .replace("{NAME}", grounded.name)
        .replace("{DOMAIN}", grounded.domain)
        .replace("{GROUNDED_STEPS}", steps_text)
    )
    def _validate_object_graph(r: ObjectGraph) -> bool:
        if not r.objects:
            return False
        # Every entry-point object (has event_sources) must declare at least one peer.
        # Without a peer, incoming events dead-end and the automation never runs.
        for obj in r.objects:
            if obj.event_sources and not obj.peers:
                return False
        return True

    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=ObjectGraph,
        item_id=f"{template['id']}:objects",
        validator=_validate_object_graph,
    )


def _write_steps(llm, grounded: GroundedTemplate, graph: ObjectGraph, template: dict, prompt_cfg: dict) -> SampleSteps | None:
    """Stage 1c: write the external trigger steps."""
    steps_text = "\n".join(f"- {s}" for s in grounded.grounded_steps)
    prompt = (
        prompt_cfg["prompt"]
        .replace("{NAME}", grounded.name)
        .replace("{GROUNDED_STEPS}", steps_text)
        .replace("{OBJECTS}", _format_objects(graph))
    )
    valid_entry_points = {obj.object_id for obj in graph.objects if obj.event_sources}
    return generate_with_retries(
        llm=llm,
        prompt=prompt,
        response_model=SampleSteps,
        item_id=f"{template['id']}:steps",
        validator=lambda r: bool(r.steps) and all(
            s.target in valid_entry_points for s in r.steps
        ),
    )


def _generate_mock_tool_data(llm, tool_name: str, description: str, step_texts: list[str]) -> MockToolDef | None:
    """Generate a mock tool for a read-service object.

    tool_name: the exact tool name (e.g. ``org_directory_data``)
    description: what the service stores (from state_description or role)
    """
    step_context = ""
    if step_texts:
        step_context = (
            "\n\nThe automation references these specific people, items, or identifiers:\n"
            + "\n".join(f"  - {t}" for t in step_texts)
            + "\n\nYour data MUST include entries for every person, item, or entity "
            "mentioned above, using exactly the same names."
        )

    messages = [
        ChatMessage(
            role="user",
            content=(
                f"Generate realistic reference data for a read-service mock API tool.\n\n"
                f"Tool: {tool_name}\n"
                f"What it stores: {description}"
                f"{step_context}\n\n"
                "IMPORTANT: Include only STATIC reference data — org structure, employee records, "
                "approval authority, availability/substitution rules. "
                "Do NOT include transactional outcomes (approvals, rejections, action history) "
                "that happen during the automation run. The tool is a directory/database, not a log.\n\n"
                "Respond with ONLY a raw JSON object (no markdown, no explanation). "
                "Use realistic names and values. Structure the data logically "
                "(e.g., an employee directory returns a list of employee objects)."
            ),
        )
    ]
    try:
        text = llm.generate_text(messages=messages)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        label = tool_name.replace("_", " ")
        return MockToolDef(
            tool_name=tool_name,
            description=f"Retrieve reference data from {label}. {description}",
            arguments_schema={"type": "object", "additionalProperties": True},
            response_template=json.dumps(data, ensure_ascii=False),
        )
    except Exception:
        return None


def _assemble_sample(template: dict, grounded: GroundedTemplate, graph: ObjectGraph, steps: SampleSteps) -> Sample:
    """Combine stage outputs into a Sample. Slugify ids. Mock tools generated separately."""
    for obj in graph.objects:
        obj.object_id = slugify(obj.object_id)
        for peer in obj.peers:
            peer.object_id = slugify(peer.object_id)
    for step in steps.steps:
        step.target = slugify(step.target)

    return Sample(
        id=template["id"],
        name=grounded.name,
        domain=grounded.domain,
        source_type=template["source_type"],
        link=template["link"],
        raw_steps=template["raw_steps"],
        objects=graph.objects,
        steps=steps.steps,
    )


_DATA_TOOL_RE = re.compile(r"call the `([a-z][a-z0-9_]*_data)` tool", re.IGNORECASE)


def _add_mock_tools(llm, sample: Sample) -> None:
    """Post-process: generate mock tools for read-service objects (mutates sample.mock_tools).

    Read services are detected by the mandatory behavior phrase:
      "call the `{object_id}_data` tool to retrieve data"
    This is more reliable than checking state_description, which is intentionally
    empty for read services (they hold no mutable state of their own).
    """
    step_texts = [s.text for s in sample.steps if s.text]
    for obj in sample.objects:
        match = _DATA_TOOL_RE.search(obj.behavior or "")
        if not match:
            continue
        tool_name = match.group(1)  # e.g. "org_directory_data"
        # Use state_description if present, otherwise fall back to role
        description = (obj.state_description or "").strip() or obj.role
        tool = _generate_mock_tool_data(llm, tool_name, description, step_texts)
        if tool:
            sample.mock_tools.append(tool)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate samples from raw Zapier automation templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml --model gpt-4o
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml --samples-per-template 5
""",
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to raw templates YAML file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSONL path (default: derived from input filename)")
    parser.add_argument("--samples-per-template", type=int, default=1,
                        help="Number of samples to generate per template (default: 1)")
    parser.add_argument("--id", dest="ids", metavar="ID", action="append", default=None,
                        help="Only process template(s) with this ID (repeatable)")
    add_common_args(parser)
    return parser


def default_output_path(input_path: Path) -> Path:
    return Path("outputs/data/zapier") / f"{input_path.stem}_samples.jsonl"


def run(args: argparse.Namespace) -> Path:
    if args.output is None:
        args.output = default_output_path(args.input)
    if args.provider is None:
        args.provider = infer_provider(args.model)
    if args.seed is not None:
        random.seed(args.seed)

    validate_paths(args.input, _GROUND_PROMPT)
    for p in [_OBJECTS_PROMPT, _STEPS_PROMPT]:
        if not p.exists():
            print(f"Error: prompt file not found: {p}", file=sys.stderr)
            sys.exit(1)

    templates = load_yaml(args.input)
    ground_cfg = load_prompt_template(_GROUND_PROMPT)
    objects_cfg = load_prompt_template(_OBJECTS_PROMPT)
    steps_cfg = load_prompt_template(_STEPS_PROMPT)

    if args.ids:
        id_set = set(args.ids)
        templates = [t for t in templates if t["id"] in id_set]
        if not templates:
            print(f"Error: no templates found with ID(s): {', '.join(sorted(id_set))}", file=sys.stderr)
            sys.exit(1)

    if args.limit:
        templates = templates[: args.limit]

    print(f"Loaded {len(templates)} templates from {args.input}")

    completed, file_mode = setup_output(
        args.output,
        args.force,
        lambda: load_completed_keys(args.output, lambda d: d.get("link")),
    )
    pending = [t for t in templates if t["link"] not in completed]

    if not pending:
        print("All templates already generated. Use --force to regenerate.")
        return args.output

    if completed:
        print(f"Resuming: {len(completed)} already completed, {len(pending)} remaining")
    else:
        print(f"Processing {len(pending)} templates")

    print_run_info(args.provider, args.model, args.seed,
                   {"Samples per template": str(args.samples_per_template)})

    llm = create_llm(provider=args.provider, model=args.model,
                     temperature=args.temperature, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0

    with open(args.output, file_mode) as f:
        for template in tqdm(pending, desc="Generating samples"):
            samples_written = 0

            for attempt in range(args.samples_per_template):
                # Stage 1a: Ground
                grounded = _ground_template(llm, template, ground_cfg)
                if not grounded:
                    fail_count += 1
                    continue

                # Stage 1b: Identify objects
                graph = _identify_objects(llm, grounded, template, objects_cfg)
                if not graph:
                    fail_count += 1
                    continue

                # Stage 1c: Write steps
                sample_steps = _write_steps(llm, grounded, graph, template, steps_cfg)
                if not sample_steps:
                    fail_count += 1
                    continue

                # Assemble Sample (slugify, combine)
                sample = _assemble_sample(template, grounded, graph, sample_steps)

                f.write(sample.model_dump_json() + "\n")
                f.flush()
                samples_written += 1

            success_count += samples_written
            if samples_written == 0:
                fail_count += 1

    print()
    print(f"Complete. Output: {args.output}")
    print(f"Samples generated: {success_count}, Templates failed: {fail_count}")
    return args.output


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
