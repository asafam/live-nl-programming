"""
Sample generator for live NL programming.

Generates concrete samples from raw Zapier automation templates using LLM-based
generation. Each sample instantiates a template with specific values.

Usage:
    python -m src.data.generate_samples \\
        --input data/zapier/raw/examples.yaml \\
        --output outputs/data/zapier/generated/samples.jsonl \\
        --model claude-sonnet-4-5-20250929 \\
        --seed 42 \\
        --samples-per-template 3
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

from src.data.schema import Samples
from src.data.llm import create_llm
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


def format_template(template: dict) -> str:
    """Format a template for the prompt."""
    steps = "\n".join(f"- {step}" for step in template["raw_steps"])
    return f"""ID: {template['id']}
Name: {template['name']}
Domain: {template.get('domain', 'general')}
Source: {template['source_type']}
Link: {template['link']}

Raw Steps:
{steps}"""


def format_prompt(prompt_template: dict, template: dict, samples_count: int, step_style: str = "plain") -> str:
    """Format prompt template with template data and parameters."""
    template_str = format_template(template)
    prompt = prompt_template["base_prompt"]
    if step_style == "actor":
        prompt += prompt_template.get("actor_style_addendum", "")
    else:
        prompt += prompt_template.get("plain_style_addendum", "")
    return prompt.format(
        TEMPLATE=template_str,
        SAMPLES_COUNT=samples_count,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from raw Zapier automation templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default model (provider inferred from model)
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml

  # Generate with OpenAI
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml --model gpt-4o

  # Multiple samples per template
  python -m src.data.generate_samples -i data/zapier/raw/examples.yaml --samples-per-template 5
""",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to raw templates YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSONL path (default: derived from input filename and step-style)",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("config/prompts/data-gen/generate_samples.yaml"),
        help="Path to prompt template (default: config/prompts/data-gen/generate_samples.yaml)",
    )
    parser.add_argument(
        "--samples-per-template",
        type=int,
        default=1,
        help="Number of samples to generate per template (default: 1)",
    )
    parser.add_argument(
        "--step-style",
        choices=["plain", "actor"],
        default="plain",
        help="Step rewriting style: 'plain' (default) or 'actor' (Actor-creation language)",
    )
    add_common_args(parser)

    args = parser.parse_args()

    # Derive default output path from input filename and step-style
    if args.output is None:
        input_stem = args.input.stem  # e.g. "templates"
        output_name = f"{input_stem}_samples_{args.step_style}.jsonl"
        args.output = Path("outputs/data/zapier") / output_name

    # Infer provider from model if not specified
    if args.provider is None:
        args.provider = infer_provider(args.model)

    # Initialize random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Validate inputs
    validate_paths(args.input, args.prompt_template)

    # Load data
    templates = load_yaml(args.input)
    prompt_template = load_prompt_template(args.prompt_template)

    # Apply limit if specified (0 or None means no limit)
    if args.limit:
        templates = templates[: args.limit]

    print(f"Loaded {len(templates)} templates from {args.input}")

    # Setup output and determine completed items
    completed, file_mode = setup_output(
        args.output,
        args.force,
        lambda: load_completed_keys(args.output, lambda d: d.get("link")),
    )

    pending = [t for t in templates if t["link"] not in completed]

    if not pending:
        print("All templates already generated. Use --force to regenerate.")
        return

    if completed:
        print(f"Resuming: {len(completed)} already completed, {len(pending)} remaining")
    else:
        print(f"Processing {len(pending)} templates")

    print_run_info(
        args.provider,
        args.model,
        args.seed,
        {
            "Samples per template": str(args.samples_per_template),
            "Step style": args.step_style,
        },
    )

    # Create LLM client
    llm = create_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Process templates
    args.output.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0

    with open(args.output, file_mode) as f:
        for template in tqdm(pending, desc="Generating"):
            # Format prompt
            prompt = format_prompt(
                prompt_template, template, args.samples_per_template, args.step_style
            )

            # Generate samples
            result = generate_with_retries(
                llm=llm,
                prompt=prompt,
                response_model=Samples,
                item_id=template["id"],
                validator=lambda r: bool(r.samples),
            )

            if result:
                # Write each sample as a separate line
                for sample in result.samples:
                    f.write(sample.model_dump_json() + "\n")
                f.flush()
                success_count += len(result.samples)
            else:
                fail_count += 1

    print()
    print(f"Complete. Output: {args.output}")
    print(f"Samples generated: {success_count}, Templates failed: {fail_count}")


if __name__ == "__main__":
    main()
