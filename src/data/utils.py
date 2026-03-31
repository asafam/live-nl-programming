"""
Shared utilities for data generation scripts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Set, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def infer_provider(model: str) -> str:
    """Infer the LLM provider from the model name."""
    if model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    else:
        raise ValueError(
            f"Cannot infer provider from model '{model}'. "
            f"Use --provider to specify 'openai' or 'anthropic'."
        )


def load_prompt_template(path: Path) -> dict:
    """Load prompt template from YAML file, returning the full config dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_yaml(path: Path) -> list:
    """Load data from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path, model_class: Optional[Type[T]] = None) -> list:
    """Load data from JSONL file, optionally parsing into Pydantic models."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if model_class:
                items.append(model_class(**data))
            else:
                items.append(data)
    return items


def load_completed_keys(
    output_path: Path,
    key_extractor: Callable[[dict], Optional[str]],
) -> Set[str]:
    """Load keys of already-generated items for resume support.

    Args:
        output_path: Path to the output JSONL file.
        key_extractor: Function that extracts a unique key from each JSON object.
                       Return None to skip the item.
    """
    completed = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = key_extractor(data)
                    if key:
                        completed.add(key)
                except json.JSONDecodeError:
                    continue
    return completed


def generate_with_retries(
    llm,
    prompt: str,
    response_model: Type[T],
    item_id: str,
    validator: Callable[[T], bool],
    max_retries: int = 3,
) -> Optional[T]:
    """Generate structured output using LLM with retries and exponential backoff.

    Args:
        llm: LLM client instance.
        prompt: Formatted prompt string.
        response_model: Pydantic model class for structured output.
        item_id: Identifier for error reporting.
        validator: Function that validates the result (returns True if valid).
        max_retries: Maximum number of retry attempts.

    Returns:
        Parsed response or None if generation fails.
    """
    from src.data.llm import user_message

    for attempt in range(max_retries):
        try:
            result = llm.generate_structured(
                messages=[user_message(prompt)],
                response_model=response_model,
            )

            if not validator(result):
                raise ValueError("Validation failed")

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"  Retry {attempt + 1}/{max_retries} for '{item_id}' "
                    f"in {wait}s: {e}",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(
                    f"  Failed '{item_id}' after {max_retries} attempts: {e}",
                    file=sys.stderr,
                )
                return None


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared across generation scripts."""
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider (inferred from model if not specified)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="claude-sonnet-4-6",
        help="Model name (default: claude-sonnet-4-6). Provider is inferred: claude-* → anthropic, gpt-*/o1-*/o3-* → openai",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all items, ignoring existing output",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process only the first N items from the input file",
    )


def validate_paths(input_path: Path, prompt_template_path: Path) -> None:
    """Validate that required input files exist."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not prompt_template_path.exists():
        print(
            f"Error: Prompt template not found: {prompt_template_path}",
            file=sys.stderr,
        )
        sys.exit(1)


def confirm_overwrite(output_path: Path) -> None:
    """Prompt user for confirmation if output file already exists. Exits if declined."""
    if output_path.exists():
        response = input(f"Output file already exists: {output_path}\nOverwrite? [y/N] ")
        if response.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)


def setup_output(
    output_path: Path,
    force: bool,
    load_completed: Callable[[], Set[str]],
) -> tuple[Set[str], str]:
    """Setup output file and determine completed items.

    Returns:
        Tuple of (completed_set, file_mode).
    """
    confirm_overwrite(output_path)
    if force:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return set(), "w"
    else:
        completed = load_completed()
        return completed, "a"


def print_run_info(
    provider: str,
    model: str,
    seed: Optional[int],
    extra_info: dict[str, str],
) -> None:
    """Print run configuration info."""
    print(f"Provider: {provider}, Model: {model}")
    for key, value in extra_info.items():
        print(f"{key}: {value}")
    if seed is not None:
        print(f"Seed: {seed}")
    print()
