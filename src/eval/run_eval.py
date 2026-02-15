"""
Evaluation runner: loads test cases, feeds them through the actor system,
and records per-input metrics for later analysis.

Usage:
    python -m src.eval.run_eval \
        -i outputs/data/zapier/generated/test_cases.jsonl \
        -o outputs/eval/results.jsonl \
        --provider openai --model gpt-4o-mini \
        --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from src.data.schema import TestCase
from src.data.utils import add_common_args, infer_provider, load_jsonl
from src.eval.schema import EvalRecord, ExpectedOutcome, TokenUsage
from src.system.actors.coordinator_actor import CoordinatorActor
from src.system.actors.user_actor import UserActor
from src.system.llm.base import AbstractLLM, LLMUsage
from src.system.llm.openai_client import OpenAIChatLLM
from src.system.llm.anthropic_client import AnthropicChatLLM
from src.system.message_bus import MessageBus


class TokenAccumulator:
    """Accumulates token usage across multiple LLM calls for a single input."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def collect(self, llm: AbstractLLM) -> None:
        """Collect usage from an LLM instance's last_usage."""
        if llm.last_usage:
            self.prompt_tokens += llm.last_usage.prompt_tokens
            self.completion_tokens += llm.last_usage.completion_tokens
            llm.last_usage = None

    def collect_all(self, bus: MessageBus) -> None:
        """Collect usage from all LLM-backed actors on the bus."""
        for actor in bus.actors.values():
            llm = getattr(actor, 'llm', None)
            if llm and isinstance(llm, AbstractLLM):
                self.collect(llm)

    def to_token_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


def get_actor_states(bus: MessageBus) -> Dict[str, Any]:
    """Snapshot the state of all stateful actors."""
    states = {}
    for name, actor in bus.actors.items():
        state = getattr(actor, 'state', None)
        if state is not None:
            # Deep copy via JSON round-trip to avoid mutation issues
            try:
                states[name] = json.loads(json.dumps(state, default=str))
            except (TypeError, ValueError):
                states[name] = str(state)
    return states


def build_llm_factory(
    provider: str,
    model: str,
    temperature: float,
    seed: Optional[int],
) -> Callable[[str], AbstractLLM]:
    """Build an LLM factory for the actor system."""
    if provider == "openai":
        def factory(_: str) -> AbstractLLM:
            return OpenAIChatLLM(model=model, temperature=temperature, seed=seed)
        return factory
    elif provider == "anthropic":
        def factory(_: str) -> AbstractLLM:
            return AnthropicChatLLM(model=model, temperature=temperature)
        return factory
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def run_input(
    user_actor: UserActor,
    bus: MessageBus,
    text: str,
) -> tuple[str, TokenAccumulator, int]:
    """Send a single input and return (response, token_accumulator, latency_ms)."""
    # Clear last_usage on all LLMs before the call
    for actor in bus.actors.values():
        llm = getattr(actor, 'llm', None)
        if llm and isinstance(llm, AbstractLLM):
            llm.last_usage = None

    acc = TokenAccumulator()
    t0 = time.perf_counter()
    response = user_actor.send_user_message(text)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Collect tokens from all actors
    acc.collect_all(bus)

    return response, acc, latency_ms


def run_test_case(
    test_case: TestCase,
    bus: MessageBus,
    user_actor: UserActor,
) -> List[EvalRecord]:
    """Run a single test case through the actor system and return eval records."""
    records: List[EvalRecord] = []
    tc_id = test_case.id

    # Phase 1: Feed steps (grounding)
    for i, step in enumerate(test_case.steps):
        step_text = f"Workflow step: {step}"
        response, acc, latency_ms = run_input(user_actor, bus, step_text)
        records.append(EvalRecord(
            test_case_id=tc_id,
            input_id=f"S{i+1:03d}",
            input_type="step",
            input_text=step_text,
            actual_response=response,
            tokens=acc.to_token_usage(),
            latency_ms=latency_ms,
            actor_states=get_actor_states(bus),
        ))

    # Phase 2: Merge and sort modifications + events by `when`, then feed in order
    timeline: List[tuple[str, Any]] = []  # (when, item)
    for mod in test_case.modifications:
        timeline.append((mod.when, ("modification", mod)))
    for evt in test_case.events:
        timeline.append((evt.when, ("event", evt)))

    # Sort by `when` field (format: W02-3T16:00)
    timeline.sort(key=lambda x: x[0])

    for _when, (item_type, item) in timeline:
        if item_type == "modification":
            input_text = f"Rule change: {item.intent}"
            response, acc, latency_ms = run_input(user_actor, bus, input_text)
            records.append(EvalRecord(
                test_case_id=tc_id,
                input_id=item.id,
                input_type="modification",
                input_text=input_text,
                when=item.when,
                actual_response=response,
                tokens=acc.to_token_usage(),
                latency_ms=latency_ms,
                actor_states=get_actor_states(bus),
            ))
        else:  # event
            input_text = f"[{item.source}] {item.input}"
            response, acc, latency_ms = run_input(user_actor, bus, input_text)
            records.append(EvalRecord(
                test_case_id=tc_id,
                input_id=item.id,
                input_type="event",
                input_text=input_text,
                when=item.when,
                actual_response=response,
                expected=ExpectedOutcome(
                    action=item.expect.action,
                    reason=item.expect.reason,
                ),
                tokens=acc.to_token_usage(),
                latency_ms=latency_ms,
                actor_states=get_actor_states(bus),
            ))

    return records


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run evaluation harness on test cases."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to test cases JSONL file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output JSONL file (default: outputs/eval/results.jsonl)",
    )
    add_common_args(parser)

    args = parser.parse_args()

    # Resolve provider
    provider = args.provider or infer_provider(args.model)
    model = args.model
    temperature = args.temperature
    seed = args.seed
    limit = args.limit

    # Resolve output path
    output_path: Path = args.output or Path("outputs/eval/results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load test cases
    test_cases: List[TestCase] = load_jsonl(args.input, TestCase)
    if limit:
        test_cases = test_cases[:limit]

    print(f"Provider: {provider}, Model: {model}")
    print(f"Input: {args.input} ({len(test_cases)} test case(s))")
    print(f"Output: {output_path}")
    print()

    total_records = 0
    with open(output_path, "w") as out_f:
        for tc_idx, test_case in enumerate(test_cases):
            print(f"[{tc_idx+1}/{len(test_cases)}] Running test case: {test_case.id}")

            # Create a fresh actor system for each test case
            llm_factory = build_llm_factory(provider, model, temperature, seed)
            bus = MessageBus()
            coordinator = CoordinatorActor(
                llm=llm_factory("Coordinator"),
                llm_factory=llm_factory,
            )
            bus.register(coordinator)
            user_actor = UserActor()
            bus.register(user_actor)

            records = run_test_case(test_case, bus, user_actor)

            for record in records:
                out_f.write(record.model_dump_json() + "\n")
            out_f.flush()

            total_records += len(records)
            print(f"  -> {len(records)} records written")

    print(f"\nDone. {total_records} total records written to {output_path}")


if __name__ == "__main__":
    main()
