"""LLM utilities for data generation."""
from typing import Optional

from .base import AbstractLLM, ChatMessage, user_message, system_message, assistant_message
from .openai_client import OpenAIChatLLM
from .anthropic_client import AnthropicChatLLM


def create_llm(
    provider: str,
    model: str,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> AbstractLLM:
    """Create an LLM client for the specified provider.

    Args:
        provider: LLM provider - "openai" or "anthropic".
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        temperature: Sampling temperature.
        seed: Random seed for reproducibility (OpenAI only).

    Returns:
        Configured LLM client.

    Raises:
        ValueError: If provider is not recognized.
    """
    if provider == "openai":
        return OpenAIChatLLM(
            model=model,
            temperature=temperature,
            seed=seed,
        )
    elif provider == "anthropic":
        return AnthropicChatLLM(
            model=model,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Use 'openai' or 'anthropic'."
        )


__all__ = [
    "AbstractLLM",
    "ChatMessage",
    "OpenAIChatLLM",
    "AnthropicChatLLM",
    "create_llm",
    "user_message",
    "system_message",
    "assistant_message",
]
