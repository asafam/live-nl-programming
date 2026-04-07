"""Google Gemini LLM client for data generation."""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Type

from pydantic import BaseModel

from .base import AbstractLLM, ChatMessage

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


class GeminiChatLLM(AbstractLLM):
    """Google Gemini client with structured output support."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> None:
        if genai is None:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = genai.Client(api_key=api_key)

    def _to_gemini_contents(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[Optional[str], list]:
        """Convert ChatMessage sequence to Gemini contents format.

        Returns:
            Tuple of (system_instruction, contents_list).
        """
        system_parts = []
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                # Gemini uses "model" instead of "assistant"
                role = "model" if msg.role == "assistant" else msg.role
                contents.append(
                    genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=msg.content)],
                    )
                )

        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        response_model: Type[BaseModel],
    ) -> BaseModel:
        system_instruction, contents = self._to_gemini_contents(messages)

        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=response_model,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        if response.parsed is not None:
            return response.parsed

        # Fallback: parse text manually
        text = response.text or "{}"
        parsed = json.loads(text)
        return response_model(**parsed)

    def generate_text(self, messages: Sequence[ChatMessage]) -> str:
        system_instruction, contents = self._to_gemini_contents(messages)

        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""
