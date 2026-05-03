"""
Unified LLM client. All agents go through here so retry, tracing,
and token accounting are handled in one place.
"""

import os
import anthropic
from typing import Any


class LLMClient:
    def __init__(self, config: dict):
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"].get("max_tokens", 2048)
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    def complete(self, messages: list[dict], system: str = "") -> str:
        """Single chat completion. Returns response text."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def complete_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> tuple[str, str | None, dict | None]:
        """
        Tool-use completion.
        Returns (stop_reason, tool_name, tool_input).
        stop_reason is "tool_use" or "end_turn".
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        if response.stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            return "tool_use", tool_block.name, tool_block.input
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        return "end_turn", None, None
