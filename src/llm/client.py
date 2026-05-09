"""
Unified LLM client. All agents go through here so retry, tracing,
and token accounting are handled in one place.
"""

import base64
import json
import os
from typing import Any

import requests

try:
    import anthropic
except ImportError:  # pragma: no cover - depends on local environment
    anthropic = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on local environment
    load_dotenv = None


class LLMClient:
    def __init__(self, config: dict):
        if load_dotenv is not None:
            load_dotenv()
        self.provider = self._resolve_provider(config)
        self.model = self._resolve_model(config)
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"].get("max_tokens", 2048)
        self._client = None
        api_key = self._resolve_api_key()
        if self.provider == "anthropic":
            if anthropic is not None and api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "groq":
            if api_key:
                session = requests.Session()
                session.headers.update(
                    {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                )
                self._client = session
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(self, messages: list[dict], system: str = "") -> str:
        """Single chat completion. Returns response text."""
        if self.provider == "groq":
            return self._groq_complete(messages, system=system)
        self._require_client()
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
        if self.provider != "anthropic":
            raise NotImplementedError(
                f"Tool-use completion is not implemented for provider '{self.provider}'."
            )
        self._require_client()
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

    def vision_json(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self.provider != "anthropic":
            raise NotImplementedError(
                f"Vision calls are not implemented for provider '{self.provider}'."
            )
        self._require_client()
        encoded = base64.standard_b64encode(image_bytes).decode()
        response = self._client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        return self._extract_json_object(text)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return {}
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

    def _resolve_provider(self, config: dict) -> str:
        configured = os.getenv("LLM_PROVIDER") or config["llm"].get("provider")
        if configured:
            return configured.strip().lower()
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        return "anthropic"

    def _resolve_model(self, config: dict) -> str:
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            return env_model
        if self.provider == "groq":
            return os.getenv("GROQ_MODEL") or config["llm"].get("groq_model") or "llama-3.1-8b-instant"
        return config["llm"]["model"]

    def _resolve_api_key(self) -> str:
        if self.provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        return os.getenv("ANTHROPIC_API_KEY", "")

    def _groq_complete(self, messages: list[dict], system: str = "") -> str:
        self._require_client()
        response = self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self._compose_chat_messages(messages, system),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"].get("content", "")

    @staticmethod
    def _compose_chat_messages(messages: list[dict], system: str = "") -> list[dict]:
        if not system:
            return messages
        return [{"role": "system", "content": system}, *messages]

    def _require_client(self) -> None:
        if self._client is None:
            env_var = "GROQ_API_KEY" if self.provider == "groq" else "ANTHROPIC_API_KEY"
            raise RuntimeError(
                f"LLM client for provider '{self.provider}' is unavailable. "
                f"Check dependencies and set {env_var}."
            )
