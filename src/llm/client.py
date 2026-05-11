"""
Unified LLM client. All agents go through here so retry, tracing,
and token accounting are handled in one place.
"""

import base64
import json
import os
import time
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
    def __init__(self, config: dict, provider_override: str | None = None, model_override: str | None = None):
        if load_dotenv is not None:
            load_dotenv()
        self.provider = (provider_override or self._resolve_provider(config)).strip().lower()
        self.model = model_override or self._resolve_model(config, provider=self.provider)
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"].get("max_tokens", 2048)
        self.retry_attempts = max(0, int(config["llm"].get("retry_attempts", 2)))
        self.retry_base_delay_seconds = max(0.0, float(config["llm"].get("retry_base_delay_seconds", 1.0)))
        self._client = None
        if self.provider not in {"anthropic", "groq"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        self._client = self._build_provider_client(self.provider, self._resolve_api_key(self.provider))

        self.vision_provider = self._resolve_vision_provider(config)
        self.vision_model = self._resolve_vision_model(config)
        self._vision_client = self._resolve_vision_client()

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
        if not self.supports_vision():
            raise NotImplementedError(
                f"Vision calls are not implemented for provider '{self.vision_provider}'."
            )
        media_type = self._infer_media_type(image_bytes)
        encoded = base64.standard_b64encode(image_bytes).decode()
        response = self._vision_client.messages.create(
            model=model or self.vision_model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
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

    def supports_vision(self) -> bool:
        return self.vision_provider == "anthropic" and self._vision_client is not None

    def is_available(self) -> bool:
        return self._client is not None

    def _resolve_provider(self, config: dict) -> str:
        configured = os.getenv("LLM_PROVIDER") or config["llm"].get("provider")
        if configured:
            return configured.strip().lower()
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        return "anthropic"

    def _resolve_model(self, config: dict, provider: str | None = None) -> str:
        provider_name = provider or self.provider
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            return env_model
        if provider_name == "groq":
            return os.getenv("GROQ_MODEL") or config["llm"].get("groq_model") or "llama-3.1-8b-instant"
        return config["llm"]["model"]

    def _resolve_api_key(self, provider: str) -> str:
        if provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        return os.getenv("ANTHROPIC_API_KEY", "")

    def _resolve_vision_provider(self, config: dict) -> str:
        configured = os.getenv("VLM_PROVIDER") or config["llm"].get("vision_provider")
        if configured:
            return configured.strip().lower()
        if self.provider == "anthropic":
            return "anthropic"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return self.provider

    def _resolve_vision_model(self, config: dict) -> str:
        env_model = os.getenv("VLM_MODEL")
        if env_model:
            return env_model
        if self.vision_provider == self.provider:
            return self.model
        if self.vision_provider == "groq":
            return os.getenv("GROQ_MODEL") or config["llm"].get("groq_model") or "llama-3.1-8b-instant"
        return config["llm"].get("vision_model") or config["llm"].get("model") or self.model

    def _resolve_vision_client(self):
        if self.vision_provider not in {"anthropic", "groq"}:
            return None
        if self.vision_provider == self.provider:
            return self._client
        return self._build_provider_client(self.vision_provider, self._resolve_api_key(self.vision_provider))

    def _build_provider_client(self, provider: str, api_key: str):
        if provider == "anthropic":
            if anthropic is not None and api_key:
                return anthropic.Anthropic(api_key=api_key)
            return None
        if provider == "groq":
            if api_key:
                session = requests.Session()
                session.headers.update(
                    {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                )
                return session
            return None
        return None

    def _groq_complete(self, messages: list[dict], system: str = "") -> str:
        self._require_client()
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": self._compose_chat_messages(messages, system),
        }
        max_attempts = self.retry_attempts + 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self._client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"].get("content", "")
            except requests.HTTPError as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                if not self._should_retry_groq_http_error(response, attempt, max_attempts):
                    raise
                time.sleep(self._retry_delay_seconds(attempt, response))
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    raise
                time.sleep(self._retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Groq completion failed without returning a response.")

    def _should_retry_groq_http_error(self, response, attempt: int, max_attempts: int) -> bool:
        if attempt >= max_attempts - 1:
            return False
        status_code = getattr(response, "status_code", None)
        return status_code == 429 or (status_code is not None and status_code >= 500)

    def _retry_delay_seconds(self, attempt: int, response=None) -> float:
        retry_after = None
        headers = getattr(response, "headers", {}) or {}
        if isinstance(headers, dict):
            retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError):
                pass
        return self.retry_base_delay_seconds * (2 ** attempt)

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

    @staticmethod
    def _infer_media_type(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"
