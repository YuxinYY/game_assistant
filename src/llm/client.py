"""
Unified LLM client. All agents go through here so retry, tracing,
and token accounting are handled in one place.
"""

import base64
import json
import os
import time
from collections.abc import Mapping
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

try:
    import streamlit as st
except ImportError:  # pragma: no cover - depends on local environment
    st = None


_STREAMLIT_SECRET_PATHS: dict[str, tuple[tuple[str, str], ...]] = {
    "LLM_PROVIDER": (("llm", "provider"),),
    "LLM_MODEL": (("llm", "model"),),
    "VLM_PROVIDER": (("llm", "vision_provider"),),
    "VLM_MODEL": (("llm", "vision_model"),),
    "OPENAI_API_KEY": (("openai", "api_key"),),
    "OPENAI_MODEL": (("openai", "model"),),
    "GROQ_API_KEY": (("groq", "api_key"),),
    "GROQ_MODEL": (("groq", "model"),),
    "ANTHROPIC_API_KEY": (("anthropic", "api_key"),),
}


def _runtime_secret(key: str) -> str | None:
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    if st is None:
        return None
    try:
        secrets = st.secrets
    except Exception:  # pragma: no cover - depends on runtime context
        return None
    value = _resolve_streamlit_secret_value(secrets, key)
    if value in (None, ""):
        return None
    return str(value)


def _resolve_streamlit_secret_value(secrets: Any, key: str) -> Any:
    for candidate in (key, key.lower()):
        value = _mapping_get(secrets, candidate)
        if value not in (None, ""):
            return value

    for section_name, field_name in _STREAMLIT_SECRET_PATHS.get(key, ()):
        section = _mapping_get(secrets, section_name)
        if not isinstance(section, Mapping):
            continue
        for candidate in (field_name, field_name.upper()):
            value = _mapping_get(section, candidate)
            if value not in (None, ""):
                return value
    return None


def _mapping_get(container: Any, key: str) -> Any:
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return container[key]
    except Exception:  # pragma: no cover - defensive
        return None


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
        self.request_timeout_seconds = self._resolve_request_timeout_seconds(config)
        self._client = None
        if self.provider not in {"anthropic", "groq", "openai"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        self._client = self._build_provider_client(self.provider, self._resolve_api_key(self.provider))

        self.vision_provider = self._resolve_vision_provider(config)
        self.vision_model = self._resolve_vision_model(config)
        self._vision_client = self._resolve_vision_client()

    def complete(self, messages: list[dict], system: str = "") -> str:
        """Single chat completion. Returns response text."""
        if self.provider in {"groq", "openai"}:
            return self._openai_compatible_complete(messages, system=system)
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
        configured = _runtime_secret("LLM_PROVIDER") or config["llm"].get("provider")
        if configured:
            return configured.strip().lower()
        if _runtime_secret("ANTHROPIC_API_KEY"):
            return "anthropic"
        if _runtime_secret("GROQ_API_KEY"):
            return "groq"
        if _runtime_secret("OPENAI_API_KEY"):
            return "openai"
        return "anthropic"

    def _resolve_model(self, config: dict, provider: str | None = None) -> str:
        provider_name = provider or self.provider
        env_model = _runtime_secret("LLM_MODEL")
        if env_model:
            return env_model
        if provider_name == "groq":
            return _runtime_secret("GROQ_MODEL") or config["llm"].get("groq_model") or "llama-3.1-8b-instant"
        if provider_name == "openai":
            return _runtime_secret("OPENAI_MODEL") or config["llm"].get("openai_model") or "gpt-4o-mini"
        return config["llm"]["model"]

    def _resolve_api_key(self, provider: str) -> str:
        if provider == "groq":
            return _runtime_secret("GROQ_API_KEY") or ""
        if provider == "openai":
            return _runtime_secret("OPENAI_API_KEY") or ""
        return _runtime_secret("ANTHROPIC_API_KEY") or ""

    def _resolve_vision_provider(self, config: dict) -> str:
        configured = _runtime_secret("VLM_PROVIDER") or config["llm"].get("vision_provider")
        if configured:
            return configured.strip().lower()
        if self.provider == "anthropic":
            return "anthropic"
        if _runtime_secret("ANTHROPIC_API_KEY"):
            return "anthropic"
        return self.provider

    def _resolve_vision_model(self, config: dict) -> str:
        env_model = _runtime_secret("VLM_MODEL")
        if env_model:
            return env_model
        if self.vision_provider == self.provider:
            return self.model
        if self.vision_provider == "groq":
            return _runtime_secret("GROQ_MODEL") or config["llm"].get("groq_model") or "llama-3.1-8b-instant"
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
        if provider in {"groq", "openai"}:
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

    def _openai_compatible_complete(self, messages: list[dict], system: str = "") -> str:
        self._require_client()
        payload = self._build_openai_compatible_payload(messages, system)
        max_attempts = self.retry_attempts + 1
        last_error = None
        endpoint = self._chat_completion_endpoint()

        for attempt in range(max_attempts):
            try:
                response = self._client.post(
                    endpoint,
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"].get("content", "")
            except requests.HTTPError as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                if not self._should_retry_openai_compatible_http_error(response, attempt, max_attempts):
                    raise
                time.sleep(self._retry_delay_seconds(attempt, response))
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    raise
                time.sleep(self._retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{self.provider} completion failed without returning a response.")

    def _should_retry_openai_compatible_http_error(self, response, attempt: int, max_attempts: int) -> bool:
        if attempt >= max_attempts - 1:
            return False
        status_code = getattr(response, "status_code", None)
        return status_code == 429 or (status_code is not None and status_code >= 500)

    def _build_openai_compatible_payload(self, messages: list[dict], system: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._compose_chat_messages(messages, system),
        }

        if self._uses_openai_reasoning_compat_mode():
            payload["max_completion_tokens"] = self.max_tokens
            if self.temperature == 1:
                payload["temperature"] = self.temperature
            return payload

        payload["temperature"] = self.temperature
        payload["max_tokens"] = self.max_tokens
        return payload

    def _uses_openai_reasoning_compat_mode(self) -> bool:
        return self.provider == "openai" and self.model.startswith("gpt-5")

    def _resolve_request_timeout_seconds(self, config: dict) -> float:
        configured = config.get("llm", {}).get("request_timeout_seconds")
        if configured is not None:
            try:
                return max(float(configured), 1.0)
            except (TypeError, ValueError):
                pass
        if self.provider == "openai" and self.model.startswith("gpt-5"):
            return 90.0
        return 30.0

    def _chat_completion_endpoint(self) -> str:
        if self.provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        if self.provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        raise RuntimeError(f"Provider '{self.provider}' does not use an OpenAI-compatible chat completions endpoint.")

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
            env_var = {
                "groq": "GROQ_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(self.provider, "API_KEY")
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
