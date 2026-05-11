"""
Targeted tests for provider selection and Groq text completion support.
"""

import requests

import src.llm.client as llm_client_module
from src.llm.client import LLMClient


DUMMY_CONFIG = {
    "llm": {
        "model": "claude-sonnet-4-7",
        "temperature": 0.3,
        "max_tokens": 64,
    }
}


def _clear_llm_env(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "VLM_PROVIDER",
        "VLM_MODEL",
        "GROQ_MODEL",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestLLMClient:
    def test_selects_groq_provider_from_env(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client = LLMClient(DUMMY_CONFIG)

        assert client.provider == "groq"
        assert client.model == "llama-3.1-8b-instant"

    def test_provider_override_takes_precedence_for_secondary_client(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        class FakeAnthropicClient:
            def __init__(self, api_key):
                self.api_key = api_key

        class FakeAnthropicModule:
            Anthropic = FakeAnthropicClient

        monkeypatch.setattr(llm_client_module, "anthropic", FakeAnthropicModule)

        client = LLMClient(
            DUMMY_CONFIG,
            provider_override="anthropic",
            model_override="claude-sonnet-4-7",
        )

        assert client.provider == "anthropic"
        assert client.model == "claude-sonnet-4-7"
        assert client.is_available() is True

    def test_groq_complete_uses_openai_compatible_payload(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")

        client = LLMClient(DUMMY_CONFIG)
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "boss_strategy"}}
                    ]
                }

        class FakeSession:
            def post(self, url, json, timeout):
                captured["url"] = url
                captured["json"] = json
                captured["timeout"] = timeout
                return FakeResponse()

        client._client = FakeSession()
        result = client.complete(
            [{"role": "user", "content": "虎先锋怎么打？"}],
            system="你是分类器。",
        )

        assert result == "boss_strategy"
        assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert captured["json"]["model"] == "llama-3.1-8b-instant"
        assert captured["json"]["messages"][0] == {
            "role": "system",
            "content": "你是分类器。",
        }
        assert captured["json"]["messages"][1] == {
            "role": "user",
            "content": "虎先锋怎么打？",
        }

    def test_groq_complete_retries_once_after_rate_limit(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        monkeypatch.setattr(llm_client_module.time, "sleep", lambda _seconds: None)

        client = LLMClient(DUMMY_CONFIG)
        calls = {"count": 0}

        class RateLimitedResponse:
            status_code = 429
            headers = {"Retry-After": "0"}

            def raise_for_status(self):
                raise requests.HTTPError(
                    "429 Client Error: Too Many Requests",
                    response=self,
                )

        class SuccessResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "boss_strategy"}}]}

        class FakeSession:
            def post(self, url, json, timeout):
                calls["count"] += 1
                if calls["count"] == 1:
                    return RateLimitedResponse()
                return SuccessResponse()

        client._client = FakeSession()
        result = client.complete(
            [{"role": "user", "content": "虎先锋怎么打？"}],
            system="你是分类器。",
        )

        assert result == "boss_strategy"
        assert calls["count"] == 2

    def test_vision_falls_back_to_anthropic_when_text_provider_is_groq(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        class FakeAnthropicClient:
            def __init__(self, api_key):
                self.api_key = api_key

        class FakeAnthropicModule:
            Anthropic = FakeAnthropicClient

        monkeypatch.setattr(llm_client_module, "anthropic", FakeAnthropicModule)

        client = LLMClient(DUMMY_CONFIG)

        assert client.provider == "groq"
        assert client.vision_provider == "anthropic"
        assert client.supports_vision() is True

    def test_vision_uses_explicit_vlm_provider(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("VLM_PROVIDER", "anthropic")
        monkeypatch.setenv("VLM_MODEL", "claude-vision-test")

        class FakeAnthropicClient:
            def __init__(self, api_key):
                self.api_key = api_key

        class FakeAnthropicModule:
            Anthropic = FakeAnthropicClient

        monkeypatch.setattr(llm_client_module, "anthropic", FakeAnthropicModule)

        client = LLMClient(DUMMY_CONFIG)

        assert client.vision_provider == "anthropic"
        assert client.vision_model == "claude-vision-test"