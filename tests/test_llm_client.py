"""
Targeted tests for provider selection and Groq text completion support.
"""

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