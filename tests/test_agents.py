"""
Unit tests for agent behavior.
Uses dummy config and mocked LLM to test agent logic without real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.core.state import AgentState, PlayerProfile, Document
from src.agents.profile_agent import ProfileAgent, _filter_by_profile
from src.agents.analysis_agent import AnalysisAgent


DUMMY_CONFIG = {
    "llm": {"model": "claude-sonnet-4-7", "temperature": 0.3, "max_tokens": 512},
    "agents": {"max_react_iterations": 3},
    "spoiler": {"enable": True},
}


def make_state(query="test", chapter=1) -> AgentState:
    return AgentState(
        user_query=query,
        player_profile=PlayerProfile(chapter=chapter),
    )


def make_doc(text: str, source: str = "nga", chapter: int = 1, url: str = "http://x") -> Document:
    return Document(text=text, source=source, url=url, chapter=chapter)


class TestProfileAgent:
    def test_filters_out_locked_transformation(self):
        # Player is in chapter 1; doc recommends 广智 (chapter 3)
        docs = [
            make_doc("推荐用广智变身", chapter=1),
            make_doc("侧向闪避是最稳的", chapter=1),
        ]
        profile = PlayerProfile(chapter=1)
        filtered = _filter_by_profile(docs, profile)
        texts = [d.text for d in filtered]
        assert "推荐用广智变身" not in texts
        assert "侧向闪避是最稳的" in texts

    def test_allows_unlocked_skill(self):
        docs = [make_doc("可以用定身术", chapter=1)]
        profile = PlayerProfile(chapter=1, unlocked_skills=["定身术"])
        filtered = _filter_by_profile(docs, profile)
        assert len(filtered) == 1


class TestAnalysisAgent:
    def test_empty_docs_returns_empty_analysis(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = AnalysisAgent.__new__(AnalysisAgent)
            agent.config = DUMMY_CONFIG
            state = make_state()
            state.retrieved_docs = []
            result = agent.execute(state)
            assert result.consensus_analysis == {"strategies": [], "conflicts": []}
