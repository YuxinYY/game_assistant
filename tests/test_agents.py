"""
Unit tests for agent behavior.
Uses dummy config and mocked LLM to test agent logic without real API calls.
"""

from unittest.mock import patch
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

    def test_merges_multiple_screenshots(self):
        class FakeClassifier:
            def __init__(self):
                self.calls = 0

            def classify(self, _image):
                self.calls += 1
                return ["combat_hud", "skill_tree"][self.calls - 1]

        class FakeParser:
            def __init__(self, payload):
                self.payload = payload

            def extract(self, _image):
                return self.payload

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = ProfileAgent(
                DUMMY_CONFIG,
                vlm_client=object(),
                knowledge_base={
                    "all_spells": ["定身术"],
                    "all_spirits": ["广智"],
                    "all_armors": [],
                    "all_skills_tree": ["闪身"],
                },
                parsers={
                    "classifier": FakeClassifier(),
                    "combat_hud": FakeParser({"chapter": 2, "equipped_spirit": "广智"}),
                    "skill_tree": FakeParser({"unlocked_skills": ["闪身"], "unlocked_spells": ["定身术"]}),
                },
            )
            state = AgentState(
                user_query="看图",
                user_screenshots=[b"hud", b"skill"],
                player_profile=PlayerProfile(),
            )

            result = agent.execute(state)

        assert result.player_profile.chapter == 2
        assert result.player_profile.equipped_spirit == "广智"
        assert result.player_profile.unlocked_skills == ["闪身"]
        assert result.player_profile.unlocked_spells == ["定身术"]
        assert len(result.profile_updates) == 4

    def test_conversational_update_corrects_profile(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = ProfileAgent(
                DUMMY_CONFIG,
                vlm_client=object(),
                knowledge_base={
                    "all_spells": ["定身术"],
                    "all_spirits": ["青背龙"],
                    "all_armors": ["行者套装"],
                    "all_skills_tree": ["闪身"],
                },
            )
            state = AgentState(
                user_query="我在第2章，我带的是青背龙，用的是定身术和行者套装",
                player_profile=PlayerProfile(),
            )

            result = agent.execute(state)

        assert result.player_profile.chapter == 2
        assert result.player_profile.equipped_spirit == "青背龙"
        assert result.player_profile.equipped_spells == ["定身术"]
        assert result.player_profile.equipped_armor == ["行者套装"]
        assert len(result.profile_updates) >= 3


class TestAnalysisAgent:
    def test_empty_docs_returns_empty_analysis(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = AnalysisAgent.__new__(AnalysisAgent)
            agent.config = DUMMY_CONFIG
            state = make_state()
            state.retrieved_docs = []
            result = agent.execute(state)
            assert result.consensus_analysis == {"strategies": [], "conflicts": []}
