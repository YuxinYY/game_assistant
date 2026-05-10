"""
Unit tests for agent behavior.
Uses dummy config and mocked LLM to test agent logic without real API calls.
"""

from unittest.mock import patch
from src.core.state import AgentState, PlayerProfile, Document
from src.agents.community_agent import CommunityAgent
from src.agents.profile_agent import ProfileAgent, _filter_by_profile
from src.agents.analysis_agent import AnalysisAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.wiki_agent import WikiAgent


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
    def test_does_not_filter_docs_when_chapter_is_unset(self):
        docs = [make_doc("推荐用广智变身", chapter=1)]
        profile = PlayerProfile(chapter=None)

        filtered = _filter_by_profile(docs, profile)

        assert len(filtered) == 1

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
        class FakeVisionClient:
            def supports_vision(self):
                return True

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
                vlm_client=FakeVisionClient(),
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

    def test_screenshot_combat_hud_can_seed_boss_entity(self):
        class FakeVisionClient:
            def supports_vision(self):
                return True

        class FakeClassifier:
            def classify(self, _image):
                return "combat_hud"

        class FakeParser:
            def extract(self, _image):
                return {
                    "current_boss": "虎先锋",
                    "chapter": 2,
                    "confidence": 0.91,
                }

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = ProfileAgent(
                DUMMY_CONFIG,
                vlm_client=FakeVisionClient(),
                knowledge_base={
                    "all_spells": [],
                    "all_spirits": [],
                    "all_armors": [],
                    "all_skills_tree": [],
                },
                parsers={
                    "classifier": FakeClassifier(),
                    "combat_hud": FakeParser(),
                },
            )
            state = AgentState(
                user_query="看图",
                user_screenshots=[b"hud"],
                player_profile=PlayerProfile(),
            )

            result = agent.execute(state)

        assert result.identified_entities == ["虎先锋"]
        assert result.player_profile.chapter == 2
        assert any(event.action == "parse_combat_hud" for event in result.trace)

    def test_other_screenshot_can_seed_boss_entity_via_generic_visual_detector(self):
        class FakeVisionClient:
            vision_provider = "anthropic"
            vision_model = "claude-sonnet-4-7"

            def supports_vision(self):
                return True

            def vision_json(self, image_bytes=None, prompt=""):
                return {"visible_entity": "虎先锋", "entity_type": "boss", "confidence": 0.88}

        class FakeClassifier:
            def classify(self, _image):
                return "other"

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = ProfileAgent(
                DUMMY_CONFIG,
                vlm_client=FakeVisionClient(),
                knowledge_base={
                    "all_spells": [],
                    "all_spirits": [],
                    "all_armors": [],
                    "all_skills_tree": [],
                },
                parsers={
                    "classifier": FakeClassifier(),
                },
            )
            state = AgentState(
                user_query="看图",
                user_screenshots=[b"scene"],
                player_profile=PlayerProfile(),
            )

            result = agent.execute(state)

        assert result.identified_entities == ["虎先锋"]
        assert any(event.action == "vision_context" for event in result.trace)
        assert any(event.action == "classify_screenshot" for event in result.trace)

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

    def test_screenshot_flow_gracefully_skips_when_vision_unavailable(self):
        class NoVisionClient:
            def supports_vision(self):
                return False

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = ProfileAgent(
                DUMMY_CONFIG,
                vlm_client=NoVisionClient(),
                knowledge_base={
                    "all_spells": [],
                    "all_spirits": [],
                    "all_armors": [],
                    "all_skills_tree": [],
                },
            )
            state = AgentState(
                user_query="看图",
                user_screenshots=[b"hud"],
                player_profile=PlayerProfile(),
            )

            result = agent.execute(state)

        assert result.profile_updates == []
        assert result.trace[0].action == "vision_unavailable"


class TestAnalysisAgent:
    def test_empty_docs_returns_empty_analysis(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = AnalysisAgent.__new__(AnalysisAgent)
            agent.config = DUMMY_CONFIG
            state = make_state()
            state.retrieved_docs = []
            result = agent.execute(state)
            assert result.consensus_analysis == {"strategies": [], "conflicts": []}


class TestWikiAgent:
    def test_execute_populates_docs_and_entities(self):
        docs = [
            make_doc(
                "虎跃斩第三段要侧闪",
                source="wiki",
                url="http://wiki/1",
                chapter=1,
            )
        ]
        docs[0].entity = "虎先锋"

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.wiki_agent.wiki_search", return_value=docs
        ):
            agent = WikiAgent(DUMMY_CONFIG)
            state = make_state("虎先锋那个招怎么躲？")

            result = agent.execute(state)

        assert len(result.retrieved_docs) == 1
        assert result.retrieved_docs[0].source == "wiki"
        assert result.identified_entities == ["虎先锋"]
        assert result.trace[0].action.startswith("wiki_search")

    def test_execute_falls_back_to_exact_lookup_when_entity_query_search_is_empty(self):
        exact_match = {
            "entity": "虎先锋",
            "text": "虎先锋共有多种高威胁招式，包括乌鸦坐飞机、血龙卷和虎突猛进。",
            "url": "http://wiki/tiger-vanguard",
        }

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.wiki_agent.infer_wiki_entity", return_value="虎先锋"
        ), patch("src.agents.wiki_agent.wiki_search", return_value=[]), patch(
            "src.agents.wiki_agent.entity_lookup", return_value=exact_match
        ):
            agent = WikiAgent(DUMMY_CONFIG)
            state = make_state("虎先锋有几个大招")

            result = agent.execute(state)

        assert result.identified_entities == ["虎先锋"]
        assert len(result.retrieved_docs) == 1
        assert result.retrieved_docs[0].url == "http://wiki/tiger-vanguard"
        assert [event.action for event in result.trace[:2]] == [
            "wiki_search({'query': '虎先锋有几个大招', 'entity': '虎先锋'})",
            "entity_lookup({'entity': '虎先锋', 'query': '虎先锋有几个大招'})",
        ]


class TestCommunityAgent:
    def test_execute_uses_unfiltered_fallback_when_chapter_filtered_search_is_empty(self):
        docs = [
            make_doc(
                "虎跃斩第三段向左闪",
                source="nga",
                chapter=None,
                url="http://nga/1",
            )
        ]

        def fake_nga_search(query, chapter_filter=None):
            if chapter_filter is not None:
                return []
            return docs

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.nga_search", side_effect=fake_nga_search
        ), patch("src.agents.community_agent.bilibili_search", return_value=[]), patch(
            "src.agents.community_agent.reddit_search", return_value=[]
        ):
            agent = CommunityAgent(DUMMY_CONFIG)
            state = make_state("虎先锋那个招怎么躲？", chapter=1)
            state.identified_entities = ["虎先锋"]

            result = agent.execute(state)

        assert len(result.retrieved_docs) == 1
        assert any(event.action.startswith("query_rewrite") for event in result.trace)
        assert any(event.action.startswith("nga_search") for event in result.trace)


class TestSynthesisAgent:
    def test_execute_uses_llm_output_and_appends_citations(self):
        docs = [
            make_doc(
                "虎跃斩第三段建议向左侧闪，时机是前爪落地。",
                source="nga",
                chapter=1,
                url="http://nga/1",
            )
        ]
        docs[0].metadata = {"author": "玩家A"}

        class FakeLLM:
            def complete(self, messages, system=""):
                return "## 招式识别\n- 虎跃斩（来源: nga http://nga/1）"

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FakeLLM()
            state = make_state("虎先锋那个招怎么躲？")
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert "## 招式识别" in result.final_answer
        assert "## 参考来源" in result.final_answer
        assert len(result.citations) == 1
        assert result.trace[0].action == "synthesis_llm"

    def test_execute_fact_lookup_uses_fact_answer_format(self):
        docs = [
            make_doc(
                "虎先锋共有多种高威胁招式，包括血龙卷与虎突猛进。",
                source="wiki",
                chapter=2,
                url="http://wiki/1",
            )
        ]

        class FakeLLM:
            def complete(self, messages, system=""):
                assert "## 直接结论" in system
                assert "针对你的 build 的建议" not in system
                return "## 直接结论\n- 虎先锋至少有多种高威胁大招。\n\n## 依据\n- wiki 已列出血龙卷与虎突猛进。"

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FakeLLM()
            state = make_state("虎先锋有几个大招？")
            state.workflow = "fact_lookup"
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert "## 直接结论" in result.final_answer
        assert "## 依据" in result.final_answer
        assert "## 针对你的 build 的建议" not in result.final_answer

    def test_execute_falls_back_to_extractive_summary_when_llm_fails(self):
        docs = [
            make_doc(
                "虎跃斩第三段向左侧闪成功率更高。",
                source="nga",
                chapter=1,
                url="http://nga/1",
            )
        ]

        class FailingLLM:
            def complete(self, messages, system=""):
                raise RuntimeError("provider error")

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FailingLLM()
            state = make_state("虎先锋那个招怎么躲？")
            state.retrieved_docs = docs
            state.identified_entities = ["虎先锋"]
            state.consensus_analysis = {"strategies": [], "conflicts": []}

            result = agent.execute(state)

        assert "## 基于检索结果的直接整理" in result.final_answer
        assert "provider error" in result.final_answer
        assert "http://nga/1" in result.final_answer
        assert result.trace[0].action == "synthesis_fallback"

    def test_execute_fact_lookup_fallback_uses_fact_sections(self):
        docs = [
            make_doc(
                "虎先锋招式包括血龙卷、虎突猛进和卧虎石。",
                source="wiki",
                chapter=2,
                url="http://wiki/1",
            )
        ]

        class FailingLLM:
            def complete(self, messages, system=""):
                raise RuntimeError("provider error")

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FailingLLM()
            state = make_state("虎先锋有几个大招？")
            state.workflow = "fact_lookup"
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert "## 直接结论" in result.final_answer
        assert "## 依据" in result.final_answer
        assert "## 针对你的 build 的建议" not in result.final_answer
        assert "provider error" in result.final_answer
        assert "[原文](http://wiki/1)" in result.final_answer
        assert "（http://wiki/1）" not in result.final_answer

    def test_execute_returns_no_results_message_without_docs(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            state = make_state("虎先锋那个招怎么躲？")

            result = agent.execute(state)

        assert "检索内容中未找到足够资料" in result.final_answer
        assert result.citations == []
        assert result.trace[0].action == "synthesis_no_results"

    def test_execute_deduplicates_docs_across_rewritten_queries(self):
        docs = [
            make_doc(
                "推荐侧闪",
                source="nga",
                chapter=1,
                url="http://nga/1",
            )
        ]

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.nga_search", return_value=docs
        ), patch("src.agents.community_agent.bilibili_search", return_value=docs), patch(
            "src.agents.community_agent.reddit_search", return_value=[]
        ):
            agent = CommunityAgent(DUMMY_CONFIG)
            state = make_state("虎先锋怎么打？", chapter=1)
            state.identified_entities = ["虎先锋"]

            result = agent.execute(state)

        assert len(result.retrieved_docs) == 1
