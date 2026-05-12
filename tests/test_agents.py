"""
Unit tests for agent behavior.
Uses dummy config and mocked LLM to test agent logic without real API calls.
"""

import re
from unittest.mock import patch
from src.core.state import AgentState, PlayerProfile, Document, ExecutionPlan
from src.agents.community_agent import CommunityAgent
from src.agents.profile_agent import ProfileAgent, _filter_by_profile
from src.agents.analysis_agent import AnalysisAgent
from src.agents.synthesis_agent import SynthesisAgent, _build_synthesis_context, _extract_citations
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

    def test_english_query_analysis_trace_observation_stays_in_english(self):
        docs = [
            make_doc(
                "Wait for the delayed slam and punish after recovery.",
                source="reddit",
                chapter=2,
                url="http://reddit/en-1",
            ),
            make_doc(
                "The delayed slam is punishable after the recovery animation.",
                source="wiki",
                chapter=2,
                url="http://wiki/en-1",
            ),
        ]

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.analysis_agent.count_source_consensus",
            return_value=[{"label": "punish after recovery", "source_count": 2, "sources": {"reddit": 1, "wiki": 1}, "is_contested": False}],
        ), patch("src.agents.analysis_agent.detect_conflicts", return_value=[]):
            agent = AnalysisAgent(DUMMY_CONFIG)
            state = make_state("How do I beat Tiger Vanguard?")
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert result.trace[-1].action == "FINISH"
        assert result.trace[-1].observation == "Consensus analysis is complete."
        assert re.search(r"[\u4e00-\u9fff]", result.trace[-1].observation) is None


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

    def test_english_query_trace_observation_stays_in_english(self):
        docs = [
            make_doc(
                "Tiger Vanguard's delayed slam can be punished after recovery.",
                source="wiki",
                url="http://wiki/en-1",
                chapter=2,
            )
        ]
        docs[0].entity = "Tiger Vanguard"

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.wiki_agent.infer_wiki_entity", return_value="Tiger Vanguard"
        ), patch("src.agents.wiki_agent.wiki_search", return_value=docs):
            agent = WikiAgent(DUMMY_CONFIG)
            state = make_state("How do I beat Tiger Vanguard?")

            result = agent.execute(state)

        assert result.trace[-1].action == "FINISH"
        assert result.trace[-1].observation == "Enough wiki evidence has already been collected."
        assert re.search(r"[\u4e00-\u9fff]", result.trace[-1].observation) is None


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

    def test_execute_uses_goal_driven_query_variant_when_initial_rewrite_is_too_broad(self):
        docs = [
            make_doc(
                "虎跃斩第三段建议向左侧闪，时机是前爪落地。",
                source="nga",
                chapter=1,
                url="http://nga/dodge",
            )
        ]
        attempted_queries = []

        def fake_nga_search(query, chapter_filter=None):
            attempted_queries.append(query)
            if "躲避" in query or "闪避" in query:
                return docs
            return []

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.has_indexed_source_documents", return_value=True
        ), patch("src.agents.community_agent.nga_search", side_effect=fake_nga_search), patch(
            "src.agents.community_agent.bilibili_search", return_value=[]
        ), patch("src.agents.community_agent.reddit_search", return_value=[]), patch(
            "src.agents.community_agent.QueryRewriter.rewrite", return_value=["虎先锋 怎么打"]
        ):
            agent = CommunityAgent(DUMMY_CONFIG)
            state = make_state("虎先锋那个招怎么躲？", chapter=1)
            state.identified_entities = ["虎先锋"]
            state.execution_plan = ExecutionPlan(
                workflow="boss_strategy",
                goals=["collect_community_counterplay", "resolve_dodge_timing"],
            )

            result = agent.execute(state)

        assert len(result.retrieved_docs) == 1
        assert any("躲避" in query or "闪避" in query for query in attempted_queries)

    def test_execute_skips_community_search_when_no_indexed_source_is_available(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.has_indexed_source_documents", return_value=False
        ), patch("src.agents.community_agent.nga_search", side_effect=AssertionError("should not search nga")), patch(
            "src.agents.community_agent.bilibili_search", side_effect=AssertionError("should not search bilibili")
        ), patch("src.agents.community_agent.reddit_search", side_effect=AssertionError("should not search reddit")):
            agent = CommunityAgent(DUMMY_CONFIG)
            state = make_state("How do I beat Tiger Vanguard?", chapter=2)
            state.retrieved_docs = [
                make_doc(
                    "Tiger Vanguard's delayed slam can be punished after recovery.",
                    source="wiki",
                    chapter=2,
                    url="http://wiki/en-1",
                )
            ]

            result = agent.execute(state)

        assert len(result.retrieved_docs) == 1
        assert result.retrieved_docs[0].source == "wiki"
        assert result.trace[0].action == "community_sources_unavailable"

    def test_execute_prioritizes_reddit_for_english_queries(self):
        reddit_docs = [
            make_doc(
                "Dodge late and punish the delayed slam.",
                source="reddit",
                chapter=2,
                url="http://reddit/en-1",
            )
        ]

        def availability(source, language=""):
            return source == "reddit" and language == "en"

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.has_indexed_source_documents", side_effect=availability
        ), patch("src.agents.community_agent.reddit_search", return_value=reddit_docs), patch(
            "src.agents.community_agent.nga_search", side_effect=AssertionError("should not search nga first")
        ), patch(
            "src.agents.community_agent.bilibili_search", side_effect=AssertionError("should not search bilibili first")
        ):
            agent = CommunityAgent(DUMMY_CONFIG)
            state = make_state("How do I beat Tiger Vanguard?", chapter=2)
            state.identified_entities = ["Tiger Vanguard"]

            result = agent.execute(state)

        assert any(doc.source == "reddit" for doc in result.retrieved_docs)
        assert any(event.action.startswith("reddit_search") for event in result.trace)

    def test_missing_query_arg_is_filled_from_state_for_query_rewrite(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.community_agent.has_indexed_source_documents", return_value=True
        ), patch(
            "src.agents.community_agent.QueryRewriter.rewrite",
            side_effect=lambda query, known_entities=None: [f"rewritten::{query}::{','.join(known_entities or [])}"],
        ):
            agent = CommunityAgent(DUMMY_CONFIG)
            decision_steps = iter([
                ("rewrite first", "query_rewrite", {"entities": ["Tiger Vanguard"]}),
                ("done", "FINISH", {}),
            ])
            agent._decide = lambda context, state: next(decision_steps)
            state = make_state("How do I dodge Tiger Vanguard's delayed slam?", chapter=2)
            state.identified_entities = ["Tiger Vanguard"]

            result = agent.execute(state)

        assert result.trace[0].action == "query_rewrite({'entities': ['Tiger Vanguard'], 'query': " + repr(state.user_query) + "})"
        assert "error" not in result.trace[0].observation.lower()
        assert agent._rewritten_queries == [f"rewritten::{state.user_query}::Tiger Vanguard"]


class TestSynthesisAgent:
    def test_extract_citations_builds_sentence_aware_excerpt(self):
        docs = [
            Document(
                text=(
                    "Location: As a required boss, you won't be able to proceed to the Chapter 2 End Boss without defeating him and obtaining one half of the Key Items needed to open various doors in Chapter 2's Yellow Wind Ridge. "
                    "You can find this Vanguard in the Fright Cliff Region by taking a right after heading through the Sandgate Village."
                ),
                source="wiki",
                url="http://wiki/stone-vanguard",
                chapter=2,
                entity="Stone Vanguard",
                metadata={"author": "ign"},
            )
        ]

        citations = _extract_citations(docs)

        assert citations[0].excerpt.startswith("Location:")
        assert not citations[0].excerpt.startswith(("d obtaining", "nd obtaining", "obtaining one"))
        assert citations[0].excerpt.endswith(("…", "."))

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

    def test_no_results_answer_uses_targeted_next_step_hint_when_entity_is_missing(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            state = make_state("这个在哪？")
            state.workflow = "navigation"
            state.evidence_gaps = ["missing_entity"]

            result = agent.execute(state)

        assert "地点名" in result.final_answer or "NPC" in result.final_answer or "item" in result.final_answer
        assert result.answer_confidence == 0.0
        assert result.trace[0].action == "synthesis_no_results"

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

    def test_execute_english_fact_lookup_uses_english_prompt_and_sections(self):
        docs = [
            make_doc(
                "Tiger Vanguard has multiple high-threat attacks, including Blood Tornado and a delayed slam.",
                source="wiki",
                chapter=2,
                url="http://wiki/en-1",
            )
        ]
        docs[0].entity = "Tiger Vanguard"

        class FakeLLM:
            def complete(self, messages, system=""):
                assert "## Direct Answer" in system
                assert "## 直接结论" not in system
                assert "Player profile" in messages[0]["content"]
                return (
                    "## Direct Answer\n"
                    "- Tiger Vanguard has multiple named high-threat attacks.\n\n"
                    "## Evidence\n"
                    "- The wiki entry explicitly lists Blood Tornado and a delayed slam."
                )

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FakeLLM()
            state = make_state("How many major attacks does Tiger Vanguard have?")
            state.workflow = "fact_lookup"
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert "## Direct Answer" in result.final_answer
        assert "## Evidence" in result.final_answer
        assert "## Sources" in result.final_answer
        assert "[original source](http://wiki/en-1)" in result.final_answer

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

    def test_execute_uses_secondary_provider_when_primary_generation_fails(self):
        docs = [
            make_doc(
                "Tiger Vanguard's delayed slam leaves a punish window after recovery.",
                source="wiki",
                chapter=2,
                url="http://wiki/en-1",
            )
        ]

        class PrimaryFailingLLM:
            provider = "groq"

            def complete(self, messages, system=""):
                raise RuntimeError("429 Client Error: Too Many Requests for url: https://api.groq.com/openai/v1/chat/completions")

        class SecondaryLLM:
            provider = "anthropic"

            def complete(self, messages, system=""):
                return "## Direct Answer\n- Dodge late and punish after the slam recovery."

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = PrimaryFailingLLM()
            agent._fallback_llm = SecondaryLLM()
            agent._fallback_llm_initialized = True
            state = make_state("How do I beat Tiger Vanguard?", chapter=2)
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert result.stop_reason == "answered_via_secondary_provider"
        assert "## Direct Answer" in result.final_answer
        assert "## Sources" in result.final_answer
        assert result.trace[0].action == "synthesis_secondary_llm"

    def test_execute_marks_rate_limit_fallback_stop_reason(self):
        docs = [
            make_doc(
                "Tiger Vanguard's delayed slam leaves a punish window after recovery.",
                source="wiki",
                chapter=2,
                url="http://wiki/en-1",
            )
        ]

        class FailingLLM:
            def complete(self, messages, system=""):
                raise RuntimeError("429 Client Error: Too Many Requests for url: https://api.groq.com/openai/v1/chat/completions")

        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            agent.llm = FailingLLM()
            state = make_state("How do I beat Tiger Vanguard?", chapter=2)
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert result.stop_reason == "generation_rate_limited_fallback"
        assert "429 Client Error" in result.final_answer
        assert result.answer_confidence < 0.7

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

    def test_execute_returns_english_no_results_message_for_english_query(self):
        with patch("src.llm.client.LLMClient.__init__", return_value=None):
            agent = SynthesisAgent(DUMMY_CONFIG)
            state = make_state("Where is Xu Dog?")
            state.workflow = "navigation"

            result = agent.execute(state)

        assert "## Location Answer" in result.final_answer
        assert "retrieved material" in result.final_answer
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

    def test_synthesis_context_uses_citation_numbering_for_duplicate_urls(self):
        docs = [
            Document(
                text="Tiger Vanguard delayed slam overview.",
                source="wiki",
                url="http://wiki/tiger",
                chapter=2,
                entity="Tiger Vanguard",
                metadata={"author": "ign"},
            ),
            Document(
                text="Tiger Vanguard recovery punish details.",
                source="wiki",
                url="http://wiki/tiger",
                chapter=2,
                entity="Tiger Vanguard",
                metadata={"author": "ign"},
            ),
            Document(
                text="Reddit players recommend a late dodge.",
                source="reddit",
                url="http://reddit/tiger",
                chapter=2,
                entity="Tiger Vanguard",
                metadata={"author": "reddit"},
            ),
        ]
        state = make_state("How do I dodge Tiger Vanguard's delayed slam?", chapter=2)
        state.retrieved_docs = docs
        state.citations = _extract_citations(docs)

        context = _build_synthesis_context(state, language="en")

        assert "[Source 1] wiki | http://wiki/tiger\nTiger Vanguard delayed slam overview." in context
        assert "[Source 1] wiki | http://wiki/tiger\nTiger Vanguard recovery punish details." in context
        assert "[Source 2] reddit | http://reddit/tiger\nReddit players recommend a late dodge." in context
        assert "[Source 3]" not in context


class TestAnalysisAgentToolBinding:
    def test_string_docs_reference_is_resolved_from_state(self):
        docs = [
            make_doc(
                "Wait for the delayed slam and punish after recovery.",
                source="reddit",
                chapter=2,
                url="http://reddit/en-1",
            )
        ]
        captured = {}

        def fake_count_source_consensus(bound_docs, topic):
            captured["docs"] = bound_docs
            captured["topic"] = topic
            return []

        with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
            "src.agents.analysis_agent.count_source_consensus",
            side_effect=fake_count_source_consensus,
        ):
            agent = AnalysisAgent(DUMMY_CONFIG)
            decision_steps = iter([
                ("count first", "consensus_count", {"docs": "retrieved_docs", "topic": "dodge Tiger Vanguard's delayed slam"}),
                ("done", "FINISH", {}),
            ])
            agent._decide = lambda context, state: next(decision_steps)
            state = make_state("How do I dodge Tiger Vanguard's delayed slam?", chapter=2)
            state.retrieved_docs = docs

            result = agent.execute(state)

        assert captured["docs"] == docs
        assert captured["topic"] == "dodge Tiger Vanguard's delayed slam"
        assert "error" not in result.trace[0].observation.lower()
