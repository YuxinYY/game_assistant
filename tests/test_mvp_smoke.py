"""
Minimal smoke test for the text-query MVP pipeline.
The test avoids live network calls by stubbing retrieval and LLM completion.
"""

from unittest.mock import patch

from src.agents.profile_agent import ProfileAgent
from src.core.orchestrator import Orchestrator
from src.core.state import Document, PlayerProfile


DUMMY_CONFIG = {
    "llm": {"model": "claude-sonnet-4-7", "temperature": 0.3, "max_tokens": 512},
    "agents": {"max_react_iterations": 3},
    "spoiler": {"enable": True},
}


def _fake_complete(self, messages, system=""):
    if "意图分类器" in system or "intent classifier" in system.lower():
        return "boss_strategy"
    return (
        "## 招式识别\n"
        "- 虎跃斩（来源: wiki http://wiki/1）\n\n"
        "## 针对你的 build 的建议\n"
        "- 第三段优先侧闪，时机看前爪落地（来源: nga http://nga/1）"
    )


def _make_doc(text: str, source: str, url: str, entity: str = "虎先锋") -> Document:
    return Document(
        text=text,
        source=source,
        url=url,
        chapter=1,
        entity=entity,
        metadata={"author": "sample"},
    )


def test_text_query_pipeline_returns_answer_and_citations():
    wiki_docs = [
        _make_doc(
            "虎跃斩第三段蓄力时间约1秒，落点面积较大。",
            source="wiki",
            url="http://wiki/1",
        )
    ]
    nga_docs = [
        _make_doc(
            "虎跃斩第三段建议向左侧闪，时机是前爪落地。",
            source="nga",
            url="http://nga/1",
        )
    ]

    with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
        "src.llm.client.LLMClient.complete", new=_fake_complete
    ), patch("src.agents.wiki_agent.wiki_search", return_value=wiki_docs), patch(
        "src.agents.community_agent.QueryRewriter.rewrite", return_value=["虎先锋 怎么打"]
    ), patch("src.agents.community_agent.nga_search", return_value=nga_docs), patch(
        "src.agents.community_agent.bilibili_search", return_value=[]
    ), patch("src.agents.community_agent.reddit_search", return_value=[]):
        orchestrator = Orchestrator(DUMMY_CONFIG)
        state = orchestrator.run("虎先锋那个招怎么躲？", PlayerProfile(chapter=1))

    assert state.workflow == "boss_strategy"
    assert len(state.retrieved_docs) == 2
    assert len(state.citations) == 2
    assert state.final_answer
    assert "TODO" not in state.final_answer
    assert "虎跃斩" in state.final_answer


def test_screenshot_only_flow_updates_profile_without_routing():
    def fake_profile_execute(self, state):
        state.player_profile.chapter = 2
        state.profile_updates = [
            {
                "field": "chapter",
                "old_value": 1,
                "new_value": 2,
                "source": "screenshot:combat_hud",
                "confidence": 0.9,
            }
        ]
        return state

    with patch("src.llm.client.LLMClient.__init__", return_value=None), patch.object(
        ProfileAgent,
        "execute",
        new=fake_profile_execute,
    ):
        orchestrator = Orchestrator(DUMMY_CONFIG)
        state = orchestrator.run("", PlayerProfile(chapter=1), screenshots=[b"img"])

    assert state.workflow == "profile_update"
    assert state.player_profile.chapter == 2
    assert state.profile_updates[0]["field"] == "chapter"


def _fake_complete_en(self, messages, system=""):
    if "意图分类器" in system or "intent classifier" in system.lower():
        return "boss_strategy"
    return (
        "## Move Or Mechanic\n"
        "- Tiger Vanguard's delayed slam is one of the key punish windows.\n\n"
        "## Recommendation For Your Build\n"
        "- Dodge late and punish after the slam recovery (source: wiki http://wiki/en-1)."
    )


def test_english_text_query_pipeline_returns_english_answer_and_citations():
    wiki_docs = [
        _make_doc(
            "Tiger Vanguard's delayed slam leaves a punish window after the recovery.",
            source="wiki",
            url="http://wiki/en-1",
            entity="Tiger Vanguard",
        )
    ]
    reddit_docs = [
        _make_doc(
            "Reddit players recommend waiting out the delayed slam before punishing.",
            source="reddit",
            url="http://reddit/en-1",
            entity="Tiger Vanguard",
        )
    ]

    with patch("src.llm.client.LLMClient.__init__", return_value=None), patch(
        "src.llm.client.LLMClient.complete", new=_fake_complete_en
    ), patch("src.agents.wiki_agent.wiki_search", return_value=wiki_docs), patch(
        "src.agents.community_agent.QueryRewriter.rewrite", return_value=["how to beat Tiger Vanguard"]
    ), patch(
        "src.agents.community_agent.has_indexed_source_documents",
        side_effect=lambda source, language="": source == "reddit" and language == "en",
    ), patch("src.agents.community_agent.nga_search", return_value=[]), patch(
        "src.agents.community_agent.bilibili_search", return_value=[]
    ), patch("src.agents.community_agent.reddit_search", return_value=reddit_docs):
        orchestrator = Orchestrator(DUMMY_CONFIG)
        state = orchestrator.run("How do I beat Tiger Vanguard?", PlayerProfile(chapter=2))

    assert state.workflow == "boss_strategy"
    assert len(state.retrieved_docs) == 2
    assert len(state.citations) == 2
    assert "## Recommendation For Your Build" in state.final_answer
    assert "## Sources" in state.final_answer
    assert "Tiger Vanguard" in state.final_answer
