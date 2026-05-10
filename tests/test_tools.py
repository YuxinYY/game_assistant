"""
Unit tests for tools (pure functions — no LLM, no network).
"""

import pytest
from src.core.state import Document
from src.tools.consensus import count_source_consensus, detect_conflicts
from src.tools.profile_ops import merge_to_profile, validate_extraction
from src.tools.spoiler_filter import apply_spoiler_filter


def make_doc(text: str, source: str = "nga", url: str = "http://a", chapter: int = 1) -> Document:
    return Document(text=text, source=source, url=url, chapter=chapter)


class TestConsensus:
    def test_counts_distinct_sources(self):
        docs = [
            make_doc("侧闪是最好的", url="http://nga/1"),
            make_doc("向左侧闪", url="http://nga/2"),
            make_doc("推荐侧向闪避", source="bilibili", url="http://bili/1"),
        ]
        results = count_source_consensus(docs, topic="虎先锋")
        strategy = next((s for s in results if s["label"] == "侧向闪避"), None)
        assert strategy is not None
        assert strategy["source_count"] == 3  # 3 distinct URLs

    def test_counts_english_strategy_mentions(self):
        docs = [
            make_doc("Players recommend a side dodge for the delayed slam.", source="reddit", url="http://reddit/1"),
            make_doc("Dodge left and punish after recovery.", source="reddit", url="http://reddit/2"),
        ]

        results = count_source_consensus(docs, topic="Tiger Vanguard")

        strategy = next((s for s in results if s["label"] == "侧向闪避"), None)
        assert strategy is not None
        assert strategy["source_count"] == 2

    def test_detects_parry_conflict(self):
        docs = [
            make_doc("能棍反这招", url="http://nga/1"),
            make_doc("棍反不稳，弹不了", url="http://nga/2"),
        ]
        conflicts = detect_conflicts(docs)
        assert any(c["topic"] == "棍反" for c in conflicts)


class TestSpoilerFilter:
    def test_skips_filter_when_chapter_is_unset(self):
        docs = [make_doc("推荐用广智", chapter=3)]
        filtered = apply_spoiler_filter(docs, max_chapter=None)
        assert len(filtered) == 1

    def test_removes_chapter3_content_for_chapter1_player(self):
        docs = [
            make_doc("推荐用广智", chapter=3),
            make_doc("侧闪即可", chapter=1),
        ]
        filtered = apply_spoiler_filter(docs, max_chapter=1)
        assert len(filtered) == 1
        assert filtered[0].text == "侧闪即可"

    def test_keyword_based_filter(self):
        docs = [make_doc("观音禅院之后解锁广智", chapter=None)]
        filtered = apply_spoiler_filter(docs, max_chapter=1)
        assert len(filtered) == 0  # keyword scan catches it

    def test_allows_same_chapter(self):
        docs = [make_doc("虎先锋打法", chapter=1)]
        filtered = apply_spoiler_filter(docs, max_chapter=1)
        assert len(filtered) == 1


class TestProfileOps:
    def test_validate_drops_unknown_items(self):
        validated = validate_extraction(
            {
                "chapter": 9,
                "equipped_spirit": "未知精魄",
                "unlocked_spells": ["定身术", "空气法术"],
            },
            {
                "all_spells": ["定身术"],
                "all_spirits": ["广智"],
                "all_armors": [],
                "all_skills_tree": [],
            },
        )
        assert "chapter" not in validated
        assert "equipped_spirit" not in validated
        assert validated["unlocked_spells"] == ["定身术"]

    def test_merge_unions_unlock_lists(self):
        from src.core.state import PlayerProfile

        profile = PlayerProfile(unlocked_skills=["闪身"])
        merged_profile, updates = merge_to_profile(
            {"unlocked_skills": ["闪身", "识破"]},
            profile,
        )
        assert merged_profile.unlocked_skills == ["闪身", "识破"]
        assert updates[0]["field"] == "unlocked_skills"
