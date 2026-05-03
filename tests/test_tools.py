"""
Unit tests for tools (pure functions — no LLM, no network).
"""

import pytest
from src.core.state import Document
from src.tools.consensus import count_source_consensus, detect_conflicts
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

    def test_detects_parry_conflict(self):
        docs = [
            make_doc("能棍反这招", url="http://nga/1"),
            make_doc("棍反不稳，弹不了", url="http://nga/2"),
        ]
        conflicts = detect_conflicts(docs)
        assert any(c["topic"] == "棍反" for c in conflicts)


class TestSpoilerFilter:
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
