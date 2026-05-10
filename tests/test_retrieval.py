"""
Targeted tests for retrieval degradation and BM25 document mapping.
"""

import pytest

from src.core.state import Document
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    _build_chroma_where,
    _is_hnsw_load_error,
)
from src.retrieval.query_rewriter import QueryRewriter


class FakeBM25:
    def __init__(self, scores):
        self._scores = scores

    def get_scores(self, tokens):
        return self._scores


DUMMY_CONFIG = {
    "retrieval": {
        "chroma_persist_dir": "data/indexes/chroma_db",
        "chroma_collection": "wukong_chunks",
        "bm25_index_path": "data/indexes/bm25_index.pkl",
    }
}


def test_sparse_search_gracefully_disables_legacy_bm25_without_documents():
    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = FakeBM25([1.0, 0.5])
    retriever._bm25_documents = []

    results = retriever._sparse_search("虎先锋", top_k=3, filters=None)

    assert results == []


def test_sparse_search_returns_ranked_and_filtered_documents():
    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = FakeBM25([0.1, 1.2, 0.8])
    retriever._bm25_documents = [
        {
            "text": "无关 wiki 文档",
            "source": "wiki",
            "url": "http://wiki/1",
            "chapter": 1,
            "entity": "虎先锋",
            "metadata": {},
        },
        {
            "text": "高分但章节过高",
            "source": "nga",
            "url": "http://nga/1",
            "chapter": 4,
            "entity": "虎先锋",
            "metadata": {"author": "a"},
        },
        {
            "text": "可用的低章节 NGA 文档",
            "source": "nga",
            "url": "http://nga/2",
            "chapter": 2,
            "entity": "虎先锋",
            "metadata": {"author": "b"},
        },
    ]

    results = retriever._sparse_search(
        "虎先锋 怎么打",
        top_k=2,
        filters={"source": "nga", "chapter__lte": 2},
    )

    assert len(results) == 1
    assert results[0].url == "http://nga/2"


def test_sparse_search_tokenizes_natural_chinese_query_without_spaces():
    captured = {}

    class CapturingBM25:
        def get_scores(self, tokens):
            captured["tokens"] = tokens
            return [1.0]

    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = CapturingBM25()
    retriever._bm25_documents = [
        {
            "text": "虎先锋 蓄力 招式 躲避",
            "source": "wiki",
            "url": "http://wiki/tiger-charge",
            "chapter": 2,
            "entity": "虎先锋",
            "metadata": {},
        }
    ]

    results = retriever._sparse_search(
        "虎先锋那个蓄力的招怎么躲？",
        top_k=1,
        filters={"source": "wiki"},
    )

    assert captured["tokens"] == ["虎先锋", "蓄力"]
    assert len(results) == 1
    assert results[0].url == "http://wiki/tiger-charge"


def test_sparse_search_tokenizes_sticky_chinese_terms():
    captured = {}

    class CapturingBM25:
        def get_scores(self, tokens):
            captured["tokens"] = tokens
            return [1.0]

    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = CapturingBM25()
    retriever._bm25_documents = [
        {
            "text": "虎先锋 蓄力 招式 躲避",
            "source": "wiki",
            "url": "http://wiki/tiger-charge",
            "chapter": 2,
            "entity": "虎先锋",
            "metadata": {},
        }
    ]

    retriever._sparse_search(
        "虎先锋蓄力招式",
        top_k=1,
        filters={"source": "wiki"},
    )

    assert captured["tokens"] == ["虎先锋", "蓄力"]


def test_sparse_search_tokenizes_count_questions_with_entity_names():
    captured = {}

    class CapturingBM25:
        def get_scores(self, tokens):
            captured["tokens"] = tokens
            return [1.0]

    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = CapturingBM25()
    retriever._bm25_documents = [
        {
            "text": "虎先锋 招式 乌鸦坐飞机 血龙卷 虎突猛进",
            "source": "wiki",
            "url": "http://wiki/tiger-ultimate",
            "chapter": 2,
            "entity": "虎先锋",
            "metadata": {},
        }
    ]

    results = retriever._sparse_search(
        "虎先锋有几个大招",
        top_k=1,
        filters={"source": "wiki", "entity": "虎先锋"},
    )

    assert captured["tokens"] == ["虎先锋", "大招"]
    assert len(results) == 1
    assert results[0].url == "http://wiki/tiger-ultimate"


def test_search_falls_back_to_dense_results_when_sparse_is_unavailable(monkeypatch):
    retriever = HybridRetriever(DUMMY_CONFIG)

    monkeypatch.setattr(
        retriever,
        "_dense_search",
        lambda query, top_k, filters: [
            type("Doc", (), {"url": "http://wiki/1", "text": "dense", "source": "wiki"})()
        ],
    )
    monkeypatch.setattr(retriever, "_sparse_search", lambda query, top_k, filters: [])

    results = retriever.search("虎先锋", top_k=1, filters={"source": "wiki"})

    assert len(results) == 1
    assert results[0].url == "http://wiki/1"


def test_search_falls_back_to_sparse_results_when_dense_errors(monkeypatch):
    retriever = HybridRetriever(DUMMY_CONFIG)
    sparse_doc = Document(
        text="虎先锋蓄力招可侧闪。",
        source="nga",
        url="http://nga/charge",
    )

    def raise_dense_error(query, top_k, filters):
        raise RuntimeError("Error loading hnsw index")

    monkeypatch.setattr(retriever, "_dense_search", raise_dense_error)
    monkeypatch.setattr(retriever, "_sparse_search", lambda query, top_k, filters: [sparse_doc])

    results = retriever.search("虎先锋那个蓄力的招怎么躲？", top_k=1, filters={"source": "nga"})

    assert len(results) == 1
    assert results[0].url == "http://nga/charge"


def test_dense_search_rebuilds_and_retries_after_hnsw_load_error(monkeypatch):
    retriever = HybridRetriever(DUMMY_CONFIG)

    class BrokenCollection:
        def query(self, query_texts, n_results, where=None):
            raise RuntimeError("Error constructing hnsw segment reader: Error loading hnsw index")

    class HealthyCollection:
        def query(self, query_texts, n_results, where=None):
            return {
                "documents": [["Tiger Vanguard guide with punish windows."]],
                "metadatas": [[{
                    "source": "wiki",
                    "url": "http://wiki/tiger",
                    "entity": "Tiger Vanguard",
                }]],
            }

    retriever._chroma = BrokenCollection()
    recovery_calls = []

    def fake_recover(exc):
        recovery_calls.append(str(exc))
        retriever._chroma = HealthyCollection()
        return True

    monkeypatch.setattr(retriever, "_recover_chroma_index", fake_recover)

    results = retriever._dense_search("Tiger Vanguard", top_k=1, filters={"source": "wiki"})

    assert len(results) == 1
    assert results[0].url == "http://wiki/tiger"
    assert recovery_calls


def test_dense_search_disables_dense_after_failed_hnsw_recovery(monkeypatch):
    retriever = HybridRetriever(DUMMY_CONFIG)
    query_calls = {"count": 0}

    class BrokenCollection:
        def query(self, query_texts, n_results, where=None):
            query_calls["count"] += 1
            raise RuntimeError("Error loading hnsw index")

    retriever._chroma = BrokenCollection()
    monkeypatch.setattr(retriever, "_recover_chroma_index", lambda exc: False)

    with pytest.raises(RuntimeError, match="Error loading hnsw index"):
        retriever._dense_search("Tiger Vanguard", top_k=1, filters=None)

    assert retriever._dense_available is False
    assert retriever._dense_search("Tiger Vanguard", top_k=1, filters=None) == []
    assert query_calls["count"] == 1


def test_is_hnsw_load_error_matches_compactor_backfill_message():
    error = RuntimeError(
        "Error sending backfill request to compactor: Error constructing hnsw segment reader: Error loading hnsw index"
    )

    assert _is_hnsw_load_error(error) is True


def test_build_chroma_where_wraps_multiple_filters_in_and_clause():
    where = _build_chroma_where({"source": "nga", "chapter__lte": 2})

    assert where == {
        "$and": [
            {"source": {"$eq": "nga"}},
            {"chapter": {"$lte": 2}},
        ]
    }


def test_build_chroma_where_maps_unknown_filters_to_metadata_fields():
    where = _build_chroma_where({"source": "wiki", "language": "en"})

    assert where == {
        "$and": [
            {"source": {"$eq": "wiki"}},
            {"meta_language": {"$eq": "en"}},
        ]
    }


def test_sparse_search_can_filter_on_metadata_fields():
    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = FakeBM25([1.1, 0.9])
    retriever._bm25_documents = [
        {
            "text": "Tiger Vanguard guide in English.",
            "source": "wiki",
            "url": "http://wiki/en",
            "chapter": 2,
            "entity": "Tiger Vanguard",
            "metadata": {"language": "en", "author": "ign"},
        },
        {
            "text": "虎先锋中文攻略。",
            "source": "wiki",
            "url": "http://wiki/zh",
            "chapter": 2,
            "entity": "虎先锋",
            "metadata": {"language": "zh", "author": "bwiki"},
        },
    ]

    results = retriever._sparse_search(
        "Tiger Vanguard guide",
        top_k=2,
        filters={"source": "wiki", "language": "en"},
    )

    assert len(results) == 1
    assert results[0].url == "http://wiki/en"


def test_query_rewriter_parses_llm_generated_queries():
    class FakeLLM:
        def complete(self, messages, system=""):
            return '{"queries": ["Tiger Vanguard guide", "how to beat Tiger Vanguard"]}'

    rewriter = QueryRewriter(FakeLLM())

    queries = rewriter.rewrite("How do I beat Tiger Vanguard?", known_entities=["Tiger Vanguard"])

    assert "Tiger Vanguard guide" in queries
    assert "how to beat Tiger Vanguard" in queries


def test_query_rewriter_salvages_incomplete_json_lines():
    class FakeLLM:
        def complete(self, messages, system=""):
            return '\n'.join([
                '{"queries": [',
                '"虎先锋蓄力招怎么躲",',
                '"虎先锋蓄力技巧",',
                '"如何躲避虎先锋蓄力"',
            ])

    rewriter = QueryRewriter(FakeLLM())

    queries = rewriter.rewrite("虎先锋那个蓄力的招怎么躲？")

    assert "虎先锋蓄力招怎么躲" in queries
    assert "虎先锋蓄力技巧" in queries
    assert '{"queries": [' not in queries


def test_search_applies_reranking_to_fused_candidates(monkeypatch):
    retriever = HybridRetriever(DUMMY_CONFIG)
    retriever._bm25_loaded = True
    retriever._bm25 = FakeBM25([])
    retriever._bm25_documents = []

    off_topic = Document(
        text="General movement tips with no Tiger Vanguard detail.",
        source="wiki",
        url="http://wiki/off-topic",
        entity="广智",
        metadata={"title": "General Tips"},
    )
    on_topic = Document(
        text="Tiger Vanguard guide with dodge timing and spinning kick punish.",
        source="wiki",
        url="http://wiki/tiger",
        entity="Tiger Vanguard",
        metadata={"title": "Tiger Vanguard"},
    )

    monkeypatch.setattr(retriever, "_dense_search", lambda query, top_k, filters: [off_topic, on_topic])
    monkeypatch.setattr(retriever, "_sparse_search", lambda query, top_k, filters: [])

    results = retriever.search("Tiger Vanguard guide", top_k=1, filters={"source": "wiki"})

    assert len(results) == 1
    assert results[0].url == "http://wiki/tiger"