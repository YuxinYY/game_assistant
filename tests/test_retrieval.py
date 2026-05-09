"""
Targeted tests for retrieval degradation and BM25 document mapping.
"""

from src.retrieval.hybrid_retriever import HybridRetriever, _build_chroma_where


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


def test_build_chroma_where_wraps_multiple_filters_in_and_clause():
    where = _build_chroma_where({"source": "nga", "chapter__lte": 2})

    assert where == {
        "$and": [
            {"source": {"$eq": "nga"}},
            {"chapter": {"$lte": 2}},
        ]
    }