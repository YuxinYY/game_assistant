from unittest.mock import patch

from src.tools.search import wiki_search


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return []


def test_wiki_search_filters_pure_english_queries_to_english_docs():
    retriever = FakeRetriever()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("How do I beat Tiger Vanguard?", top_k=3)

    assert retriever.calls == [
        (
            "How do I beat Tiger Vanguard?",
            3,
            {"source": "wiki", "language": "en"},
        )
    ]


def test_wiki_search_filters_pure_chinese_queries_to_chinese_docs():
    retriever = FakeRetriever()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("虎先锋怎么打？")

    assert retriever.calls[0][2] == {"source": "wiki", "language": "zh"}


def test_wiki_search_keeps_mixed_language_queries_unfiltered():
    retriever = FakeRetriever()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("Tiger Vanguard 怎么打？")

    assert retriever.calls[0][2] == {"source": "wiki"}