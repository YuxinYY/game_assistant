"""
Reranker: second-pass scoring of retrieved candidates.
Uses LLM-as-reranker (cheaper than cross-encoder, good enough for this corpus size).
"""

from src.core.state import Document


class LLMReranker:
    def __init__(self, llm_client):
        self.llm = llm_client

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """Score each doc for relevance to query, return top_k sorted by score."""
        if not docs:
            return []
        scored = [(doc, self._score(query, doc)) for doc in docs]
        scored.sort(key=lambda x: -x[1])
        return [doc for doc, _ in scored[:top_k]]

    def _score(self, query: str, doc: Document) -> float:
        """
        Ask LLM: "On a scale 1-10, how relevant is this passage to the query?"
        TODO: batch scoring for efficiency.
        """
        raise NotImplementedError
