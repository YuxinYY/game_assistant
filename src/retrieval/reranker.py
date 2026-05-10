"""
Reranker: second-pass scoring of retrieved candidates.
Uses LLM-as-reranker (cheaper than cross-encoder, good enough for this corpus size).
"""

from __future__ import annotations

import re

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
        """
        llm_score = self._llm_score(query, doc)
        if llm_score is not None:
            return llm_score
        return self._lexical_score(query, doc)

    def _llm_score(self, query: str, doc: Document) -> float | None:
        if self.llm is None or not hasattr(self.llm, "complete"):
            return None
        prompt = (
            f"用户问题: {query}\n\n"
            f"候选片段来源: {doc.source}\n"
            f"实体: {doc.entity or '未知'}\n"
            f"片段内容: {doc.text[:800]}\n\n"
            "请给出 0 到 10 的相关性分数。"
            "只返回一个数字，不要解释。"
        )
        try:
            response = self.llm.complete(
                [{"role": "user", "content": prompt}],
                system="你是检索重排打分器，只返回数字分数。",
            )
        except Exception:
            return None

        match = re.search(r"(10(?:\.0+)?)|([0-9](?:\.\d+)?)", response)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _lexical_score(self, query: str, doc: Document) -> float:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        entity = getattr(doc, "entity", None) or ""
        metadata = getattr(doc, "metadata", {}) or {}

        searchable_text = " ".join(
            part
            for part in [
                doc.text,
                entity,
                str(metadata.get("title", "")),
            ]
            if part
        ).lower()

        overlap = sum(1 for token in query_tokens if token in searchable_text)
        score = overlap / len(query_tokens)

        entity = entity.lower()
        if entity and entity in query.lower():
            score += 0.75
        title = str(metadata.get("title", "")).lower()
        if title and title in query.lower():
            score += 0.5
        if doc.source == "wiki":
            score += 0.1
        return score


def _tokenize(text: str) -> list[str]:
    tokens = set(match.lower() for match in re.findall(r"[A-Za-z0-9]+", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        if not chunk:
            continue
        tokens.add(chunk)
        if len(chunk) > 1:
            for index in range(len(chunk) - 1):
                tokens.add(chunk[index : index + 2])
    return [token for token in tokens if token]
