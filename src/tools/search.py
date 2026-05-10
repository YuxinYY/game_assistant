"""
Search tools: thin wrappers over HybridRetriever that expose a per-source interface.
Each returns a list[Document] so agents get structured, not raw-string, results.
"""

import re
from typing import Optional
from src.core.state import Document


PURE_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
PURE_LATIN_PATTERN = re.compile(r"[A-Za-z]")


def wiki_search(query: str, entity_filter: str = "", top_k: int = 5) -> list[Document]:
    """Search the wiki corpus. Returns official game info only."""
    from src.retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()
    filters = {"source": "wiki"}
    language = _infer_wiki_language_filter(query)
    if language:
        filters["language"] = language
    if entity_filter:
        filters["entity"] = entity_filter
    return retriever.search(query, top_k=top_k, filters=filters)


def entity_lookup(entity: str) -> dict:
    """Exact-match lookup for a boss or move name in the wiki index."""
    from src.retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()
    results = retriever.search(entity, top_k=1, filters={"source": "wiki", "entity": entity})
    if not results:
        return {}
    doc = results[0]
    return {"entity": entity, "text": doc.text, "url": doc.url}


def nga_search(
    query: str, top_k: int = 8, chapter_filter: Optional[int] = None
) -> list[Document]:
    """Search NGA forum corpus, optionally restricting to chapter <= chapter_filter."""
    from src.retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()
    filters: dict = {"source": "nga"}
    if chapter_filter is not None:
        filters["chapter__lte"] = chapter_filter
    return retriever.search(query, top_k=top_k, filters=filters)


def bilibili_search(query: str, top_k: int = 5) -> list[Document]:
    from src.retrieval.hybrid_retriever import get_retriever
    return get_retriever().search(query, top_k=top_k, filters={"source": "bilibili"})


def reddit_search(query: str, top_k: int = 5) -> list[Document]:
    from src.retrieval.hybrid_retriever import get_retriever
    return get_retriever().search(query, top_k=top_k, filters={"source": "reddit"})


def _infer_wiki_language_filter(query: str) -> str:
    has_cjk = bool(PURE_CJK_PATTERN.search(query))
    has_latin = bool(PURE_LATIN_PATTERN.search(query))
    if has_latin and not has_cjk:
        return "en"
    if has_cjk and not has_latin:
        return "zh"
    return ""
