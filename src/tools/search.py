"""
Search tools: thin wrappers over HybridRetriever that expose a per-source interface.
Each returns a list[Document] so agents get structured, not raw-string, results.
"""

from functools import lru_cache
import re
from typing import Optional
from src.core.state import Document


PURE_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
PURE_LATIN_PATTERN = re.compile(r"[A-Za-z]")
MOVE_ENTRY_PATTERN = re.compile(r"招式\s+([^：:\s]{2,40})：(.+?)(?=招式\s+[^：:\s]{2,40}：|$)", re.S)

SUSPICIOUS_MOVE_NAME_KEYWORDS = (
    "7777",
    "坐飞机",
    "横跳",
    "耗油",
    "火星",
    "咸鱼",
    "接，化，发",
    "偷袭",
)

SUSPICIOUS_MOVE_DESC_KEYWORDS = (
    "马保国",
    "阿福",
    "街霸",
    "能源城",
    "蓝毒兽",
    "超音速",
    "军体拳",
)


def wiki_search(query: str, entity_filter: str = "", top_k: int = 5) -> list[Document]:
    """Search the wiki corpus. Returns official game info only."""
    from src.retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()
    filters = {"source": "wiki"}
    language = _infer_wiki_language_filter(query)
    if language:
        filters["language"] = language
    if not entity_filter:
        entity_filter = infer_wiki_entity(query)
    if entity_filter:
        filters["entity"] = entity_filter
    return retriever.search(query, top_k=top_k, filters=filters)


def infer_wiki_entity(query: str) -> str:
    """Infer a likely wiki entity directly from the user query when an exact name is present."""
    matches = _match_wiki_entities(query)
    if not matches:
        return ""
    if len(matches) == 1:
        return matches[0]

    first = matches[0]
    second = matches[1]
    if len(_normalize_entity_text(first)) > len(_normalize_entity_text(second)):
        return first
    return ""


def entity_lookup(entity: str, query: str = "", top_k: int = 3) -> dict:
    """Exact-match lookup for a boss or move name in the wiki index."""
    from src.retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()
    lookup_query = query or entity
    language = _infer_wiki_language_filter(lookup_query) or _infer_wiki_language_filter(entity)
    direct_match = _direct_entity_lookup(retriever, entity, lookup_query, language, top_k=top_k)
    if direct_match:
        return direct_match
    results = retriever.search(lookup_query, top_k=top_k, filters={"source": "wiki", "entity": entity})
    if not results:
        return {}
    merged_text = _build_entity_summary(entity, [
        {
            "text": doc.text,
            "url": doc.url,
            "entity": doc.entity,
            "metadata": doc.metadata,
        }
        for doc in results
    ], lookup_query)
    return {"entity": entity, "text": merged_text, "url": results[0].url}


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


@lru_cache(maxsize=1)
def _get_wiki_entities() -> tuple[str, ...]:
    from src.retrieval.hybrid_retriever import get_retriever

    entities: list[str] = []
    seen: set[str] = set()
    for chunk in getattr(get_retriever(), "bm25_documents", []):
        if chunk.get("source") != "wiki":
            continue
        entity = str(chunk.get("entity") or "").strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
    entities.sort(key=lambda item: len(_normalize_entity_text(item)), reverse=True)
    return tuple(entities)


def _match_wiki_entities(query: str) -> list[str]:
    normalized_query = _normalize_entity_text(query)
    if not normalized_query:
        return []
    return [entity for entity in _get_wiki_entities() if _normalize_entity_text(entity) in normalized_query]


def _normalize_entity_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text).lower()


def _direct_entity_lookup(retriever, entity: str, query: str, language: str, top_k: int = 3) -> dict:
    candidates = _matching_entity_chunks(retriever, entity, language)
    if not candidates and language:
        candidates = _matching_entity_chunks(retriever, entity, "")
    if not candidates:
        return {}

    ranked = _rank_entity_chunks(candidates, entity, query)
    selected = _select_entity_chunks(ranked, query, top_k=top_k)
    chunk = selected[0]
    return {
        "entity": entity,
        "text": _build_entity_summary(entity, selected, query),
        "url": chunk.get("url", ""),
    }


def _matching_entity_chunks(retriever, entity: str, language: str) -> list[dict]:
    matches = []
    for chunk in getattr(retriever, "bm25_documents", []):
        if chunk.get("source") != "wiki":
            continue
        if chunk.get("entity") != entity:
            continue
        metadata = chunk.get("metadata", {})
        if language and metadata.get("language") != language:
            continue
        matches.append(chunk)
    return matches


def _rank_entity_chunks(chunks: list[dict], entity: str, query: str) -> list[dict]:
    scored = [(chunk, _entity_chunk_score(chunk, entity, query)) for chunk in chunks]
    scored.sort(key=lambda item: -item[1])
    return [chunk for chunk, _ in scored]


def _entity_chunk_score(chunk: dict, entity: str, query: str) -> float:
    searchable = " ".join(
        part
        for part in [
            chunk.get("text", ""),
            str(chunk.get("entity", "")),
            str(chunk.get("metadata", {}).get("title", "")),
        ]
        if part
    ).lower()
    tokens = _query_tokens(query or entity)
    score = 0.0
    for token in tokens:
        if token.lower() in searchable:
            score += max(len(token), 1)

    if _is_count_query(query):
        score += len(_extract_move_names([chunk])) * 3
        score += searchable.count("招式")

    if entity and entity.lower() in searchable:
        score += 2
    return score


def _select_entity_chunks(chunks: list[dict], query: str, top_k: int = 3) -> list[dict]:
    if not chunks:
        return []
    if _is_count_query(query):
        return chunks[: max(top_k, 4)]
    if _is_move_query(query):
        return chunks[: max(2, min(top_k, len(chunks)))]
    return chunks[:1]


def _build_entity_summary(entity: str, chunks: list[dict], query: str) -> str:
    move_entries = _extract_move_entries(chunks)
    reliable_moves = [entry for entry in move_entries if not _is_suspicious_move_entry(entry)]
    suspicious_moves = [entry for entry in move_entries if _is_suspicious_move_entry(entry)]
    header_lines = [f"实体：{entity}"]
    if move_entries:
        if _is_count_query(query):
            return _build_count_query_summary(entity, move_entries, reliable_moves, suspicious_moves)

        if _is_move_listing_query(query):
            return _build_move_listing_summary(entity, reliable_moves, suspicious_moves)

        related_names = [entry["name"] for entry in reliable_moves[:5]]
        if related_names:
            header_lines.append(f"相关招式条目：{'、'.join(related_names)}")
        if suspicious_moves:
            header_lines.append(
                "命名风险：该页面包含社区戏称或玩梗命名，不能直接当作官方招式名。"
            )
    body = "\n\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))
    return "\n".join(header_lines) + ("\n\n" + body if body else "")


def _build_count_query_summary(
    entity: str,
    move_entries: list[dict],
    reliable_moves: list[dict],
    suspicious_moves: list[dict],
) -> str:
    header_lines = [f"实体：{entity}"]
    header_lines.append(f"当前 wiki 页面列出的招式条目数：{len(move_entries)}")

    if reliable_moves:
        header_lines.append(
            "较稳定的高威胁条目：" + "、".join(entry["name"] for entry in reliable_moves[:5])
        )

    if suspicious_moves:
        header_lines.append(
            "命名质量警告：页面里有明显社区戏称或玩梗命名，例如："
            + "、".join(entry["name"] for entry in suspicious_moves[:5])
        )
        header_lines.append(
            "说明：这更适合作为“该页面列了多少条招式”的统计，不宜直接当成严格官方大招数量。"
        )
    else:
        header_lines.append("说明：这是按页面招式条目整理出的数量。")

    return "\n".join(header_lines)


def _build_move_listing_summary(
    entity: str,
    reliable_moves: list[dict],
    suspicious_moves: list[dict],
) -> str:
    header_lines = [f"实体：{entity}"]

    if reliable_moves:
        header_lines.append(
            "较稳定的招式条目：" + "、".join(entry["name"] for entry in reliable_moves[:8])
        )
    else:
        header_lines.append("当前页面里没有足够稳定的招式命名，无法直接给出干净的招式列表。")

    if suspicious_moves:
        header_lines.append(
            "已降权的可疑条目：" + "、".join(entry["name"] for entry in suspicious_moves[:5])
        )
        header_lines.append(
            "说明：该页面包含社区戏称、玩梗命名和台词化描述，已避免把正文台词短句当作招式名。"
        )

    return "\n".join(header_lines)


def _extract_move_entries(chunks: list[dict]) -> list[dict]:
    entries: list[dict] = []
    seen_names: set[str] = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        for name, description in MOVE_ENTRY_PATTERN.findall(text):
            normalized_name = name.strip()
            if not normalized_name or normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            entries.append(
                {
                    "name": normalized_name,
                    "description": " ".join(description.split()),
                }
            )
    return entries


def _extract_move_names(chunks: list[dict]) -> list[str]:
    return [entry["name"] for entry in _extract_move_entries(chunks)]


def _is_suspicious_move_entry(entry: dict) -> bool:
    name = str(entry.get("name", ""))
    description = str(entry.get("description", ""))
    if not name:
        return False
    if re.search(r"\d", name):
        return True
    if "未收录" in name:
        return True
    if any(keyword in name for keyword in SUSPICIOUS_MOVE_NAME_KEYWORDS):
        return True
    if any(keyword in description for keyword in SUSPICIOUS_MOVE_DESC_KEYWORDS):
        return True
    return False


def _query_tokens(query: str) -> list[str]:
    if not query:
        return []
    from src.retrieval.hybrid_retriever import _tokenize_query

    tokens = _tokenize_query(query)
    return list(dict.fromkeys([token for token in tokens if token]))


def _is_count_query(query: str) -> bool:
    count_keywords = ("几个", "多少", "哪些", "有哪些", "哪几个", "有哪几", "几种")
    return any(keyword in query for keyword in count_keywords)


def _is_move_listing_query(query: str) -> bool:
    listing_keywords = (
        "都叫什么",
        "叫什么",
        "什么招式",
        "哪些招式",
        "招式有哪些",
        "大招有哪些",
        "有哪些大招",
        "有什么大招",
        "招式名字",
    )
    return any(keyword in query for keyword in listing_keywords)


def _is_move_query(query: str) -> bool:
    return any(keyword in query for keyword in ("怎么躲", "怎么打", "躲", "闪", "招", "蓄力"))
