"""
Hybrid retriever: fuses dense (ChromaDB) + sparse (BM25) results via Reciprocal Rank Fusion.
Singleton pattern so the index loads once per process.
"""

import gc
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional
from src.core.state import Document
from src.retrieval.index_builder import (
    DEFAULT_CHUNKS_PATH,
    rebuild_chroma_from_chunks,
    resolve_chroma_dir,
)
from src.retrieval.reranker import LLMReranker

_retriever_instance = None
DOCUMENT_FILTER_FIELDS = {"text", "source", "url", "chapter", "entity", "credibility", "post_date"}
LOGGER = logging.getLogger(__name__)
CHINESE_QUERY_STOPWORDS = (
    "那个",
    "这个",
    "如何",
    "怎么",
    "什么",
    "一下",
    "一下子",
    "打法",
    "招式",
    "有几个",
    "几个",
    "有哪些",
    "都有哪些",
    "有什么",
    "都有什么",
    "这一招",
    "这招",
    "的招",
    "可以",
    "需要",
    "请问",
    "吗",
    "呢",
    "呀",
    "啊",
    "吧",
    "了",
    "的",
    "躲避",
    "怎么躲",
    "怎么打",
    "躲",
    "打",
)
CHINESE_QUERY_KEYWORDS = (
    "大招",
    "蓄力",
    "闪避",
    "躲避",
    "连招",
    "技能",
    "法术",
    "变身",
    "攻略",
)
HNSW_ERROR_MARKERS = (
    "error loading hnsw index",
    "constructing hnsw segment reader",
    "creating hnsw segment reader",
    "backfill request to compactor",
)


def get_retriever(config: dict | None = None) -> "HybridRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        if config is None:
            from src.core.orchestrator import load_config

            config = load_config()
        _retriever_instance = HybridRetriever(config)
    return _retriever_instance


class HybridRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.cfg = config["retrieval"]
        self._chroma_dir = resolve_chroma_dir(self.cfg["chroma_persist_dir"])
        self._chroma = None
        self._dense_available = True
        self._dense_recovery_attempted = False
        self._bm25 = None
        self._bm25_documents: list[dict[str, Any]] | None = None
        self._bm25_loaded = False
        self._reranker: LLMReranker | None = None

    @property
    def chroma(self):
        if not self._dense_available:
            return None
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            self._chroma = client.get_or_create_collection(self.cfg["chroma_collection"])
        return self._chroma

    @property
    def bm25(self):
        if not self._bm25_loaded:
            self._load_bm25()
        return self._bm25

    @property
    def bm25_documents(self) -> list[dict[str, Any]]:
        if not self._bm25_loaded:
            self._load_bm25()
        return self._bm25_documents or []

    @property
    def reranker(self) -> LLMReranker:
        if self._reranker is None:
            reranker_mode = str(self.cfg.get("reranker_mode", "llm")).strip().lower()
            if reranker_mode not in {"llm", "lexical"}:
                LOGGER.warning(
                    "Unknown reranker_mode %r; falling back to lexical reranking.",
                    reranker_mode,
                )
                reranker_mode = "lexical"
            llm_client = None
            if reranker_mode == "llm" and isinstance(self.config, dict) and "llm" in self.config:
                try:
                    from src.llm.client import LLMClient

                    llm_client = LLMClient(self.config)
                except Exception:
                    llm_client = None
            self._reranker = LLMReranker(llm_client, mode=reranker_mode)
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[Document]:
        candidate_k = max(top_k * 2, int(self.cfg.get("rerank_top_k", top_k)))
        try:
            dense = self._dense_search(query, candidate_k, filters)
        except Exception as exc:
            LOGGER.warning("Dense retrieval failed for query %r: %s", query, exc)
            dense = []

        try:
            sparse = self._sparse_search(query, candidate_k, filters)
        except Exception as exc:
            LOGGER.warning("Sparse retrieval failed for query %r: %s", query, exc)
            sparse = []

        fused = _reciprocal_rank_fusion(dense, sparse, top_k=candidate_k)
        if not fused:
            return []

        rerank_window = max(top_k, int(self.cfg.get("rerank_top_k", candidate_k)))
        candidates = fused[:rerank_window]
        reranked = self.reranker.rerank(query, candidates, top_k=min(top_k, len(candidates)))

        if len(reranked) >= top_k:
            return reranked[:top_k]

        seen_urls = {doc.url for doc in reranked}
        for doc in fused[rerank_window:]:
            if len(reranked) >= top_k:
                break
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            reranked.append(doc)
        return reranked[:top_k]

    def _dense_search(self, query: str, top_k: int, filters) -> list[Document]:
        collection = self.chroma
        if collection is None:
            return []
        where = _build_chroma_where(filters) if filters else None
        try:
            results = collection.query(query_texts=[query], n_results=top_k, where=where)
        except Exception as exc:
            if _is_hnsw_load_error(exc) and self._should_recover_dense_error():
                collection = None
                if self._recover_chroma_index(exc):
                    retry_collection = self.chroma
                    if retry_collection is None:
                        return []
                    try:
                        results = retry_collection.query(
                            query_texts=[query], n_results=top_k, where=where
                        )
                    except Exception as retry_exc:
                        if _is_hnsw_load_error(retry_exc):
                            self._disable_dense_search(retry_exc)
                        raise retry_exc
                else:
                    self._disable_dense_search(exc)
                    raise exc
            else:
                if _is_hnsw_load_error(exc):
                    self._disable_dense_search(exc)
                raise exc
        return _chroma_results_to_docs(results)

    def _sparse_search(self, query: str, top_k: int, filters) -> list[Document]:
        if self.bm25 is None or not self.bm25_documents:
            return []
        tokens = _tokenize_query(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        docs = []
        for index, score in ranked:
            if len(docs) >= top_k:
                break
            if index >= len(self.bm25_documents) or score <= 0:
                continue
            doc = _chunk_to_doc(self.bm25_documents[index])
            if _doc_matches_filters(doc, filters):
                docs.append(doc)
        return docs

    def _load_bm25(self) -> None:
        self._bm25_loaded = True
        idx_path = Path(self.cfg["bm25_index_path"])
        if not idx_path.exists():
            return

        with open(idx_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self._bm25 = payload.get("bm25")
            self._bm25_documents = payload.get("documents") or payload.get("chunks") or []
            return

        # Legacy format only contains the BM25 object, so sparse retrieval is disabled.
        self._bm25 = payload
        self._bm25_documents = []

    def _should_recover_dense_error(self) -> bool:
        return self._dense_available and not self._dense_recovery_attempted

    def _recover_chroma_index(self, exc: Exception) -> bool:
        chunks_path = Path(self.cfg.get("chunks_path", DEFAULT_CHUNKS_PATH))
        if not chunks_path.is_absolute():
            chunks_path = Path(__file__).resolve().parents[2] / chunks_path
        if not chunks_path.exists():
            LOGGER.warning(
                "Chroma HNSW recovery skipped because chunk source is missing: %s",
                chunks_path,
            )
            return False

        self._dense_recovery_attempted = True
        self._reset_chroma_connection()
        LOGGER.warning(
            "Detected Chroma HNSW load failure. Rebuilding dense index from %s: %s",
            chunks_path,
            exc,
        )
        try:
            doc_count = rebuild_chroma_from_chunks(
                chunks_path=chunks_path,
                chroma_dir=self._chroma_dir,
                collection_name=self.cfg["chroma_collection"],
            )
        except Exception as rebuild_exc:
            LOGGER.warning("Chroma index rebuild failed: %s", rebuild_exc)
            return False

        self._dense_available = True
        self._reset_chroma_connection()
        LOGGER.info("Chroma dense index rebuilt successfully with %s docs", doc_count)
        return True

    def _disable_dense_search(self, reason: Exception) -> None:
        if self._dense_available:
            LOGGER.warning(
                "Disabling dense retrieval for this process after Chroma failure: %s",
                reason,
            )
        self._dense_available = False
        self._reset_chroma_connection()

    def _reset_chroma_connection(self) -> None:
        self._chroma = None
        gc.collect()


def _reciprocal_rank_fusion(
    dense: list[Document], sparse: list[Document], top_k: int, k: int = 60
) -> list[Document]:
    """Standard RRF: score = sum(1 / (k + rank)) across retrieval systems."""
    scores: dict[str, float] = {}
    url_to_doc: dict[str, Document] = {}

    for rank, doc in enumerate(dense):
        scores[doc.url] = scores.get(doc.url, 0) + 1 / (k + rank + 1)
        url_to_doc[doc.url] = doc

    for rank, doc in enumerate(sparse):
        scores[doc.url] = scores.get(doc.url, 0) + 1 / (k + rank + 1)
        url_to_doc[doc.url] = doc

    sorted_urls = sorted(scores, key=lambda u: -scores[u])
    return [url_to_doc[u] for u in sorted_urls[:top_k]]


def _build_chroma_where(filters: dict) -> dict | None:
    clauses = []
    for k, v in filters.items():
        if k == "chapter__lte":
            clauses.append({"chapter": {"$lte": v}})
        else:
            clauses.append({_normalize_filter_key(k): {"$eq": v}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _doc_matches_filters(doc: Document, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if key == "chapter__lte":
            if doc.chapter is None or doc.chapter > value:
                return False
            continue
        field_value = None
        if key in DOCUMENT_FILTER_FIELDS:
            field_value = getattr(doc, key, None)
        elif key.startswith("meta_"):
            field_value = doc.metadata.get(key[5:])
        else:
            field_value = doc.metadata.get(key)
        if field_value != value:
            return False
    return True


def _normalize_filter_key(key: str) -> str:
    if key.startswith("meta_") or key in DOCUMENT_FILTER_FIELDS:
        return key
    return f"meta_{key}"


def _chunk_to_doc(chunk: dict[str, Any]) -> Document:
    return Document(
        text=chunk.get("text", ""),
        source=chunk.get("source", ""),
        url=chunk.get("url", ""),
        chapter=chunk.get("chapter"),
        entity=chunk.get("entity"),
        credibility=chunk.get("credibility", 0.8),
        post_date=chunk.get("post_date"),
        metadata=chunk.get("metadata", {}),
    )


def _tokenize_query(query: str) -> list[str]:
    cleaned = re.sub(r"[\u3000\s]+", " ", query).strip()
    if not cleaned:
        return []

    tokens: list[str] = []
    for raw_token in re.split(r"\s+", cleaned):
        if not raw_token:
            continue
        tokens.extend(_normalize_query_token(raw_token))

    return list(dict.fromkeys(token for token in tokens if token))


def _normalize_query_token(token: str) -> list[str]:
    if not re.search(r"[\u4e00-\u9fff]", token):
        return [token]

    normalized = re.sub(r"[，。？！、,:;!?.（）()【】\[\]\"'`]+", " ", token)
    for stopword in sorted(CHINESE_QUERY_STOPWORDS, key=len, reverse=True):
        normalized = normalized.replace(stopword, " ")

    cjk_tokens: list[str] = []
    for chunk in normalized.split():
        for part in _split_cjk_keywords(chunk):
            if part and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", part):
                cjk_tokens.append(part)

    return cjk_tokens or [token]


def _split_cjk_keywords(token: str) -> list[str]:
    for keyword in CHINESE_QUERY_KEYWORDS:
        if keyword in token and token != keyword:
            left, right = token.split(keyword, 1)
            parts: list[str] = []
            if left:
                parts.extend(_split_cjk_keywords(left))
            parts.append(keyword)
            if right:
                parts.extend(_split_cjk_keywords(right))
            return parts
    return [token]


def _chroma_results_to_docs(results: dict) -> list[Document]:
    docs = []
    if not results or not results.get("documents"):
        return docs
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        payload_meta = _restore_chroma_metadata(meta)
        docs.append(Document(
            text=text,
            source=meta.get("source", ""),
            url=meta.get("url", ""),
            chapter=meta.get("chapter"),
            entity=meta.get("entity"),
            credibility=meta.get("credibility", 0.8),
            post_date=meta.get("post_date"),
            metadata=payload_meta,
        ))
    return docs


def _restore_chroma_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    restored = {}
    for key, value in meta.items():
        if key.startswith("meta_"):
            restored[key[5:]] = value
        else:
            restored[key] = value
    return restored


def _is_hnsw_load_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in HNSW_ERROR_MARKERS)
