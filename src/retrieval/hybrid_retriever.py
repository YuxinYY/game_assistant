"""
Hybrid retriever: fuses dense (ChromaDB) + sparse (BM25) results via Reciprocal Rank Fusion.
Singleton pattern so the index loads once per process.
"""

import pickle
from pathlib import Path
from typing import Any, Optional
from src.core.state import Document

_retriever_instance = None
DOCUMENT_FILTER_FIELDS = {"text", "source", "url", "chapter", "entity", "credibility", "post_date"}


def get_retriever(config: dict | None = None) -> "HybridRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        import yaml
        if config is None:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
        _retriever_instance = HybridRetriever(config)
    return _retriever_instance


class HybridRetriever:
    def __init__(self, config: dict):
        self.cfg = config["retrieval"]
        self._chroma = None
        self._bm25 = None
        self._bm25_documents: list[dict[str, Any]] | None = None
        self._bm25_loaded = False

    @property
    def chroma(self):
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.cfg["chroma_persist_dir"])
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

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[Document]:
        dense = self._dense_search(query, top_k * 2, filters)
        sparse = self._sparse_search(query, top_k * 2, filters)
        fused = _reciprocal_rank_fusion(dense, sparse, top_k=top_k)
        return fused

    def _dense_search(self, query: str, top_k: int, filters) -> list[Document]:
        if self.chroma is None:
            return []
        where = _build_chroma_where(filters) if filters else None
        results = self.chroma.query(
            query_texts=[query], n_results=top_k, where=where
        )
        return _chroma_results_to_docs(results)

    def _sparse_search(self, query: str, top_k: int, filters) -> list[Document]:
        if self.bm25 is None or not self.bm25_documents:
            return []
        tokens = query.split()
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
