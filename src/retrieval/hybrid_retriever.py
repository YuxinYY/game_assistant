"""
Hybrid retriever: fuses dense (ChromaDB) + sparse (BM25) results via Reciprocal Rank Fusion.
Singleton pattern so the index loads once per process.
"""

import pickle
from pathlib import Path
from typing import Optional
from src.core.state import Document

_retriever_instance = None


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

    @property
    def chroma(self):
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.cfg["chroma_persist_dir"])
            self._chroma = client.get_or_create_collection(self.cfg["chroma_collection"])
        return self._chroma

    @property
    def bm25(self):
        if self._bm25 is None:
            idx_path = Path(self.cfg["bm25_index_path"])
            if idx_path.exists():
                with open(idx_path, "rb") as f:
                    self._bm25 = pickle.load(f)
        return self._bm25

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
        if self.bm25 is None:
            return []
        # TODO: apply filters post-retrieval for BM25
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        # TODO: map scores back to Document objects
        raise NotImplementedError("BM25 search not fully implemented")


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
    where = {}
    for k, v in filters.items():
        if k == "chapter__lte":
            where["chapter"] = {"$lte": v}
        else:
            where[k] = {"$eq": v}
    return where if where else None


def _chroma_results_to_docs(results: dict) -> list[Document]:
    docs = []
    if not results or not results.get("documents"):
        return docs
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(
            text=text,
            source=meta.get("source", ""),
            url=meta.get("url", ""),
            chapter=meta.get("chapter"),
            entity=meta.get("entity"),
            credibility=meta.get("credibility", 0.8),
            post_date=meta.get("post_date"),
            metadata=meta,
        ))
    return docs
