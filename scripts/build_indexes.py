"""
One-time script: read data/processed/chunks.jsonl → build ChromaDB + BM25 indexes.
Run AFTER chunk_and_clean.py.
Run: python scripts/build_indexes.py
"""

import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CHROMA_DIR = "data/indexes/chroma_db"
BM25_PATH = Path("data/indexes/bm25_index.pkl")
COLLECTION_NAME = "wukong_chunks"
BATCH_SIZE = 100


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_chroma(chunks: list[dict]):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[_to_chroma_metadata(c) for c in batch],
            ids=[f"chunk_{i + j}" for j, _ in enumerate(batch)],
        )
        print(f"  chroma: {i + len(batch)}/{len(chunks)}")

    print(f"ChromaDB: {collection.count()} docs in '{COLLECTION_NAME}'")


def build_bm25(chunks: list[dict]):
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": chunks}, f)
    print(f"BM25 index saved to {BM25_PATH}")


def _to_chroma_metadata(chunk: dict) -> dict:
    metadata = {
        k: v
        for k, v in chunk.items()
        if k not in {"text", "metadata"} and v is not None
    }
    for key, value in (chunk.get("metadata") or {}).items():
        if value is not None:
            metadata[f"meta_{key}"] = value
    return metadata


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Building ChromaDB index...")
    build_chroma(chunks)

    print("Building BM25 index...")
    build_bm25(chunks)

    print("Done.")


if __name__ == "__main__":
    main()
