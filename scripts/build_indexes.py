"""
One-time script: read data/processed/chunks.jsonl → build ChromaDB + BM25 indexes.
Run AFTER chunk_and_clean.py.
Run: python scripts/build_indexes.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.index_builder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BM25_PATH as BM25_PATH,
    DEFAULT_CHROMA_DIR as CHROMA_DIR,
    DEFAULT_CHUNKS_PATH as CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME as COLLECTION_NAME,
    _to_chroma_metadata,
    build_bm25,
    build_chroma,
    load_chunks,
    resolve_chroma_dir,
)


def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    print("Building ChromaDB index...")
    print(f"Resolved Chroma path: {resolve_chroma_dir(CHROMA_DIR)}")
    chroma_count = build_chroma(
        chunks,
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        batch_size=DEFAULT_BATCH_SIZE,
        progress_callback=lambda done, total: print(f"  chroma: {done}/{total}"),
    )
    print(f"ChromaDB: {chroma_count} docs in '{COLLECTION_NAME}'")

    print("Building BM25 index...")
    build_bm25(chunks, bm25_path=BM25_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
