"""
Helpers for building and rebuilding retrieval indexes from processed chunks.
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Callable

import chromadb
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CHROMA_DIR = Path("data/indexes/chroma_db")
DEFAULT_BM25_PATH = Path("data/indexes/bm25_index.pkl")
DEFAULT_COLLECTION_NAME = "wukong_chunks"
DEFAULT_BATCH_SIZE = 100
LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int], None]


def load_chunks(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict]:
    path = _resolve_project_path(chunks_path)
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_chroma(
    chunks: list[dict],
    chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> int:
    chroma_path = resolve_chroma_dir(chroma_dir)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(collection_name)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            documents=[chunk["text"] for chunk in batch],
            metadatas=[_to_chroma_metadata(chunk) for chunk in batch],
            ids=[f"chunk_{start + offset}" for offset, _ in enumerate(batch)],
        )
        if progress_callback is not None:
            progress_callback(start + len(batch), len(chunks))

    return collection.count()


def rebuild_chroma_from_chunks(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> int:
    chunks = load_chunks(chunks_path)
    return build_chroma(
        chunks,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


def build_bm25(chunks: list[dict], bm25_path: str | Path = DEFAULT_BM25_PATH) -> None:
    path = _resolve_project_path(bm25_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [chunk["text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": chunks}, f)


def resolve_chroma_dir(chroma_dir: str | Path = DEFAULT_CHROMA_DIR) -> Path:
    requested_path = _resolve_project_path(chroma_dir)
    if not _should_relocate_chroma_dir(requested_path):
        return requested_path

    local_root = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT / ".cache")) / "game_assistant"
    digest = hashlib.sha1(str(requested_path).encode("utf-8")).hexdigest()[:10]
    relocated_path = local_root / f"chroma_{digest}"
    LOGGER.warning(
        "Relocating Chroma persist directory from %s to %s because Windows OneDrive-backed paths are unstable for HNSW indexes.",
        requested_path,
        relocated_path,
    )
    return relocated_path


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _should_relocate_chroma_dir(path: Path) -> bool:
    if os.name != "nt":
        return False

    path_string = str(path).lower()
    if "onedrive" in path_string:
        return True

    for env_name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = os.environ.get(env_name)
        if not root:
            continue
        try:
            if path.is_relative_to(Path(root).resolve()):
                return True
        except Exception:
            continue
    return False


def _to_chroma_metadata(chunk: dict) -> dict:
    metadata = {
        key: value
        for key, value in chunk.items()
        if key not in {"text", "metadata"} and value is not None
    }
    for key, value in (chunk.get("metadata") or {}).items():
        if value is not None:
            metadata[f"meta_{key}"] = value
    return metadata