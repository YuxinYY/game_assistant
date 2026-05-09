"""
One-time script: read data/raw/, clean + chunk → data/processed/chunks.jsonl
Each chunk gets full metadata for citation + filtering.
Run: python scripts/chunk_and_clean.py
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT = Path("data/processed/chunks.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MAX_CHUNK_CHARS = 600
OVERLAP_CHARS = 100

CREDIBILITY = {
    "wiki": 1.0,
    "nga": 0.85,
    "bilibili": 0.75,
    "reddit": 0.70,
}


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Sliding window chunker by character count. Splits on sentence boundaries when possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to find a sentence break near the end
        break_pos = text.rfind("。", start, end)
        if break_pos > start + max_chars // 2:
            end = break_pos + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def process_wiki(file: Path) -> list[dict]:
    data = json.loads(file.read_text(encoding="utf-8"))
    boss = data["boss_name"]
    chapter = data.get("chapter", 1)
    chunks = chunk_text(data.get("raw_text", ""))
    return [
        {
            "text": c,
            "source": "wiki",
            "url": data["url"],
            "chapter": chapter,
            "entity": boss,
            "credibility": CREDIBILITY["wiki"],
            "post_date": None,
            "metadata": {"author": "bwiki"},
        }
        for c in chunks if c.strip()
    ]


def process_nga_jsonl(file: Path) -> list[dict]:
    results = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        post = json.loads(line)
        for chunk_text_str in chunk_text(post.get("content", "")):
            results.append({
                "text": chunk_text_str,
                "source": "nga",
                "url": post.get("url", ""),
                "chapter": post.get("chapter"),
                "entity": post.get("boss_tags", [None])[0],
                "credibility": CREDIBILITY["nga"],
                "post_date": post.get("timestamp", ""),
                "metadata": {"author": post.get("author", ""), "title": post.get("title", "")},
            })
    return results


def main():
    all_chunks = []

    for f in (RAW_DIR / "wiki").glob("*.json"):
        all_chunks.extend(process_wiki(f))

    for f in (RAW_DIR / "nga").glob("*.jsonl"):
        all_chunks.extend(process_nga_jsonl(f))

    # TODO: add bilibili and reddit processors

    print(f"Total chunks: {len(all_chunks)}")
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
