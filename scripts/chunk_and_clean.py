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

SENTENCE_BREAK_PATTERN = re.compile(r"[。！？.!?]")


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
        window = text[start:end]
        break_positions = [match.end() for match in SENTENCE_BREAK_PATTERN.finditer(window)]
        if break_positions:
            candidate_end = start + break_positions[-1]
            if candidate_end > start + max_chars // 2:
                end = candidate_end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = _align_next_chunk_start(
            text,
            current_start=start,
            current_end=end,
            desired_start=max(end - overlap, start + 1),
        )
    return chunks


def _align_next_chunk_start(text: str, current_start: int, current_end: int, desired_start: int) -> int:
    desired_start = max(min(desired_start, current_end), current_start + 1)

    next_sentence_start = _find_next_sentence_start(text, desired_start, current_end)
    if next_sentence_start is not None:
        return next_sentence_start

    previous_sentence_start = _find_previous_sentence_start(text, current_start, desired_start, current_end)
    if previous_sentence_start is not None:
        return previous_sentence_start

    next_word_start = _find_next_word_start(text, desired_start, current_end)
    if next_word_start is not None:
        return next_word_start

    return desired_start


def _find_next_sentence_start(text: str, desired_start: int, current_end: int) -> int | None:
    for match in SENTENCE_BREAK_PATTERN.finditer(text, desired_start, current_end):
        candidate = _skip_inline_whitespace(text, match.end(), current_end)
        if candidate < current_end:
            return candidate
    return None


def _find_previous_sentence_start(text: str, current_start: int, desired_start: int, current_end: int) -> int | None:
    matches = list(SENTENCE_BREAK_PATTERN.finditer(text, current_start, desired_start))
    for match in reversed(matches):
        candidate = _skip_inline_whitespace(text, match.end(), current_end)
        if current_start < candidate < current_end:
            return candidate
    return None


def _find_next_word_start(text: str, desired_start: int, current_end: int) -> int | None:
    for index in range(desired_start, current_end):
        if text[index].isspace():
            candidate = _skip_inline_whitespace(text, index, current_end)
            if candidate < current_end:
                return candidate
    return None


def _skip_inline_whitespace(text: str, index: int, current_end: int) -> int:
    while index < current_end and text[index].isspace():
        index += 1
    return index


def process_wiki(file: Path) -> list[dict]:
    data = json.loads(file.read_text(encoding="utf-8"))
    boss = data["boss_name"]
    chapter = data.get("chapter", 1)
    raw_text = data.get("raw_text") or _synthesize_wiki_text(data)
    chunks = chunk_text(raw_text)
    source_site = str(data.get("source_site") or "bwiki").strip() or "bwiki"
    source_language = str(data.get("source_language") or _infer_wiki_language(source_site)).strip()
    page_title = str(data.get("page_title") or data.get("title") or boss).strip() or boss
    return [
        {
            "text": c,
            "source": "wiki",
            "url": data["url"],
            "chapter": chapter,
            "entity": boss,
            "credibility": CREDIBILITY["wiki"],
            "post_date": None,
            "metadata": {
                "author": source_site,
                "language": source_language,
                "title": page_title,
            },
        }
        for c in chunks if c.strip()
    ]


def _infer_wiki_language(source_site: str) -> str:
    return "en" if source_site.lower() == "ign" else "zh"


def _synthesize_wiki_text(data: dict) -> str:
    sections = []
    boss_name = data.get("boss_name")
    if boss_name:
        sections.append(f"Boss：{boss_name}。")

    for move in data.get("moves", []):
        name = move.get("name", "")
        description = move.get("description", "")
        phase = move.get("phase")
        if name and description:
            phase_prefix = f"第{phase}阶段" if phase is not None else ""
            sections.append(f"{phase_prefix}{name}：{description}")

    tips = data.get("tips")
    if tips:
        sections.append(f"打法提示：{tips}")

    return " ".join(section.strip() for section in sections if section.strip())


def iter_wiki_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    wiki_dir = raw_dir / "wiki"
    files = sorted(wiki_dir.glob("*.json"))
    if not files:
        return []

    selected: list[Path] = []
    sample_files: list[Path] = []
    real_boss_names: set[str] = set()

    for file in files:
        if file.stem.endswith("_sample"):
            sample_files.append(file)
            continue
        selected.append(file)
        boss_name = _read_wiki_boss_name(file)
        if boss_name:
            real_boss_names.add(boss_name)

    for file in sample_files:
        boss_name = _read_wiki_boss_name(file)
        if boss_name and boss_name in real_boss_names:
            continue
        selected.append(file)

    return selected


def _read_wiki_boss_name(file: Path) -> str:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("boss_name", "")).strip()


def process_nga_jsonl(file: Path) -> list[dict]:
    return _process_community_jsonl(file, source="nga", default_language="zh")


def process_bilibili_jsonl(file: Path) -> list[dict]:
    return _process_community_jsonl(file, source="bilibili", default_language="zh")


def process_reddit_jsonl(file: Path) -> list[dict]:
    return _process_community_jsonl(file, source="reddit", default_language="en")


def _process_community_jsonl(file: Path, source: str, default_language: str) -> list[dict]:
    results = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        post = json.loads(line)
        text = _build_community_text(post)
        if not text.strip():
            continue
        language = str(post.get("language") or default_language).strip() or default_language
        entity = _extract_post_entity(post)
        for chunk_text_str in chunk_text(text):
            results.append({
                "text": chunk_text_str,
                "source": source,
                "url": post.get("url", ""),
                "chapter": post.get("chapter"),
                "entity": entity,
                "credibility": CREDIBILITY[source],
                "post_date": post.get("timestamp", ""),
                "metadata": {
                    "author": post.get("author", ""),
                    "title": post.get("title", ""),
                    "language": language,
                },
            })
    return results


def _build_community_text(post: dict) -> str:
    title = str(post.get("title") or "").strip()
    content = str(post.get("content") or "").strip()
    summary = str(post.get("summary") or "").strip()
    return " ".join(part for part in [title, summary, content] if part)


def _extract_post_entity(post: dict) -> str | None:
    tags = post.get("boss_tags") or post.get("entities") or []
    if isinstance(tags, list):
        for tag in tags:
            cleaned = str(tag or "").strip()
            if cleaned:
                return cleaned
    entity = str(post.get("entity") or "").strip()
    return entity or None


def main():
    all_chunks = []

    for f in iter_wiki_files(RAW_DIR):
        all_chunks.extend(process_wiki(f))

    for f in (RAW_DIR / "nga").glob("*.jsonl"):
        all_chunks.extend(process_nga_jsonl(f))

    for f in (RAW_DIR / "bilibili").glob("*.jsonl"):
        all_chunks.extend(process_bilibili_jsonl(f))

    for f in (RAW_DIR / "reddit").glob("*.jsonl"):
        all_chunks.extend(process_reddit_jsonl(f))

    print(f"Total chunks: {len(all_chunks)}")
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
