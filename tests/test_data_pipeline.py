"""
Targeted tests for chunking and index build prerequisites.
"""

import json

from scripts.build_indexes import _to_chroma_metadata
from scripts.chunk_and_clean import chunk_text, process_wiki


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "虎先锋是第二章的重要 boss。"

    chunks = chunk_text(text, max_chars=600, overlap=100)

    assert chunks == [text]


def test_chunk_text_makes_progress_on_overlapping_windows():
    text = "。".join([f"第{i}句测试文本" for i in range(80)]) + "。"

    chunks = chunk_text(text, max_chars=80, overlap=20)

    assert len(chunks) > 1
    assert len(chunks) < 50
    assert all(chunk.strip() for chunk in chunks)


def test_to_chroma_metadata_flattens_nested_metadata():
    metadata = _to_chroma_metadata(
        {
            "text": "虎先锋攻略",
            "source": "nga",
            "url": "http://nga/1",
            "metadata": {"author": "玩家A", "title": "速通心得"},
        }
    )

    assert metadata["source"] == "nga"
    assert metadata["meta_author"] == "玩家A"
    assert metadata["meta_title"] == "速通心得"


def test_process_wiki_uses_moves_and_tips_when_raw_text_is_missing(tmp_path):
    sample = {
        "boss_name": "虎先锋",
        "url": "https://wiki.example/tiger-vanguard",
        "moves": [
            {"name": "虎跃斩", "description": "起跳后落爪三连击。", "phase": 1},
            {"name": "震地怒吼", "description": "第二阶段范围震击。", "phase": 2},
        ],
        "tips": "建议优先侧闪。",
    }
    wiki_file = tmp_path / "tiger_vanguard.json"
    wiki_file.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    chunks = process_wiki(wiki_file)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "wiki"
    assert "虎跃斩" in chunks[0]["text"]
    assert "建议优先侧闪" in chunks[0]["text"]