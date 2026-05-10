"""
Targeted tests for chunking and index build prerequisites.
"""

import json

from scripts.build_indexes import _to_chroma_metadata
from scripts.chunk_and_clean import chunk_text, iter_wiki_files, process_wiki


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


def test_chunk_text_prefers_english_sentence_breaks_when_available():
    text = " ".join([
        "Tiger Vanguard opens with a fast punch combo.",
        "Wait for the delayed sweep before punishing.",
        "Do not attack into the stone stance.",
        "Save stamina for the sword follow-up.",
    ])

    chunks = chunk_text(text, max_chars=80, overlap=20)

    assert len(chunks) >= 2
    assert chunks[0].endswith(".")


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


def test_process_wiki_preserves_source_site_language_and_title_in_metadata(tmp_path):
    sample = {
        "boss_name": "Tiger Vanguard",
        "page_title": "Tiger Vanguard",
        "url": "https://www.ign.com/wikis/black-myth-wukong/Tiger_Vanguard",
        "source_site": "ign",
        "source_language": "en",
        "raw_text": "Tiger Vanguard is a Chapter 2 boss.",
    }
    wiki_file = tmp_path / "Tiger Vanguard.json"
    wiki_file.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    chunks = process_wiki(wiki_file)

    assert chunks[0]["metadata"] == {
        "author": "ign",
        "language": "en",
        "title": "Tiger Vanguard",
    }


def test_iter_wiki_files_skips_sample_when_real_file_exists(tmp_path):
    raw_dir = tmp_path / "raw"
    wiki_dir = raw_dir / "wiki"
    wiki_dir.mkdir(parents=True)

    real_file = wiki_dir / "虎先锋.json"
    real_file.write_text(
        json.dumps({"boss_name": "虎先锋", "url": "https://wiki.example/real"}, ensure_ascii=False),
        encoding="utf-8",
    )
    sample_file = wiki_dir / "tiger_vanguard_sample.json"
    sample_file.write_text(
        json.dumps({"boss_name": "虎先锋", "url": "https://wiki.example/sample"}, ensure_ascii=False),
        encoding="utf-8",
    )
    unmatched_sample = wiki_dir / "guangzhi_sample.json"
    unmatched_sample.write_text(
        json.dumps({"boss_name": "广智", "url": "https://wiki.example/guangzhi"}, ensure_ascii=False),
        encoding="utf-8",
    )

    selected = iter_wiki_files(raw_dir)
    selected_names = {path.name for path in selected}

    assert "虎先锋.json" in selected_names
    assert "tiger_vanguard_sample.json" not in selected_names
    assert "guangzhi_sample.json" in selected_names