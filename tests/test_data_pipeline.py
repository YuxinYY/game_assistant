"""
Targeted tests for chunking and index build prerequisites.
"""

import json

from scripts.build_indexes import _to_chroma_metadata
from src.retrieval import index_builder
from src.retrieval.index_builder import build_chroma, resolve_chroma_dir
from scripts.chunk_and_clean import (
    chunk_text,
    iter_wiki_files,
    process_reddit_jsonl,
    process_wiki,
)


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


def test_chunk_text_avoids_mid_word_or_mid_sentence_overlap_starts():
    text = (
        "Boss: Stone Vanguard. Type: Yaoguai King. Chapter: 2. "
        "Summary: The Stone Vanguard is a required Yaoguai King boss fight that you can encounter in Chapter 2 of Black Myth Wukong. "
        "Located near the end of the Fright Cliff area, imposing Rock Guai sports a formidable defense and exploding attacks. "
        "Location: As a required boss, you won't be able to proceed to the Chapter 2 End Boss without defeating him and obtaining one half of the Key Items needed to open various doors in Chapter 2's Yellow Wind Ridge. "
        "You can find this Vanguard in the Fright Cliff Region by taking a right after heading through the Sandgate Village."
    )

    chunks = chunk_text(text, max_chars=180, overlap=70)

    assert len(chunks) >= 2
    assert chunks[1].startswith(("Summary:", "Located near", "Location:"))
    assert not chunks[1].startswith(("d obtaining", "nd obtaining", "obtaining one"))


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


def test_build_chroma_resets_persist_dir_before_rebuild(tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    stale_file = chroma_dir / "stale.bin"
    stale_file.write_text("stale", encoding="utf-8")

    count = build_chroma(
        [
            {
                "text": "Tiger Vanguard guide",
                "source": "wiki",
                "url": "http://wiki/tiger",
                "metadata": {"language": "en"},
            }
        ],
        chroma_dir=chroma_dir,
        collection_name="test_chunks",
        batch_size=1,
    )

    assert count == 1
    assert not stale_file.exists()


def test_resolve_chroma_dir_moves_onedrive_windows_paths_to_local_cache(monkeypatch, tmp_path):
    one_drive_root = tmp_path / "OneDrive"
    local_app_data = tmp_path / "LocalAppData"
    requested_path = one_drive_root / "game_assistant" / "data" / "indexes" / "chroma_db"

    monkeypatch.setenv("OneDrive", str(one_drive_root))
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    resolved = resolve_chroma_dir(requested_path)

    if index_builder.os.name == "nt":
        assert resolved.is_relative_to(local_app_data)
        assert resolved.name.startswith("chroma_")
    else:
        assert resolved == requested_path


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


def test_process_reddit_jsonl_preserves_english_metadata_and_entity(tmp_path):
    reddit_file = tmp_path / "reddit_sample.jsonl"
    reddit_file.write_text(
        json.dumps(
            {
                "post_id": "reddit_001",
                "author": "playerA",
                "title": "Tiger Vanguard delayed slam timing",
                "content": "Dodge late, then punish after the slam recovery.",
                "url": "https://reddit.example/post-1",
                "boss_tags": ["Tiger Vanguard"],
                "source": "reddit",
                "language": "en",
                "timestamp": "2024-09-12",
                "chapter": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunks = process_reddit_jsonl(reddit_file)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "reddit"
    assert chunks[0]["entity"] == "Tiger Vanguard"
    assert chunks[0]["metadata"] == {
        "author": "playerA",
        "title": "Tiger Vanguard delayed slam timing",
        "language": "en",
    }