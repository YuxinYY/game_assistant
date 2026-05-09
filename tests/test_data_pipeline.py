"""
Targeted tests for chunking and index build prerequisites.
"""

from scripts.build_indexes import _to_chroma_metadata
from scripts.chunk_and_clean import chunk_text


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