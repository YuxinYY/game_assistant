"""
Spoiler filter: remove or flag doc chunks that reveal content
beyond the player's current chapter.
"""

from src.core.state import Document

# Content that only exists in later chapters
CHAPTER_CONTENT: dict[int, list[str]] = {
    3: ["广智", "小西天", "观音禅院", "黄眉"],
    4: ["盘丝岭", "蜘蛛精", "百眼魔君"],
    5: ["火焰山", "红孩儿", "牛魔王"],
    6: ["须弥山"],
}


def apply_spoiler_filter(docs: list[Document], max_chapter: int | None) -> list[Document]:
    """
    Remove docs that mention content gated behind chapters > max_chapter.
    If a doc has an explicit chapter field, use that; otherwise fall back to keyword scan.
    """
    if max_chapter is None:
        return list(docs)

    allowed = []
    for doc in docs:
        if doc.chapter is not None and doc.chapter > max_chapter:
            continue
        if _contains_spoiler(doc.text, max_chapter):
            continue
        allowed.append(doc)
    return allowed


def _contains_spoiler(text: str, max_chapter: int) -> bool:
    for chapter, keywords in CHAPTER_CONTENT.items():
        if chapter > max_chapter and any(kw in text for kw in keywords):
            return True
    return False
