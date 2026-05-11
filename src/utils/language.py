import re


PURE_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
PURE_LATIN_PATTERN = re.compile(r"[A-Za-z]")


def detect_query_language(text: str) -> str:
    has_cjk = bool(PURE_CJK_PATTERN.search(text or ""))
    has_latin = bool(PURE_LATIN_PATTERN.search(text or ""))
    if has_latin and not has_cjk:
        return "en"
    if has_cjk and not has_latin:
        return "zh"
    if has_cjk and has_latin:
        return "mixed"
    return "unknown"


def wants_english(text: str) -> bool:
    return detect_query_language(text) == "en"


def preferred_response_language(text: str) -> str:
    return "en" if wants_english(text) else "zh"