"""
Consensus quantification tools.
count_source_consensus: how many distinct sources support each strategy.
detect_conflicts: find strategy pairs that contradict each other.
"""

from collections import defaultdict
from src.core.state import Document


STRATEGY_KEYWORDS: dict[str, list[str]] = {
    "侧向闪避": ["侧闪", "侧向闪", "向左闪", "向右闪", "完美闪避"],
    "棍反": ["棍反", "弹反", "格挡反击"],
    "定身术": ["定身", "冰结"],
    "拉开距离": ["拉开", "保持距离", "远离"],
    "变身广智": ["广智", "火狼", "变身"],
}


def count_source_consensus(docs: list[Document], topic: str) -> list[dict]:
    """
    For each known strategy keyword group, count distinct source URLs that mention it.
    Returns list sorted by source_count descending.
    """
    strategy_sources: dict[str, set[str]] = defaultdict(set)
    strategy_platform: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for doc in docs:
        for strategy, keywords in STRATEGY_KEYWORDS.items():
            if any(kw in doc.text for kw in keywords):
                strategy_sources[strategy].add(doc.url)
                strategy_platform[strategy][doc.source].add(doc.url)

    results = []
    for strategy, urls in sorted(strategy_sources.items(), key=lambda x: -len(x[1])):
        results.append({
            "label": strategy,
            "source_count": len(urls),
            "sources": {src: len(u) for src, u in strategy_platform[strategy].items()},
            "is_contested": False,  # filled in by detect_conflicts
        })

    return results


def detect_conflicts(docs: list[Document]) -> list[dict]:
    """
    Identify topics where docs contain contradictory claims.
    Simple heuristic: count pro/con mentions of key contested topics.
    """
    contested_topics = {
        "棍反": {
            "pro": ["能棍反", "可以棍反", "棍反成功"],
            "con": ["不能棍反", "棍反不稳", "棍反失败", "弹不了"],
        },
        "定身有效": {
            "pro": ["定身有效", "定身成功", "可以定身"],
            "con": ["定身无效", "定不住", "boss免疫"],
        },
    }

    conflicts = []
    for topic, keywords in contested_topics.items():
        pro_docs = [d for d in docs if any(kw in d.text for kw in keywords["pro"])]
        con_docs = [d for d in docs if any(kw in d.text for kw in keywords["con"])]
        if pro_docs and con_docs:
            conflicts.append({
                "topic": topic,
                "pro": [d.url for d in pro_docs],
                "con": [d.url for d in con_docs],
            })
    return conflicts
