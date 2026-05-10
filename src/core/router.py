"""
Intent classification: maps a user query to a named workflow.
Uses a single cheap LLM call with a structured prompt.
"""

import json
import re

from src.core.state import AgentState

INTENT_CATEGORIES = {
    "boss_strategy": "如何打某个 boss，招式如何破解，打法推荐",
    "decision_making": "装备选择、技能点分配、流派搭配",
    "navigation": "某个道具/NPC/地点在哪里",
    "fact_lookup": "某个机制/属性/数值是什么",
}


class Router:
    def __init__(self, config: dict, llm_client=None):
        self.config = config
        self.llm = llm_client  # injected to avoid circular import

    def route(self, state: AgentState) -> str:
        """Classify query intent and fall back to heuristics until LLM routing is ready."""
        query = state.user_query
        override = self._priority_rule_route(query)
        if override:
            return override
        if self.llm is not None:
            try:
                return self._llm_route(query)
            except NotImplementedError:
                pass
            except Exception:
                pass
        return self._heuristic_route(query)

    def _llm_route(self, query: str) -> str:
        prompt = self._build_routing_prompt(query)
        response = self.llm.complete(
            [{"role": "user", "content": prompt}],
            system=(
                "你是一个游戏问题意图分类器。"
                "你只能返回以下四个类别之一："
                "boss_strategy, decision_making, navigation, fact_lookup。"
                "不要解释，不要补充文字。"
            ),
        )
        return self._parse_intent(response)

    def _heuristic_route(self, query: str) -> str:
        """Keyword fallback used during development."""
        boss_keywords = ["怎么打", "破解", "闪避", "招式", "boss", "先锋", "大圣"]
        decision_keywords = ["选择", "装备", "技能点", "流派", "哪个好"]
        nav_keywords = ["在哪", "怎么找", "位置", "NPC"]
        if any(k in query for k in boss_keywords):
            return "boss_strategy"
        if any(k in query for k in decision_keywords):
            return "decision_making"
        if any(k in query for k in nav_keywords):
            return "navigation"
        return "fact_lookup"

    def _priority_rule_route(self, query: str) -> str | None:
        if self._is_fact_listing_or_count_query(query):
            return "fact_lookup"
        return None

    def _is_fact_listing_or_count_query(self, query: str) -> bool:
        listing_or_count_keywords = (
            "几个",
            "多少",
            "哪些",
            "有哪些",
            "哪几个",
            "有哪几",
            "几种",
            "都叫什么",
            "叫什么",
            "什么招式",
            "招式名字",
            "大招名字",
        )
        action_keywords = (
            "怎么躲",
            "怎么打",
            "如何打",
            "打法",
            "闪避",
            "躲",
            "打",
            "推荐",
            "建议",
            "应对",
            "破解",
        )
        has_fact_shape = any(keyword in query for keyword in listing_or_count_keywords)
        has_action_shape = any(keyword in query for keyword in action_keywords)
        return has_fact_shape and not has_action_shape

    def _build_routing_prompt(self, query: str) -> str:
        categories = "\n".join(f"- {k}: {v}" for k, v in INTENT_CATEGORIES.items())
        return (
            f"将以下游戏问题分类到最合适的类别。只返回类别名称，不加任何解释。\n\n"
            f"类别:\n{categories}\n\n"
            f"问题: {query}"
        )

    def _parse_intent(self, response: str) -> str:
        cleaned = response.strip().strip("`")
        if cleaned in INTENT_CATEGORIES:
            return cleaned

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            for key in ("workflow", "intent", "category"):
                value = payload.get(key)
                if value in INTENT_CATEGORIES:
                    return value

        pattern = "|".join(re.escape(intent) for intent in INTENT_CATEGORIES)
        match = re.search(pattern, cleaned)
        if match:
            return match.group(0)

        raise ValueError(f"Unrecognized routing response: {response!r}")
