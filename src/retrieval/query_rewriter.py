"""
Query rewriter: expands a vague player description into multiple search queries.
E.g. "那个转圈蓄力的招" → ["虎跃斩", "虎先锋 蓄力 攻击", "tiger vanguard charge attack"]

This is one of the highest-leverage modules: retrieval recall depends heavily on query quality.
"""

from __future__ import annotations

import json
import re


class QueryRewriter:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def rewrite(self, query: str, known_entities: list[str] | None = None) -> list[str]:
        """
        Returns a list of rewritten queries.
        Falls back to simple expansion rules if LLM not available.
        """
        entities = known_entities or []
        base_queries = self._rule_based_rewrite(query, entities)
        if self.llm is None or not hasattr(self.llm, "complete"):
            return base_queries

        try:
            response = self.llm.complete(
                [{"role": "user", "content": self._build_prompt(query, entities)}],
                system=(
                    "你负责为游戏攻略检索改写查询。"
                    "输出 JSON，格式为 {\"queries\": [\"...\"]}。"
                    "保持短语化，不要解释。"
                ),
            )
            llm_queries = self._parse_queries(response)
        except Exception:
            llm_queries = []

        merged = [q for q in llm_queries + base_queries if q]
        return list(dict.fromkeys(merged))

    def _rule_based_rewrite(self, query: str, entities: list[str]) -> list[str]:
        queries = [query]

        for entity in entities:
            queries.append(f"{entity} {query}")
            queries.append(f"{entity} 攻略")
            queries.append(f"{entity} 怎么打")
            if self._contains_latin(entity) or self._contains_latin(query):
                queries.append(f"{entity} boss guide")
                queries.append(f"how to beat {entity}")
            else:
                queries.append(f"{entity} 招式 躲避")
                queries.append(f"{entity} 打法")

        if any(keyword in query for keyword in ["怎么躲", "躲", "闪", "避"]):
            queries.append(f"{query} 躲避")
            queries.append(f"{query} dodge timing")
        if any(keyword in query for keyword in ["怎么打", "打法", "攻略"]):
            queries.append(f"{query} guide")

        return list(dict.fromkeys(queries))  # deduplicate, preserve order

    @staticmethod
    def _contains_latin(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text))

    def _parse_queries(self, response: str) -> list[str]:
        payload = self._extract_json(response)
        if isinstance(payload, dict):
            values = payload.get("queries")
            if isinstance(values, list):
                return [str(item).strip() for item in values if str(item).strip()]

        quoted_values = [
            match.strip()
            for match in re.findall(r'"([^"\n]+)"', response)
            if match.strip() and match.strip().lower() != "queries"
        ]
        if quoted_values:
            return list(dict.fromkeys(quoted_values))

        lines = []
        for line in response.splitlines():
            normalized = re.sub(r"^[\-\d\.)\s]+", "", line).strip()
            normalized = normalized.strip(", ").strip("\"'")
            if normalized and normalized.lower() != "queries" and not normalized.startswith("{"):
                lines.append(normalized)
        return lines

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _build_prompt(self, query: str, entities: list[str]) -> str:
        entity_hint = f"\n已知实体: {entities}" if entities else ""
        return (
            f"玩家问题: {query}{entity_hint}\n\n"
            "将上述问题扩展为3-5个不同的搜索查询，用于在游戏攻略数据库中检索。\n"
            "包括: 中文口语版、wiki标准术语版、英文版（如适用）。\n"
            "每行一个查询，不加编号。"
        )
