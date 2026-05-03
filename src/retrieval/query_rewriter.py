"""
Query rewriter: expands a vague player description into multiple search queries.
E.g. "那个转圈蓄力的招" → ["虎跃斩", "虎先锋 蓄力 攻击", "tiger vanguard charge attack"]

This is one of the highest-leverage modules: retrieval recall depends heavily on query quality.
"""


class QueryRewriter:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def rewrite(self, query: str, known_entities: list[str] | None = None) -> list[str]:
        """
        Returns a list of rewritten queries.
        Falls back to simple expansion rules if LLM not available.
        """
        if self.llm is None:
            return self._rule_based_rewrite(query, known_entities or [])
        # TODO: LLM-based rewrite
        # prompt = self._build_prompt(query, known_entities)
        # response = self.llm.complete([{"role": "user", "content": prompt}])
        # return self._parse_queries(response)
        raise NotImplementedError

    def _rule_based_rewrite(self, query: str, entities: list[str]) -> list[str]:
        queries = [query]
        # Add entity-anchored versions
        for entity in entities:
            queries.append(f"{entity} {query}")
            queries.append(f"{entity} 攻略")
            queries.append(f"{entity} 怎么打")
        return list(dict.fromkeys(queries))  # deduplicate, preserve order

    def _build_prompt(self, query: str, entities: list[str]) -> str:
        entity_hint = f"\n已知实体: {entities}" if entities else ""
        return (
            f"玩家问题: {query}{entity_hint}\n\n"
            "将上述问题扩展为3-5个不同的搜索查询，用于在游戏攻略数据库中检索。\n"
            "包括: 中文口语版、wiki标准术语版、英文版（如适用）。\n"
            "每行一个查询，不加编号。"
        )
