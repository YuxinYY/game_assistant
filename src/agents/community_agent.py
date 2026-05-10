"""
CommunityAgent: retrieves real player tactics from NGA, Bilibili, Reddit.
Key feature: query rewriting — turns "那个连砍蓄力的招" into wiki terminology before searching.
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState
from src.tools.search import nga_search, bilibili_search, reddit_search
from src.retrieval.query_rewriter import QueryRewriter


class _NGASearch(Tool):
    name = "nga_search"
    description = "在 NGA 论坛数据中搜索玩家攻略帖"

    def __call__(self, query: str, chapter_filter: int | None = None) -> list:
        docs = nga_search(query, chapter_filter=chapter_filter)
        if docs or chapter_filter is None:
            return docs
        return nga_search(query)


class _BilibiliSearch(Tool):
    name = "bilibili_search"
    description = "在 B站评论/字幕数据中搜索玩家经验"

    def __call__(self, query: str) -> list:
        return bilibili_search(query)


class _RedditSearch(Tool):
    name = "reddit_search"
    description = "在 Reddit r/BlackMythWukong 数据中搜索英语攻略"

    def __call__(self, query: str) -> list:
        return reddit_search(query)


class _QueryRewrite(Tool):
    name = "query_rewrite"
    description = "将模糊描述改写为 wiki 标准术语，提高检索召回率"

    def __init__(self, llm_client=None):
        self.rewriter = QueryRewriter(llm_client)

    def __call__(self, query: str, entities: list[str] | None = None) -> list[str]:
        return self.rewriter.rewrite(query, known_entities=entities or [])


class CommunityAgent(BaseAgent):
    name = "community_agent"
    prompt_file = "community_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_NGASearch(), _BilibiliSearch(), _RedditSearch(), _QueryRewrite(self.llm)]

    def execute(self, state: AgentState) -> AgentState:
        self._rewritten_queries: list[str] = []
        self._searched_sources: list[str] = []
        self.max_iterations = max(self.max_iterations, 4)
        initial_context = (
            "目标: 先把用户问题改写成更适合检索的查询，再到社区来源搜实战经验。\n"
            "优先搜索 NGA；如证据不足，再补 Bilibili 和 Reddit。"
        )
        return self.react_loop(state, initial_context)

    def _apply_tool_result(self, state: AgentState, action_name: str, action_args: dict, tool_result) -> None:
        if action_name == "query_rewrite":
            if isinstance(tool_result, list):
                self._rewritten_queries = [query for query in tool_result if isinstance(query, str) and query.strip()]
            if not self._rewritten_queries:
                self._rewritten_queries = [state.user_query]
            return

        if action_name in {"nga_search", "bilibili_search", "reddit_search"}:
            docs = tool_result if isinstance(tool_result, list) else []
            state.retrieved_docs = _merge_docs(state.retrieved_docs, docs)
            self._searched_sources.append(action_name.replace("_search", ""))

    def _fallback_decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        if not self._rewritten_queries:
            return (
                "先改写查询，补齐标准术语和英文表达",
                "query_rewrite",
                {"query": state.user_query, "entities": state.identified_entities},
            )

        if "nga" not in self._searched_sources:
            return (
                "优先搜索 NGA 社区经验",
                "nga_search",
                {"query": self._pick_query("nga"), "chapter_filter": state.player_profile.chapter},
            )

        if "bilibili" not in self._searched_sources:
            return "补搜 Bilibili 玩家经验", "bilibili_search", {"query": self._pick_query("bilibili")}

        if "reddit" not in self._searched_sources:
            return "补搜 Reddit 英文经验", "reddit_search", {"query": self._pick_query("reddit")}

        return "社区检索已完成", "FINISH", {}

    def _pick_query(self, source: str) -> str:
        if not self._rewritten_queries:
            return ""
        if source == "reddit":
            for query in self._rewritten_queries:
                if any("a" <= char.lower() <= "z" for char in query):
                    return query
        return self._rewritten_queries[0]


def _merge_docs(existing, new_docs):
    merged = list(existing)
    seen = {(doc.url, doc.text) for doc in existing}
    for doc in new_docs:
        key = (doc.url, doc.text)
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged
