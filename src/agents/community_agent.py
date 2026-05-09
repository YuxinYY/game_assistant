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
        return nga_search(query, chapter_filter=chapter_filter)


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

    def __init__(self):
        self.rewriter = QueryRewriter()

    def __call__(self, query: str, entities: list[str] | None = None) -> list[str]:
        return self.rewriter.rewrite(query, known_entities=entities or [])


class CommunityAgent(BaseAgent):
    name = "community_agent"
    prompt_file = "community_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_NGASearch(), _BilibiliSearch(), _RedditSearch(), _QueryRewrite()]

    def execute(self, state: AgentState) -> AgentState:
        queries = QueryRewriter().rewrite(
            state.user_query,
            known_entities=state.identified_entities,
        )

        retrieved_docs = []
        for query in queries:
            docs = nga_search(query, chapter_filter=state.player_profile.chapter)
            if not docs:
                docs = nga_search(query)
            retrieved_docs = _merge_docs(retrieved_docs, docs)

        state.retrieved_docs = _merge_docs(state.retrieved_docs, retrieved_docs)
        self._trace(
            state,
            0,
            "deterministic_community_search",
            str(
                {
                    "queries": queries,
                    "source": "nga",
                    "doc_count": len(retrieved_docs),
                }
            ),
        )
        return state


def _merge_docs(existing, new_docs):
    merged = list(existing)
    seen = {(doc.url, doc.text) for doc in existing}
    for doc in new_docs:
        key = (doc.url, doc.text)
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged
