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
        """
        1. Rewrite query using identified wiki entities from WikiAgent
        2. ReAct: search community sources with chapter_filter = player chapter
        3. Append community docs to state.retrieved_docs
        """
        entity_hint = ", ".join(state.identified_entities) if state.identified_entities else ""
        context = (
            f"用户问题: {state.user_query}\n"
            f"已识别实体: {entity_hint}\n"
            f"玩家当前章节: {state.player_profile.chapter}\n"
            f"任务: 在玩家社区数据中搜索针对上述实体的实战攻略，优先搜索章节≤{state.player_profile.chapter}的内容"
        )
        state = self.react_loop(state, context)
        return state
