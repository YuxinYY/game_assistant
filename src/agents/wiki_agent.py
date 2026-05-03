"""
WikiAgent: resolves move names and game facts from the official wiki corpus.
Responsible for: "这是什么招式" → structured entity identification.
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState
from src.tools.search import wiki_search, entity_lookup


class _WikiSearch(Tool):
    name = "wiki_search"
    description = "在 bwiki 数据中搜索 boss 招式、属性、机制"

    def __call__(self, query: str, entity: str = "") -> list:
        return wiki_search(query, entity_filter=entity)


class _EntityLookup(Tool):
    name = "entity_lookup"
    description = "精确查找某个实体（boss名、招式名）的 wiki 条目"

    def __call__(self, entity: str) -> dict:
        return entity_lookup(entity)


class WikiAgent(BaseAgent):
    name = "wiki_agent"
    prompt_file = "wiki_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_WikiSearch(), _EntityLookup()]

    def execute(self, state: AgentState) -> AgentState:
        """
        1. Rewrite query into wiki-friendly terms (move name, boss name)
        2. ReAct: search wiki until move is identified
        3. Write identified entities + wiki docs into state
        """
        context = (
            f"用户问题: {state.user_query}\n"
            f"玩家状态: {state.player_profile.to_context_string()}\n"
            f"任务: 识别用户描述的游戏实体（boss名、招式名），从 wiki 获取准确信息"
        )
        state = self.react_loop(state, context)
        return state
