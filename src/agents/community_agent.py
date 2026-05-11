"""
CommunityAgent: retrieves real player tactics from NGA, Bilibili, Reddit.
Key feature: query rewriting — turns "那个连砍蓄力的招" into wiki terminology before searching.
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState
from src.tools.search import nga_search, bilibili_search, reddit_search, has_indexed_source_documents
from src.retrieval.query_rewriter import QueryRewriter
from src.utils.language import detect_query_language


class _NGASearch(Tool):
    name = "nga_search"
    description = "在 NGA 论坛数据中搜索玩家攻略帖"
    description_en = "Search player guide posts in the NGA forum data."

    def __call__(self, query: str, chapter_filter: int | None = None) -> list:
        docs = nga_search(query, chapter_filter=chapter_filter)
        if docs or chapter_filter is None:
            return docs
        return nga_search(query)


class _BilibiliSearch(Tool):
    name = "bilibili_search"
    description = "在 B站评论/字幕数据中搜索玩家经验"
    description_en = "Search player experience from Bilibili comments and subtitles."

    def __call__(self, query: str) -> list:
        return bilibili_search(query)


class _RedditSearch(Tool):
    name = "reddit_search"
    description = "在 Reddit r/BlackMythWukong 数据中搜索英语攻略"
    description_en = "Search English strategy posts in Reddit r/BlackMythWukong data."

    def __call__(self, query: str) -> list:
        return reddit_search(query)


class _QueryRewrite(Tool):
    name = "query_rewrite"
    description = "将模糊描述改写为 wiki 标准术语，提高检索召回率"
    description_en = "Rewrite vague descriptions into wiki-standard terms to improve retrieval recall."

    def __init__(self, llm_client=None):
        self.rewriter = QueryRewriter(llm_client)

    def __call__(self, query: str, entities: list[str] | None = None) -> list[str]:
        return self.rewriter.rewrite(query, known_entities=entities or [])


class CommunityAgent(BaseAgent):
    name = "community_agent"
    prompt_file = "community_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [
            _NGASearch(),
            _BilibiliSearch(),
            _RedditSearch(),
            _QueryRewrite(self.llm),
        ]

    def execute(self, state: AgentState) -> AgentState:
        self._rewritten_queries: list[str] = []
        self._searched_sources: list[str] = []
        self._searched_pairs: set[tuple[str, str]] = set()
        self._source_hits: set[str] = set()
        self._source_queries: dict[str, list[str]] = {}
        self._query_language = detect_query_language(state.user_query)
        self._source_plan = self._build_source_plan()
        self._search_goals = self._resolve_search_goals(state)
        self.max_iterations = max(self.max_iterations, min(7, len(self._source_plan) * 2 + 2))
        if not self._source_plan:
            self._trace(
                state,
                0,
                "community_sources_unavailable",
                self._localize(
                    state,
                    f"当前问题语言 '{self._query_language or 'unknown'}' 没有可用的已索引社区来源。",
                    f"No indexed community sources are available for query language '{self._query_language or 'unknown'}'.",
                ),
            )
            return state
        initial_context = self._localize(
            state,
            f"目标: 先把用户问题改写成更适合检索的查询，再到社区来源搜实战经验。\n本次社区来源计划: {', '.join(self._source_plan)}。",
            f"Goal: rewrite the user's question into more retrieval-friendly queries and then search community sources for grounded player tactics. Community source plan for this run: {', '.join(self._source_plan)}.",
        )
        state = self.react_loop(state, initial_context)
        if not any(doc.source in {"nga", "bilibili", "reddit"} for doc in state.retrieved_docs):
            self._trace(
                state,
                len(state.trace),
                "community_docs_not_found",
                self._localize(
                    state,
                    "未检索到社区文档；下游回答将退回到仅使用 wiki 证据。",
                    "No community documents were retrieved; downstream answer will fall back to wiki-only evidence.",
                ),
            )
        return state

    def _apply_tool_result(self, state: AgentState, action_name: str, action_args: dict, tool_result) -> None:
        if action_name == "query_rewrite":
            if isinstance(tool_result, list):
                self._rewritten_queries = [query for query in tool_result if isinstance(query, str) and query.strip()]
            if not self._rewritten_queries:
                self._rewritten_queries = [state.user_query]
            self._source_queries = self._build_source_queries(state)
            return

        if action_name in {"nga_search", "bilibili_search", "reddit_search"}:
            docs = tool_result if isinstance(tool_result, list) else []
            state.retrieved_docs = _merge_docs(state.retrieved_docs, docs)
            source_name = action_name.replace("_search", "")
            if source_name not in self._searched_sources:
                self._searched_sources.append(source_name)
            if docs:
                self._source_hits.add(source_name)

    def _fallback_decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        if not self._rewritten_queries:
            return (
                self._localize(
                    state,
                    "先改写查询，补齐标准术语和英文表达",
                    "Rewrite the query first to add standard terms and better search phrasing.",
                ),
                "query_rewrite",
                {"query": state.user_query, "entities": state.identified_entities},
            )

        next_task = self._next_search_task(state)
        if next_task is not None:
            source, query = next_task
            goal_summary = self._goal_summary(state)
            if source == "nga":
                return (
                    self._localize(
                        state,
                        f"搜索 NGA 社区经验，当前证据目标: {goal_summary}",
                        f"Search NGA for community tactics. Current evidence target: {goal_summary}",
                    ),
                    "nga_search",
                    {"query": query, "chapter_filter": state.player_profile.chapter},
                )
            if source == "bilibili":
                return (
                    self._localize(
                        state,
                        f"搜索 Bilibili 玩家经验，当前证据目标: {goal_summary}",
                        f"Search Bilibili for player experience. Current evidence target: {goal_summary}",
                    ),
                    "bilibili_search",
                    {"query": query},
                )
            if source == "reddit":
                return (
                    self._localize(
                        state,
                        f"搜索 Reddit 英文经验，当前证据目标: {goal_summary}",
                        f"Search Reddit for English player strategy. Current evidence target: {goal_summary}",
                    ),
                    "reddit_search",
                    {"query": query},
                )

        return self._localize(
            state,
            "社区检索已完成",
            "Community retrieval is complete.",
        ), "FINISH", {}

    def _build_source_plan(self) -> list[str]:
        if self._query_language == "en":
            return [
                source
                for source in ["reddit", "nga", "bilibili"]
                if has_indexed_source_documents(source, language="en")
            ]

        source_plan: list[str] = []
        if has_indexed_source_documents("nga"):
            source_plan.append("nga")
        if has_indexed_source_documents("bilibili"):
            source_plan.append("bilibili")
        if has_indexed_source_documents("reddit", language="en"):
            source_plan.append("reddit")
        return source_plan

    def _resolve_search_goals(self, state: AgentState) -> list[str]:
        if state.execution_plan and state.execution_plan.goals:
            return list(state.execution_plan.goals)

        goals = ["collect_community_counterplay"]
        normalized_query = state.user_query.lower()
        if any(keyword in state.user_query for keyword in ["怎么躲", "闪避", "躲", "侧闪"]) or any(
            keyword in normalized_query for keyword in ["dodge", "avoid", "evade"]
        ):
            goals.append("resolve_dodge_timing")
        if any(keyword in state.user_query for keyword in ["怎么打", "反打", "后摇", "输出"]) or any(
            keyword in normalized_query for keyword in ["beat", "punish", "counter", "recovery"]
        ):
            goals.append("resolve_punish_window")
        if state.player_profile.build is not None or any(keyword in state.user_query for keyword in ["流派", "法术", "配装"]) or any(
            keyword in normalized_query for keyword in ["build", "spell", "armor", "skill"]
        ):
            goals.append("collect_build_specific_advice")
        return list(dict.fromkeys(goals))

    def _next_search_task(self, state: AgentState) -> tuple[str, str] | None:
        if not self._source_queries:
            self._source_queries = self._build_source_queries(state)

        for source in self._source_plan:
            if source in self._source_hits:
                continue
            for query in self._source_queries.get(source, []):
                key = (source, query)
                if key in self._searched_pairs:
                    continue
                self._searched_pairs.add(key)
                return source, query
        return None

    def _build_source_queries(self, state: AgentState) -> dict[str, list[str]]:
        base_queries = [query for query in self._rewritten_queries if query.strip()] or [state.user_query]
        goal_queries = self._build_goal_queries(state)
        merged_queries = list(dict.fromkeys(goal_queries + base_queries))

        return {
            source: self._filter_queries_for_source(source, merged_queries, state.user_query)
            for source in self._source_plan
        }

    def _build_goal_queries(self, state: AgentState) -> list[str]:
        entity = state.identified_entities[0] if state.identified_entities else ""
        anchor = entity or state.user_query
        queries: list[str] = []

        if "resolve_dodge_timing" in self._search_goals:
            if self._query_language == "en":
                queries.extend([
                    f"{anchor} dodge timing",
                    f"{anchor} avoid attack",
                ])
            else:
                queries.extend([
                    f"{anchor} 躲避 时机",
                    f"{anchor} 闪避 处理",
                ])

        if "resolve_punish_window" in self._search_goals:
            if self._query_language == "en":
                queries.extend([
                    f"{anchor} punish window",
                    f"{anchor} recovery punish",
                ])
            else:
                queries.extend([
                    f"{anchor} 反打 时机",
                    f"{anchor} 后摇 输出",
                ])

        if "collect_build_specific_advice" in self._search_goals:
            build_hint = state.player_profile.build or ""
            if self._query_language == "en":
                queries.extend([
                    f"{anchor} {build_hint} build advice".strip(),
                    f"{anchor} spell choice".strip(),
                ])
            else:
                queries.extend([
                    f"{anchor} {build_hint} 配装 建议".strip(),
                    f"{anchor} 法术 选择".strip(),
                ])

        if "compare_build_options" in self._search_goals:
            if self._query_language == "en":
                queries.append(f"{anchor} build comparison")
            else:
                queries.append(f"{anchor} 流派 对比")

        return [query.strip() for query in queries if query.strip()]

    def _filter_queries_for_source(self, source: str, queries: list[str], original_query: str) -> list[str]:
        if source == "reddit":
            english_queries = [query for query in queries if any("a" <= char.lower() <= "z" for char in query)]
            return english_queries[:2] or queries[:2] or [original_query]
        return queries[:2] or [original_query]

    def _goal_summary(self, state: AgentState) -> str:
        goals = [self._goal_label(goal, state) for goal in self._search_goals[:2]]
        if goals:
            return ", ".join(goals)
        return self._localize(state, "通用打法", "general strategy")

    def _goal_label(self, goal: str, state: AgentState) -> str:
        labels = {
            "collect_community_counterplay": self._localize(state, "社区应对经验", "community counterplay"),
            "resolve_dodge_timing": self._localize(state, "闪避时机", "dodge timing"),
            "resolve_punish_window": self._localize(state, "反打窗口", "punish window"),
            "collect_build_specific_advice": self._localize(state, "流派相关建议", "build-specific advice"),
            "compare_build_options": self._localize(state, "流派对比", "build comparison"),
        }
        return labels.get(goal, goal)


def _merge_docs(existing, new_docs):
    merged = list(existing)
    seen = {(doc.url, doc.text) for doc in existing}
    for doc in new_docs:
        key = (doc.url, doc.text)
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged
