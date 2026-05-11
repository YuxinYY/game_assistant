"""
WikiAgent: resolves move names and game facts from the official wiki corpus.
Responsible for: "这是什么招式" → structured entity identification.
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Document
from src.tools.search import wiki_search, entity_lookup, infer_wiki_entity


class _WikiSearch(Tool):
    name = "wiki_search"
    description = "在 bwiki 数据中搜索 boss 招式、属性、机制"
    description_en = "Search boss moves, stats, and mechanics in the BWIKI data."

    def __call__(self, query: str, entity: str = "") -> list:
        return wiki_search(query, entity_filter=entity)


class _EntityLookup(Tool):
    name = "entity_lookup"
    description = "精确查找某个实体（boss名、招式名）的 wiki 条目"
    description_en = "Look up the exact wiki entry for an entity such as a boss or move name."

    def __call__(self, entity: str, query: str = "") -> dict:
        return entity_lookup(entity, query=query)


class WikiAgent(BaseAgent):
    name = "wiki_agent"
    prompt_file = "wiki_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_WikiSearch(), _EntityLookup()]

    def execute(self, state: AgentState) -> AgentState:
        self._did_wiki_search = False
        self._looked_up_entities: set[str] = set()
        self._query_entity = infer_wiki_entity(state.user_query)
        if self._query_entity:
            state.identified_entities = _merge_entities(state.identified_entities, [self._query_entity])
        initial_context = self._localize(
            state,
            "目标: 识别用户问题里的 boss/招式实体，并补充 wiki 依据。\n优先用 wiki_search 找候选文档；只有在已经识别到实体但需要精确条目时才用 entity_lookup。",
            "Goal: identify the boss or move entity mentioned in the user's question and add official wiki evidence. Use wiki_search first for candidate passages; only call entity_lookup when an entity is already known and an exact entry is needed.",
        )
        return self.react_loop(state, initial_context)

    def _decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        primary_entity = self._primary_entity(state)
        wiki_docs = [doc for doc in state.retrieved_docs if doc.source == "wiki"]

        if not self._did_wiki_search:
            return self._localize(
                state,
                "先搜索 wiki 候选文档",
                "Search the wiki candidate passages first.",
            ), "wiki_search", {"query": state.user_query, "entity": primary_entity}

        if not wiki_docs and primary_entity and primary_entity not in self._looked_up_entities:
            return self._localize(
                state,
                "候选搜索为空，改用实体精确查找",
                "The candidate search returned nothing, so switch to an exact entity lookup.",
            ), "entity_lookup", {"entity": primary_entity, "query": state.user_query}

        return super()._decide(context, state)

    def _apply_tool_result(self, state: AgentState, action_name: str, action_args: dict, tool_result) -> None:
        if action_name == "wiki_search":
            docs = tool_result if isinstance(tool_result, list) else []
            entity_hint = action_args.get("entity")
            if entity_hint:
                state.identified_entities = _merge_entities(state.identified_entities, [entity_hint])
            state.retrieved_docs = _merge_docs(state.retrieved_docs, docs)
            entities = _extract_entities(docs)
            state.identified_entities = _merge_entities(state.identified_entities, entities)
            self._did_wiki_search = True
            return

        if action_name == "entity_lookup" and isinstance(tool_result, dict):
            entity = tool_result.get("entity") or action_args.get("entity")
            if entity:
                self._looked_up_entities.add(entity)
                state.identified_entities = _merge_entities(state.identified_entities, [entity])
            if tool_result.get("text") and tool_result.get("url"):
                doc = Document(
                    text=tool_result["text"],
                    source="wiki",
                    url=tool_result["url"],
                    entity=entity,
                    metadata={"author": "wiki_exact_lookup"},
                )
                state.retrieved_docs = _merge_docs(state.retrieved_docs, [doc])

    def _fallback_decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        if not self._did_wiki_search:
            entity = self._primary_entity(state)
            return self._localize(
                state,
                "先搜索 wiki 候选文档",
                "Search the wiki candidate passages first.",
            ), "wiki_search", {"query": state.user_query, "entity": entity}

        if (
            self._primary_entity(state)
            and self._primary_entity(state) not in self._looked_up_entities
            and not any(
                doc.source == "wiki" and doc.entity == self._primary_entity(state)
                for doc in state.retrieved_docs
            )
        ):
            return self._localize(
                state,
                "补一次实体精确查找",
                "Add one exact entity lookup pass.",
            ), "entity_lookup", {"entity": self._primary_entity(state), "query": state.user_query}

        return self._localize(
            state,
            "已有足够 wiki 证据",
            "Enough wiki evidence has already been collected.",
        ), "FINISH", {}

    def _primary_entity(self, state: AgentState) -> str:
        if self._query_entity:
            return self._query_entity
        if len(state.identified_entities) == 1:
            return state.identified_entities[0]
        return ""


def _extract_entities(docs) -> list[str]:
    entities = []
    for doc in docs:
        if doc.entity and doc.entity not in entities:
            entities.append(doc.entity)
    return entities


def _merge_entities(existing: list[str], new_entities: list[str]) -> list[str]:
    merged = list(existing)
    for entity in new_entities:
        if entity not in merged:
            merged.append(entity)
    return merged


def _merge_docs(existing, new_docs):
    merged = list(existing)
    seen = {(doc.url, doc.text) for doc in existing}
    for doc in new_docs:
        key = (doc.url, doc.text)
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged
