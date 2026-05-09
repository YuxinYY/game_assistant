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
        docs = wiki_search(state.user_query)
        state.retrieved_docs = _merge_docs(state.retrieved_docs, docs)

        entities = _extract_entities(docs)
        state.identified_entities = _merge_entities(state.identified_entities, entities)

        self._trace(
            state,
            0,
            "deterministic_wiki_search",
            str(
                {
                    "query": state.user_query,
                    "doc_count": len(docs),
                    "entities": entities,
                }
            ),
        )
        return state


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
