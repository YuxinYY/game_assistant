"""
SynthesisAgent: writes the final answer.
Enforces: every claim must cite a source, contested items must be flagged,
unsupported claims must be admitted as "不确定".
"""

from pathlib import Path

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Citation
from src.utils.language import wants_english


WORKFLOW_PROMPT_FILES = {
    "zh": {
        "default": "synthesis_agent.txt",
        "fact_lookup": "synthesis_fact_lookup.txt",
        "navigation": "synthesis_navigation.txt",
    },
    "en": {
        "default": "synthesis_agent_en.txt",
        "fact_lookup": "synthesis_fact_lookup_en.txt",
        "navigation": "synthesis_navigation_en.txt",
    },
}


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    prompt_file = "synthesis_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return []  # synthesis is a single LLM call, no tool loop needed

    def execute(self, state: AgentState) -> AgentState:
        language = "en" if wants_english(state.user_query) else "zh"
        state.citations = _extract_citations(state.retrieved_docs)
        if not state.retrieved_docs:
            state.final_answer = _build_no_results_answer(state, language)
            self._trace(state, 0, "synthesis_no_results", "No retrieved docs available for synthesis.")
            return state

        prompt = self._load_prompt_for_workflow(state.workflow, language)
        context = _build_synthesis_context(state, language)
        messages = [{"role": "user", "content": context}]

        try:
            answer = self.llm.complete(messages, system=prompt).strip()
            if not answer:
                raise ValueError("LLM returned an empty synthesis response")
            state.final_answer = _append_citation_block(answer, state.citations, language)
            self._trace(state, 0, "synthesis_llm", f"Generated answer from {len(state.retrieved_docs)} docs.")
        except Exception as exc:
            state.final_answer = _build_extract_fallback_answer(state, language, error=str(exc))
            self._trace(state, 0, "synthesis_fallback", str(exc))
        return state

    def _load_prompt_for_workflow(self, workflow: str | None, language: str) -> str:
        prompt_map = WORKFLOW_PROMPT_FILES.get(language, WORKFLOW_PROMPT_FILES["zh"])
        prompt_file = prompt_map.get(workflow or "", prompt_map["default"])
        path = Path(__file__).parent.parent / "llm" / "prompts" / prompt_file
        return path.read_text(encoding="utf-8") if path.exists() else self._load_prompt()


def _build_synthesis_context(state: AgentState, language: str) -> str:
    docs_text = "\n\n".join(
        (
            f"[Source {i+1}] {d.source} | {d.url}\n{d.text}"
            if language == "en"
            else f"[来源 {i+1}] {d.source} | {d.url}\n{d.text}"
        )
        for i, d in enumerate(state.retrieved_docs)
    )
    consensus_text = ""
    if state.consensus_analysis:
        strategies = state.consensus_analysis.get("strategies", [])
        if language == "en":
            consensus_text = "\n".join(
                f"- {s['label']}: {s['source_count']} supporting source(s)"
                + (" ⚠️ contested" if s.get("is_contested") else "")
                for s in strategies
            )
        else:
            consensus_text = "\n".join(
                f"- {s['label']}: {s['source_count']} 个来源支持"
                + (" ⚠️ 存在争议" if s.get("is_contested") else "")
                for s in strategies
            )
        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            if language == "en":
                consensus_text += "\n\nConflicts:\n" + "\n".join(
                    f"- {c['topic']}: support({len(c['pro'])}) vs oppose({len(c['con'])})"
                    for c in conflicts
                )
            else:
                consensus_text += "\n\n争议点:\n" + "\n".join(
                    f"- {c['topic']}: 支持({len(c['pro'])}) vs 反对({len(c['con'])})"
                    for c in conflicts
                )

    if language == "en":
        return (
            f"Current workflow:\n{state.workflow or 'unknown'}\n\n"
            f"Player profile:\n{state.player_profile.to_context_string(language='en')}\n\n"
            f"User question:\n{state.user_query}\n\n"
            f"Retrieved material:\n{docs_text}\n\n"
            f"Consensus:\n{consensus_text or 'None'}"
        )

    return (
        f"当前 workflow:\n{state.workflow or 'unknown'}\n\n"
        f"玩家状态:\n{state.player_profile.to_context_string(language='zh')}\n\n"
        f"用户问题:\n{state.user_query}\n\n"
        f"检索到的内容:\n{docs_text}\n\n"
        f"共识分析:\n{consensus_text or '无'}"
    )


def _extract_citations(docs) -> list[Citation]:
    seen = set()
    citations = []
    for doc in docs:
        if doc.url not in seen:
            seen.add(doc.url)
            citations.append(Citation(
                source=doc.source,
                url=doc.url,
                excerpt=doc.text[:120] + "…",
                author=doc.metadata.get("author", ""),
            ))
    return citations


def _build_no_results_answer(state: AgentState, language: str) -> str:
    if language == "en":
        if state.workflow == "fact_lookup":
            return (
                "## Direct Answer\n"
                "- The retrieved material does not contain enough evidence to answer this reliably yet.\n\n"
                "## Your Question\n"
                f"- {state.user_query}\n\n"
                "## Next Step\n"
                "- Add a more specific boss, mechanic, or move name and try again."
            )

        if state.workflow == "navigation":
            return (
                "## Location Answer\n"
                "- The retrieved material does not establish this location reliably yet.\n\n"
                "## Your Question\n"
                f"- {state.user_query}\n\n"
                "## Next Step\n"
                "- Add the area, NPC, item name, or chapter and try again."
            )

        return (
            "## Current Result\n"
            "- The retrieved material is not sufficient to produce a source-grounded guide answer yet.\n\n"
            "## Your Question\n"
            f"- {state.user_query}\n\n"
            "## Next Step\n"
            "- Add a clearer boss, move, location, or chapter reference and try again."
        )

    if state.workflow == "fact_lookup":
        return (
            "## 当前结论\n"
            "- 检索内容中未找到足够依据，暂时无法给出可靠的事实回答。\n\n"
            "## 你的问题\n"
            f"- {state.user_query}\n\n"
            "## 建议\n"
            "- 请补充更明确的实体名、机制名或关键词后再试。"
        )

    if state.workflow == "navigation":
        return (
            "## 当前位置结论\n"
            "- 检索内容中未找到足够依据，暂时无法确认目标位置。\n\n"
            "## 你的问题\n"
            f"- {state.user_query}\n\n"
            "## 建议\n"
            "- 请补充地点、NPC、道具名或章节信息后再试。"
        )

    return (
        "## 当前结果\n"
        "- 检索内容中未找到足够资料，暂时无法生成有依据的攻略回答。\n\n"
        "## 你的问题\n"
        f"- {state.user_query}\n\n"
        "## 建议\n"
        "- 请先补充更明确的 boss、招式、地点或当前章节信息后再试。"
    )


def _build_extract_fallback_answer(state: AgentState, language: str, error: str = "") -> str:
    if state.workflow == "fact_lookup":
        return _build_fact_lookup_fallback_answer(state, language, error)

    if state.workflow == "navigation":
        return _build_navigation_fallback_answer(state, language, error)

    sections = ["## Retrieved Evidence Summary" if language == "en" else "## 基于检索结果的直接整理"]

    entities = state.identified_entities or [doc.entity for doc in state.retrieved_docs if doc.entity]
    unique_entities = []
    for entity in entities:
        if entity and entity not in unique_entities:
            unique_entities.append(entity)
    if unique_entities:
        if language == "en":
            sections.append(f"- Current retrieval mainly covers: {', '.join(unique_entities)}.")
        else:
            sections.append(f"- 当前检索主要围绕：{'、'.join(unique_entities)}。")

    sections.append("\n## Recommendation For Your Build" if language == "en" else "\n## 针对你的 build 的建议")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            (
                f"- {_to_brief_evidence(doc.text)} Source: {doc.source} {_format_reference_link(doc.url, language)}"
                if language == "en"
                else f"- {_to_brief_evidence(doc.text)} 来源: {doc.source} {_format_reference_link(doc.url, language)}"
            )
        )

    sections.append("\n## Consensus" if language == "en" else "\n## 共识度")
    consensus_text = _build_consensus_lines(state, language)
    sections.extend(consensus_text or (["- The current evidence sample is too small to establish a stable consensus."] if language == "en" else ["- 当前检索样本较少，无法得出稳定共识。"]))

    sections.append("\n## Uncertainty" if language == "en" else "\n## 系统不确定的地方")
    if error:
        if language == "en":
            sections.append(f"- This answer used the extractive fallback because the generation step failed: {error}")
        else:
            sections.append(f"- 本次使用了非 LLM 降级整理，因为生成阶段失败：{error}")
    conflicts = (state.consensus_analysis or {}).get("conflicts", [])
    if conflicts:
        for conflict in conflicts:
            if language == "en":
                sections.append(
                    f"- {conflict['topic']} remains disputed: support {len(conflict['pro'])} vs oppose {len(conflict['con'])}."
                )
            else:
                sections.append(
                    f"- {conflict['topic']} 存在争议：支持 {len(conflict['pro'])} 条，反对 {len(conflict['con'])} 条。"
                )
    else:
        sections.append(
            "- The main uncertainty comes from limited evidence volume; more sources would make the answer more stable."
            if language == "en"
            else "- 当前主要不确定性来自样本规模较小，后续补充更多来源会更稳。"
        )

    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs), language)


def _build_fact_lookup_fallback_answer(state: AgentState, language: str, error: str = "") -> str:
    sections = ["## Direct Answer" if language == "en" else "## 直接结论"]
    sections.append(f"- {_to_brief_evidence(state.retrieved_docs[0].text, max_chars=140)}")

    sections.append("\n## Evidence" if language == "en" else "\n## 依据")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            f"- {doc.source}: {_to_brief_evidence(doc.text, max_chars=120)} {_format_reference_link(doc.url, language)}"
        )

    sections.append("\n## Uncertainty" if language == "en" else "\n## 系统不确定的地方")
    if error:
        if language == "en":
            sections.append(f"- This answer used the fallback formatter because the generation step failed: {error}")
        else:
            sections.append(f"- 本次使用了降级整理，因为生成阶段失败：{error}")
    sections.append(
        "- This answer only reorganizes retrieved fact snippets and avoids extra inference."
        if language == "en"
        else "- 当前回答基于检索到的事实片段整理，未做额外推断。"
    )
    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs), language)


def _build_navigation_fallback_answer(state: AgentState, language: str, error: str = "") -> str:
    sections = ["## Location Answer" if language == "en" else "## 位置结论"]
    sections.append(f"- {_to_brief_evidence(state.retrieved_docs[0].text, max_chars=140)}")

    sections.append("\n## Route Or Requirements" if language == "en" else "\n## 路线或前置条件")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            f"- {doc.source}: {_to_brief_evidence(doc.text, max_chars=120)} {_format_reference_link(doc.url, language)}"
        )

    sections.append("\n## Uncertainty" if language == "en" else "\n## 系统不确定的地方")
    if error:
        if language == "en":
            sections.append(f"- This answer used the fallback formatter because the generation step failed: {error}")
        else:
            sections.append(f"- 本次使用了降级整理，因为生成阶段失败：{error}")
    sections.append(
        "- If the target has multiple names or map branches, add the chapter or area name for a tighter answer."
        if language == "en"
        else "- 如果目标存在多种译名或地图分支，可能还需要补充章节或区域名。"
    )
    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs), language)


def _build_consensus_lines(state: AgentState, language: str) -> list[str]:
    analysis = state.consensus_analysis or {}
    strategies = analysis.get("strategies", [])
    lines = []
    for strategy in strategies[:5]:
        if language == "en":
            line = f"- {strategy['label']}: {strategy['source_count']} supporting source(s)"
            if strategy.get("is_contested"):
                line += " ⚠️ contested"
        else:
            line = f"- {strategy['label']}: {strategy['source_count']} 个来源支持"
            if strategy.get("is_contested"):
                line += " ⚠️ 存在争议"
        lines.append(line)
    return lines


def _append_citation_block(answer: str, citations: list[Citation], language: str) -> str:
    if not citations:
        return answer
    citation_lines = ["\n## Sources" if language == "en" else "\n## 参考来源"]
    for index, citation in enumerate(citations, start=1):
        author_suffix = (
            f" | Author: {citation.author}"
            if citation.author and language == "en"
            else f" | 作者: {citation.author}"
            if citation.author
            else ""
        )
        citation_lines.append(f"- [{index}] {citation.source} | {_format_reference_link(citation.url, language)}{author_suffix}")
    return answer.rstrip() + "\n\n" + "\n".join(citation_lines)


def _format_reference_link(url: str, language: str) -> str:
    if not url:
        return "No original URL" if language == "en" else "无原文链接"
    return f"[original source]({url})" if language == "en" else f"[原文]({url})"


def _to_brief_evidence(text: str, max_chars: int = 70) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"
