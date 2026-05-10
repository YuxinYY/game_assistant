"""
SynthesisAgent: writes the final answer.
Enforces: every claim must cite a source, contested items must be flagged,
unsupported claims must be admitted as "不确定".
"""

from pathlib import Path

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Citation


WORKFLOW_PROMPT_FILES = {
    "fact_lookup": "synthesis_fact_lookup.txt",
    "navigation": "synthesis_navigation.txt",
}


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    prompt_file = "synthesis_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return []  # synthesis is a single LLM call, no tool loop needed

    def execute(self, state: AgentState) -> AgentState:
        state.citations = _extract_citations(state.retrieved_docs)
        if not state.retrieved_docs:
            state.final_answer = _build_no_results_answer(state)
            self._trace(state, 0, "synthesis_no_results", "No retrieved docs available for synthesis.")
            return state

        prompt = self._load_prompt_for_workflow(state.workflow)
        context = _build_synthesis_context(state)
        messages = [{"role": "user", "content": context}]

        try:
            answer = self.llm.complete(messages, system=prompt).strip()
            if not answer:
                raise ValueError("LLM returned an empty synthesis response")
            state.final_answer = _append_citation_block(answer, state.citations)
            self._trace(state, 0, "synthesis_llm", f"Generated answer from {len(state.retrieved_docs)} docs.")
        except Exception as exc:
            state.final_answer = _build_extract_fallback_answer(state, error=str(exc))
            self._trace(state, 0, "synthesis_fallback", str(exc))
        return state

    def _load_prompt_for_workflow(self, workflow: str | None) -> str:
        prompt_file = WORKFLOW_PROMPT_FILES.get(workflow, self.prompt_file)
        path = Path(__file__).parent.parent / "llm" / "prompts" / prompt_file
        return path.read_text(encoding="utf-8") if path.exists() else self._load_prompt()


def _build_synthesis_context(state: AgentState) -> str:
    docs_text = "\n\n".join(
        f"[来源 {i+1}] {d.source} | {d.url}\n{d.text}"
        for i, d in enumerate(state.retrieved_docs)
    )
    consensus_text = ""
    if state.consensus_analysis:
        strategies = state.consensus_analysis.get("strategies", [])
        consensus_text = "\n".join(
            f"- {s['label']}: {s['source_count']} 个来源支持"
            + (" ⚠️ 存在争议" if s.get("is_contested") else "")
            for s in strategies
        )
        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            consensus_text += "\n\n争议点:\n" + "\n".join(
                f"- {c['topic']}: 支持({len(c['pro'])}) vs 反对({len(c['con'])})"
                for c in conflicts
            )

    return (
        f"当前 workflow:\n{state.workflow or 'unknown'}\n\n"
        f"玩家状态:\n{state.player_profile.to_context_string()}\n\n"
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


def _build_no_results_answer(state: AgentState) -> str:
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


def _build_extract_fallback_answer(state: AgentState, error: str = "") -> str:
    if state.workflow == "fact_lookup":
        return _build_fact_lookup_fallback_answer(state, error)

    if state.workflow == "navigation":
        return _build_navigation_fallback_answer(state, error)

    sections = ["## 基于检索结果的直接整理"]

    entities = state.identified_entities or [doc.entity for doc in state.retrieved_docs if doc.entity]
    unique_entities = []
    for entity in entities:
        if entity and entity not in unique_entities:
            unique_entities.append(entity)
    if unique_entities:
        sections.append(f"- 当前检索主要围绕：{'、'.join(unique_entities)}。")

    sections.append("\n## 针对你的 build 的建议")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            f"- {_to_brief_evidence(doc.text)} 来源: {doc.source} {_format_reference_link(doc.url)}"
        )

    sections.append("\n## 共识度")
    consensus_text = _build_consensus_lines(state)
    sections.extend(consensus_text or ["- 当前检索样本较少，无法得出稳定共识。"])

    sections.append("\n## 系统不确定的地方")
    if error:
        sections.append(f"- 本次使用了非 LLM 降级整理，因为生成阶段失败：{error}")
    conflicts = (state.consensus_analysis or {}).get("conflicts", [])
    if conflicts:
        for conflict in conflicts:
            sections.append(
                f"- {conflict['topic']} 存在争议：支持 {len(conflict['pro'])} 条，反对 {len(conflict['con'])} 条。"
            )
    else:
        sections.append("- 当前主要不确定性来自样本规模较小，后续补充更多来源会更稳。")

    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs))


def _build_fact_lookup_fallback_answer(state: AgentState, error: str = "") -> str:
    sections = ["## 直接结论"]
    sections.append(f"- {_to_brief_evidence(state.retrieved_docs[0].text, max_chars=140)}")

    sections.append("\n## 依据")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            f"- {doc.source}: {_to_brief_evidence(doc.text, max_chars=120)} {_format_reference_link(doc.url)}"
        )

    sections.append("\n## 系统不确定的地方")
    if error:
        sections.append(f"- 本次使用了降级整理，因为生成阶段失败：{error}")
    sections.append("- 当前回答基于检索到的事实片段整理，未做额外推断。")
    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs))


def _build_navigation_fallback_answer(state: AgentState, error: str = "") -> str:
    sections = ["## 位置结论"]
    sections.append(f"- {_to_brief_evidence(state.retrieved_docs[0].text, max_chars=140)}")

    sections.append("\n## 路线或前置条件")
    for doc in state.retrieved_docs[:3]:
        sections.append(
            f"- {doc.source}: {_to_brief_evidence(doc.text, max_chars=120)} {_format_reference_link(doc.url)}"
        )

    sections.append("\n## 系统不确定的地方")
    if error:
        sections.append(f"- 本次使用了降级整理，因为生成阶段失败：{error}")
    sections.append("- 如果目标存在多种译名或地图分支，可能还需要补充章节或区域名。")
    return _append_citation_block("\n".join(sections), _extract_citations(state.retrieved_docs))


def _build_consensus_lines(state: AgentState) -> list[str]:
    analysis = state.consensus_analysis or {}
    strategies = analysis.get("strategies", [])
    lines = []
    for strategy in strategies[:5]:
        line = f"- {strategy['label']}: {strategy['source_count']} 个来源支持"
        if strategy.get("is_contested"):
            line += " ⚠️ 存在争议"
        lines.append(line)
    return lines


def _append_citation_block(answer: str, citations: list[Citation]) -> str:
    if not citations:
        return answer
    citation_lines = ["\n## 参考来源"]
    for index, citation in enumerate(citations, start=1):
        author_suffix = f" | 作者: {citation.author}" if citation.author else ""
        citation_lines.append(f"- [{index}] {citation.source} | {_format_reference_link(citation.url)}{author_suffix}")
    return answer.rstrip() + "\n\n" + "\n".join(citation_lines)


def _format_reference_link(url: str) -> str:
    if not url:
        return "无原文链接"
    return f"[原文]({url})"


def _to_brief_evidence(text: str, max_chars: int = 70) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"
