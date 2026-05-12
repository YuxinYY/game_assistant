"""
SynthesisAgent: writes the final answer.
Enforces: every claim must cite a source, contested items must be flagged,
unsupported claims must be admitted as "不确定".
"""

from pathlib import Path
from functools import lru_cache
import re

from src.agents.base_agent import BaseAgent, Tool
from src.agents.profile_agent import CHAPTER_GATES
from src.core.state import AgentState, Citation
from src.llm.client import LLMClient
from src.tools.profile_ops import load_knowledge_base
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

_EXCERPT_SENTENCE_BREAK_PATTERN = re.compile(r"[。！？.!?]")

_ENGLISH_NAMED_OPTIONS = {
    "Immobilize": ("spell", "定身术"),
    "Rock Solid": ("spell", "铜头铁臂"),
    "Ring of Fire": ("spell", "安身法"),
    "Cloud Step": ("spell", "聚形散气"),
    "Pluck of Many": ("spell", "身外身法"),
    "Spell Binder": ("spell", "禁字法"),
    "Red Tides": ("transformation", "Red Tides"),
    "Azure Dust": ("transformation", "Azure Dust"),
    "Ashen Slumber": ("transformation", "Ashen Slumber"),
    "Hoarfrost": ("transformation", "Hoarfrost"),
    "Umbral Abyss": ("transformation", "Umbral Abyss"),
    "Golden Lining": ("transformation", "Golden Lining"),
    "Violet Hail": ("transformation", "Violet Hail"),
    "Ebon Flow": ("transformation", "Ebon Flow"),
    "Dark Thunder": ("transformation", "Dark Thunder"),
}


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    prompt_file = "synthesis_agent.txt"

    def __init__(self, config: dict):
        super().__init__(config)
        self._fallback_llm = None
        self._fallback_llm_initialized = False

    def _register_tools(self) -> list[Tool]:
        return []  # synthesis is a single LLM call, no tool loop needed

    def execute(self, state: AgentState) -> AgentState:
        language = "en" if wants_english(state.user_query) else "zh"
        state.citations = _extract_citations(state.retrieved_docs)
        if not state.retrieved_docs:
            state.final_answer = _build_no_results_answer(state, language)
            state.answer_confidence = 0.0
            state.stop_reason = "insufficient_evidence"
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
            state.stop_reason = "answered"
        except Exception as primary_exc:
            fallback_llm = self._get_fallback_llm()
            if fallback_llm is not None:
                try:
                    answer = fallback_llm.complete(messages, system=prompt).strip()
                    if not answer:
                        raise ValueError("Fallback LLM returned an empty synthesis response")
                    state.final_answer = _append_citation_block(answer, state.citations, language)
                    self._trace(
                        state,
                        0,
                        "synthesis_secondary_llm",
                        (
                            f"Primary provider {getattr(self.llm, 'provider', 'unknown')} failed: {primary_exc}; "
                            f"generated answer using secondary provider {fallback_llm.provider}."
                        ),
                    )
                    state.stop_reason = "answered_via_secondary_provider"
                except Exception as secondary_exc:
                    combined_error = (
                        f"primary {getattr(self.llm, 'provider', 'unknown')} failed: {primary_exc}; "
                        f"secondary {fallback_llm.provider} failed: {secondary_exc}"
                    )
                    state.final_answer = _build_extract_fallback_answer(state, language, error=combined_error)
                    self._trace(state, 0, "synthesis_fallback", combined_error)
                    state.stop_reason = _classify_generation_failure(secondary_exc)
            else:
                state.final_answer = _build_extract_fallback_answer(state, language, error=str(primary_exc))
                self._trace(state, 0, "synthesis_fallback", str(primary_exc))
                state.stop_reason = _classify_generation_failure(primary_exc)

        state.answer_confidence = _estimate_answer_confidence(state)
        return state

    def _load_prompt_for_workflow(self, workflow: str | None, language: str) -> str:
        prompt_map = WORKFLOW_PROMPT_FILES.get(language, WORKFLOW_PROMPT_FILES["zh"])
        prompt_file = prompt_map.get(workflow or "", prompt_map["default"])
        path = Path(__file__).parent.parent / "llm" / "prompts" / prompt_file
        return path.read_text(encoding="utf-8") if path.exists() else self._load_prompt()

    def _get_fallback_llm(self):
        if self._fallback_llm_initialized:
            return self._fallback_llm

        self._fallback_llm_initialized = True
        primary_provider = getattr(self.llm, "provider", "")
        fallback_provider = _resolve_synthesis_fallback_provider(self.config, primary_provider)
        if not fallback_provider or fallback_provider == primary_provider:
            return None

        fallback_model = _resolve_synthesis_fallback_model(self.config, fallback_provider)
        try:
            fallback_llm = LLMClient(
                self.config,
                provider_override=fallback_provider,
                model_override=fallback_model,
            )
        except Exception:
            self._fallback_llm = None
            return None

        if not fallback_llm.is_available():
            self._fallback_llm = None
            return None

        self._fallback_llm = fallback_llm
        return self._fallback_llm


def _build_synthesis_context(state: AgentState, language: str) -> str:
    citations = state.citations or _extract_citations(state.retrieved_docs)
    citation_index = {
        _citation_key(citation.source, citation.url): index + 1
        for index, citation in enumerate(citations)
    }
    docs_text = "\n\n".join(
        (
            f"[Source {citation_index.get(_citation_key(d.source, d.url), i+1)}] {d.source} | {d.url}\n{d.text}"
            if language == "en"
            else f"[来源 {citation_index.get(_citation_key(d.source, d.url), i+1)}] {d.source} | {d.url}\n{d.text}"
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

    compatibility_text = _build_profile_compatibility_context(state, language, citation_index)

    if language == "en":
        return (
            f"Current workflow:\n{state.workflow or 'unknown'}\n\n"
            f"Player profile:\n{state.player_profile.to_context_string(language='en')}\n\n"
            f"User question:\n{state.user_query}\n\n"
                f"Profile compatibility analysis:\n{compatibility_text}\n\n"
            f"Retrieved material:\n{docs_text}\n\n"
            f"Consensus:\n{consensus_text or 'None'}"
        )

    return (
        f"当前 workflow:\n{state.workflow or 'unknown'}\n\n"
        f"玩家状态:\n{state.player_profile.to_context_string(language='zh')}\n\n"
        f"用户问题:\n{state.user_query}\n\n"
            f"玩家当前可实现性分析:\n{compatibility_text}\n\n"
        f"检索到的内容:\n{docs_text}\n\n"
        f"共识分析:\n{consensus_text or '无'}"
    )


@lru_cache(maxsize=1)
def _profile_knowledge_base() -> dict[str, list[str]]:
    return load_knowledge_base()


def _build_profile_compatibility_context(
    state: AgentState,
    language: str,
    citation_index: dict[tuple[str, str], int],
) -> str:
    notes = _collect_profile_compatibility_notes(state.retrieved_docs, state.player_profile, citation_index, language)
    if language == "en":
        header = [
            f"- Skills explicitly declared: {'yes' if state.player_profile.skills_explicit else 'no'}",
            f"- Spells explicitly declared: {'yes' if state.player_profile.spells_explicit else 'no'}",
            f"- Transformations explicitly declared: {'yes' if state.player_profile.transformations_explicit else 'no'}",
            "- Only treat a source suggestion as unavailable when it is chapter-gated or contradicted by an explicit unlocked list.",
        ]
        if not notes:
            header.append("- No retrieved source recommendation is clearly blocked by the current profile.")
            return "\n".join(header)
        return "\n".join(header + notes)

    header = [
        f"- 技能列表是否明确给出：{'是' if state.player_profile.skills_explicit else '否'}",
        f"- 法术列表是否明确给出：{'是' if state.player_profile.spells_explicit else '否'}",
        f"- 变身列表是否明确给出：{'是' if state.player_profile.transformations_explicit else '否'}",
        "- 只有在来源建议被章节门槛限制，或与你明确给出的已解锁列表冲突时，才能判定为当前不可用。",
    ]
    if not notes:
        header.append("- 当前检索到的来源建议里，没有哪一条能被明确判定为你现在做不到。")
        return "\n".join(header)
    return "\n".join(header + notes)


def _collect_profile_compatibility_notes(
    docs,
    profile,
    citation_index: dict[tuple[str, str], int],
    language: str,
) -> list[str]:
    notes: list[str] = []
    seen: set[tuple[int, str, str]] = set()
    for doc in docs:
        doc_key = _citation_key(doc.source, doc.url)
        source_number = citation_index.get(doc_key, 0)
        for category, display_name, canonical_name, reason in _find_unavailable_doc_options(doc.text, profile):
            note_key = (source_number, category, canonical_name)
            if note_key in seen:
                continue
            seen.add(note_key)
            notes.append(
                _format_profile_compatibility_note(
                    source_number,
                    display_name,
                    reason,
                    category,
                    profile,
                    language=language,
                )
            )
    return notes[:8]


def _find_unavailable_doc_options(text: str, profile) -> list[tuple[str, str, str, str]]:
    unavailable: list[tuple[str, str, str, str]] = []
    for category, display_name, canonical_name in _extract_profile_option_mentions(text):
        availability, reason = _assess_option_availability(category, canonical_name, profile)
        if availability == "unavailable" and reason:
            unavailable.append((category, display_name, canonical_name, reason))
    return unavailable


def _extract_profile_option_mentions(text: str) -> list[tuple[str, str, str]]:
    kb = _profile_knowledge_base()
    mentions: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in kb.get("all_skills_tree", []):
        if item in text and ("skill", item) not in seen:
            mentions.append(("skill", item, item))
            seen.add(("skill", item))

    for item in kb.get("all_spells", []):
        if item in text and ("spell", item) not in seen:
            mentions.append(("spell", item, item))
            seen.add(("spell", item))

    for english_name, (category, canonical_name) in _ENGLISH_NAMED_OPTIONS.items():
        if english_name in text and (category, canonical_name) not in seen:
            mentions.append((category, english_name, canonical_name))
            seen.add((category, canonical_name))

    for match in re.finditer(r"变身[:：]\s*([^\s，。,；;]+)", text):
        transformation = match.group(1).strip()
        if transformation and ("transformation", transformation) not in seen:
            mentions.append(("transformation", transformation, transformation))
            seen.add(("transformation", transformation))

    return mentions


def _assess_option_availability(category: str, option_name: str, profile) -> tuple[str, str]:
    chapter_gate = CHAPTER_GATES.get(option_name)
    if chapter_gate is not None and profile.chapter is not None and profile.chapter < chapter_gate:
        return "unavailable", f"chapter_gate:{chapter_gate}"

    if category == "skill":
        if option_name in profile.unlocked_skills:
            return "available", "owned"
        if profile.skills_explicit:
            return "unavailable", "missing_from_explicit_skill_list"
        return "unknown", ""

    if category == "spell":
        if option_name in set(profile.unlocked_spells) | set(profile.equipped_spells):
            return "available", "owned"
        if profile.spells_explicit:
            return "unavailable", "missing_from_explicit_spell_list"
        return "unknown", ""

    if category == "transformation":
        if option_name in profile.unlocked_transformations:
            return "available", "owned"
        if profile.transformations_explicit:
            return "unavailable", "missing_from_explicit_transformation_list"
        return "unknown", ""

    return "unknown", ""


def _format_profile_compatibility_note(
    source_number: int,
    display_name: str,
    reason: str,
    category: str,
    profile,
    language: str,
) -> str:
    source_label_en = f"[Source {source_number}]" if source_number else "[Source]"
    source_label_zh = f"[来源 {source_number}]" if source_number else "[来源]"

    if reason.startswith("chapter_gate:"):
        gate = reason.split(":", 1)[1]
        if language == "en":
            return f"- {source_label_en} {display_name}: unavailable now because it is gated until Chapter {gate}."
        return f"- {source_label_zh} {display_name}：当前不可用，因为它至少要到第 {gate} 章。"

    if reason == "missing_from_explicit_skill_list":
        if language == "en":
            return f"- {source_label_en} {display_name}: unavailable now because it is not in your declared unlocked skills."
        return f"- {source_label_zh} {display_name}：当前不可用，因为它不在你明确给出的已解锁技能里。"

    if reason == "missing_from_explicit_spell_list":
        if language == "en":
            return f"- {source_label_en} {display_name}: unavailable now because it is not in your declared unlocked spells."
        return f"- {source_label_zh} {display_name}：当前不可用，因为它不在你明确给出的已解锁法术里。"

    if reason == "missing_from_explicit_transformation_list":
        if language == "en":
            return f"- {source_label_en} {display_name}: unavailable now because it is not in your declared unlocked transformations."
        return f"- {source_label_zh} {display_name}：当前不可用，因为它不在你明确给出的已解锁变身里。"

    if language == "en":
        return f"- {source_label_en} {display_name}: currently not usable for your build context."
    return f"- {source_label_zh} {display_name}：当前不适用于你给出的 build 上下文。"


def _resolve_synthesis_fallback_provider(config: dict, primary_provider: str) -> str:
    llm_config = config.get("llm", {})
    configured = llm_config.get("synthesis_fallback_provider") or llm_config.get("fallback_provider")
    if configured:
        return str(configured).strip().lower()
    if primary_provider == "groq":
        return "anthropic"
    return ""


def _resolve_synthesis_fallback_model(config: dict, provider: str) -> str | None:
    llm_config = config.get("llm", {})
    configured = llm_config.get("synthesis_fallback_model") or llm_config.get("fallback_model")
    if configured:
        return str(configured)
    if provider == "anthropic":
        return llm_config.get("model")
    return None


def _extract_citations(docs) -> list[Citation]:
    seen = set()
    citations = []
    for doc in docs:
        key = _citation_key(doc.source, doc.url)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source=doc.source,
                url=doc.url,
                excerpt=_build_citation_excerpt(doc.text),
                author=doc.metadata.get("author", ""),
            ))
    return citations


def _build_citation_excerpt(text: str, max_chars: int = 160) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    sentence_breaks = [match.end() for match in _EXCERPT_SENTENCE_BREAK_PATTERN.finditer(cleaned[: max_chars + 1])]
    if sentence_breaks:
        candidate_end = sentence_breaks[-1]
        if candidate_end >= max_chars // 2:
            excerpt = cleaned[:candidate_end].rstrip()
            return excerpt if candidate_end == len(cleaned) else f"{excerpt} …"

    candidate = cleaned[:max_chars].rstrip()
    last_space = candidate.rfind(" ")
    if last_space >= max_chars // 2:
        candidate = candidate[:last_space]
    return candidate.rstrip(" ,;:") + "…"


def _citation_key(source: str, url: str) -> tuple[str, str]:
    return (source or "", url or "")


def _build_no_results_answer(state: AgentState, language: str) -> str:
    next_step_hint = _build_next_step_hint(state, language)
    if language == "en":
        if state.workflow == "fact_lookup":
            return (
                "## Direct Answer\n"
                "- The retrieved material does not contain enough evidence to answer this reliably yet.\n\n"
                "## Your Question\n"
                f"- {state.user_query}\n\n"
                "## Next Step\n"
                f"- {next_step_hint}"
            )

        if state.workflow == "navigation":
            return (
                "## Location Answer\n"
                "- The retrieved material does not establish this location reliably yet.\n\n"
                "## Your Question\n"
                f"- {state.user_query}\n\n"
                "## Next Step\n"
                f"- {next_step_hint}"
            )

        return (
            "## Current Result\n"
            "- The retrieved material is not sufficient to produce a source-grounded guide answer yet.\n\n"
            "## Your Question\n"
            f"- {state.user_query}\n\n"
            "## Next Step\n"
            f"- {next_step_hint}"
        )

    if state.workflow == "fact_lookup":
        return (
            "## 当前结论\n"
            "- 检索内容中未找到足够依据，暂时无法给出可靠的事实回答。\n\n"
            "## 你的问题\n"
            f"- {state.user_query}\n\n"
            "## 建议\n"
            f"- {next_step_hint}"
        )

    if state.workflow == "navigation":
        return (
            "## 当前位置结论\n"
            "- 检索内容中未找到足够依据，暂时无法确认目标位置。\n\n"
            "## 你的问题\n"
            f"- {state.user_query}\n\n"
            "## 建议\n"
            f"- {next_step_hint}"
        )

    return (
        "## 当前结果\n"
        "- 检索内容中未找到足够资料，暂时无法生成有依据的攻略回答。\n\n"
        "## 你的问题\n"
        f"- {state.user_query}\n\n"
        "## 建议\n"
        f"- {next_step_hint}"
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


def _build_next_step_hint(state: AgentState, language: str) -> str:
    gaps = set(state.evidence_gaps or [])
    if language == "en":
        if "missing_entity" in gaps:
            return "Add the boss, move, NPC, item, or location name so the agent can narrow the search."
        if "missing_build_context" in gaps and "missing_chapter_context" in gaps:
            return "Add your current chapter and build so the recommendation can be personalized."
        if "missing_build_context" in gaps:
            return "Add your build or spell preference so the recommendation can be compared more precisely."
        if "missing_chapter_context" in gaps:
            return "Add your current chapter so the answer can avoid spoiler-heavy or locked recommendations."
        if "limited_community_evidence" in gaps:
            return "The system can still answer from wiki evidence, but a clearer boss or move name will help compensate for thin community coverage."
        return "Add a clearer boss, move, location, or chapter reference and try again."

    if "missing_entity" in gaps:
        return "请补充 boss、招式、NPC、道具或地点名，方便系统缩小检索范围。"
    if "missing_build_context" in gaps and "missing_chapter_context" in gaps:
        return "请补充你当前章节和流派，这样系统才能给出更贴合的建议。"
    if "missing_build_context" in gaps:
        return "请补充你当前流派、法术或偏好，这样系统才能做更准确的比较。"
    if "missing_chapter_context" in gaps:
        return "请补充你当前章节，这样系统可以避免给出剧透或未解锁建议。"
    if "limited_community_evidence" in gaps:
        return "当前社区资料较薄，但系统仍可基于 wiki 回答；如果你补充更明确的 boss 或招式名，答案会更稳。"
    return "请先补充更明确的 boss、招式、地点或当前章节信息后再试。"


def _estimate_answer_confidence(state: AgentState) -> float:
    if not state.retrieved_docs:
        return 0.0

    unique_sources = {doc.source for doc in state.retrieved_docs}
    score = 0.25
    if any(doc.source == "wiki" for doc in state.retrieved_docs):
        score += 0.25
    if len(unique_sources) >= 2:
        score += 0.2
    if state.citations:
        score += 0.15
    if state.consensus_analysis is not None:
        score += 0.1
    if state.stop_reason == "answered_via_secondary_provider":
        score -= 0.05
    if state.need_user_clarification:
        score -= 0.1
    if state.evidence_gaps:
        score -= min(0.05 * len(state.evidence_gaps), 0.15)
    if state.stop_reason and state.stop_reason != "answered":
        score -= 0.15
    return max(0.0, min(score, 1.0))


def _classify_generation_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "429" in message or "too many requests" in message or "rate limit" in message:
        return "generation_rate_limited_fallback"
    return "generation_failed_fallback"


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
