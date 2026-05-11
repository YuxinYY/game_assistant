"""
Source panel: shows retrieved docs, consensus stats, and agent trace.
This is the demo killer — reviewers see exactly WHY the system gave that answer.
"""

import streamlit as st
from app.session import get_last_state
from src.utils.language import preferred_response_language


def render_source_panel():
    state = get_last_state()
    if state is None:
        st.info(_ui_text("zh", "提问后即可查看来源、共识和 agent trace。", "Ask a question to see sources, consensus, and the agent trace."))
        return

    language = preferred_response_language(state.user_query)

    if state.profile_updates:
        st.subheader(_ui_text(language, "玩家状态更新", "Profile Updates"))
        for update in state.profile_updates:
            st.write(
                f"**{update['field']}**: {update['old_value']} -> {update['new_value']}"
            )
            if update.get("source"):
                st.caption(f"{_ui_text(language, '来源', 'Source')}: {update['source']}")

    # --- Citations ---
    st.subheader(_ui_text(language, "参考来源", "Sources"))
    if state.citations:
        for i, cite in enumerate(state.citations, 1):
            title = f"[{i}] {cite.source}"
            if cite.author:
                title += f" | {cite.author}"
            with st.expander(title):
                if cite.url:
                    st.markdown(f"[{_ui_text(language, '打开原始来源', 'Open original source')}]({cite.url})")
                    st.caption(cite.url)
                else:
                    st.caption(_ui_text(language, "该引用没有原始 URL。", "This citation has no original URL."))
                st.write(cite.excerpt)
                if cite.author:
                    st.caption(f"{_ui_text(language, '作者 / 站点', 'Author / site')}: {cite.author}")
    else:
        st.caption(_ui_text(language, "暂无引用", "No citations"))

    # --- Consensus ---
    if state.consensus_analysis:
        st.subheader(_ui_text(language, "共识分析", "Consensus"))
        strategies = state.consensus_analysis.get("strategies", [])
        for s in strategies:
            label = s["label"]
            count = s["source_count"]
            contested = _ui_text(language, " ⚠️ 存在争议", " ⚠️ contested") if s.get("is_contested") else ""
            st.write(f"**{label}**: {_format_support_count(count, language)}{contested}")
            breakdown = s.get("sources", {})
            if breakdown:
                st.caption(" | ".join(f"{src}: {n}" for src, n in breakdown.items()))

        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            st.warning(_ui_text(language, "冲突观点", "Conflicts"))
            for c in conflicts:
                if language == "en":
                    st.write(f"- **{c['topic']}**: support({len(c['pro'])}) vs oppose({len(c['con'])})")
                else:
                    st.write(f"- **{c['topic']}**: 支持({len(c['pro'])}) vs 反对({len(c['con'])})")

    # --- Execution Summary ---
    st.subheader(_ui_text(language, "执行摘要", "Execution Summary"))
    if state.execution_plan is not None:
        if state.execution_plan.goals:
            st.write(f"**{_ui_text(language, '计划目标', 'Plan goals')}**")
            for goal in state.execution_plan.goals:
                st.write(f"- {_goal_label(goal, language)}")

        if state.evidence_gaps:
            st.write(f"**{_ui_text(language, '证据缺口', 'Evidence gaps')}**")
            for gap in state.evidence_gaps:
                st.write(f"- {_gap_label(gap, language)}")

        if state.completed_steps:
            st.write(f"**{_ui_text(language, '已完成步骤', 'Completed steps')}**")
            for step in state.completed_steps:
                st.write(f"- {step}")

        if state.skipped_steps:
            st.write(f"**{_ui_text(language, '已跳过步骤', 'Skipped steps')}**")
            for step in state.skipped_steps:
                st.write(f"- {step['agent']}: {step['reason']}")

    st.caption(
        f"{_ui_text(language, '停止原因', 'Stop reason')}: {_stop_reason_label(state.stop_reason, language)} | "
        f"{_ui_text(language, '答案置信度', 'Answer confidence')}: {state.answer_confidence:.2f}"
    )

    # --- Agent Trace ---
    with st.expander(_ui_text(language, "Agent Trace（调试）", "Agent Trace (Debug)")):
        for event in state.trace:
            st.code(
                f"[{event.agent}] step {event.step}\n"
                f"  {_ui_text(language, 'action', 'action')}: {event.action}\n"
                f"  {_ui_text(language, 'obs', 'obs')}: {event.observation[:200]}",
                language="text",
            )

    st.caption(
        f"{_ui_text(language, '工作流', 'Workflow')}: {state.workflow} | "
        f"{_ui_text(language, 'Trace 步数', 'Trace steps')}: {len(state.trace)}"
    )


def _ui_text(language: str, zh_text: str, en_text: str) -> str:
    return en_text if language == "en" else zh_text


def _format_support_count(count: int, language: str) -> str:
    if language == "en":
        return f"{count} supporting source(s)"
    return f"{count} 个来源支持"


def _goal_label(goal: str, language: str) -> str:
    labels = {
        "identify_entity": _ui_text(language, "识别实体", "Identify the entity"),
        "collect_official_evidence": _ui_text(language, "收集官方依据", "Collect official evidence"),
        "collect_community_counterplay": _ui_text(language, "收集社区应对经验", "Collect community counterplay"),
        "ground_answer_with_citations": _ui_text(language, "给出带引用的答案", "Ground the answer with citations"),
        "resolve_dodge_timing": _ui_text(language, "明确闪避时机", "Resolve dodge timing"),
        "resolve_punish_window": _ui_text(language, "明确反打窗口", "Resolve punish window"),
        "collect_build_specific_advice": _ui_text(language, "补充流派相关建议", "Collect build-specific advice"),
        "compare_build_options": _ui_text(language, "比较流派选择", "Compare build options"),
        "collect_location_evidence": _ui_text(language, "收集地点依据", "Collect location evidence"),
        "collect_exact_fact": _ui_text(language, "收集精确事实", "Collect the exact fact"),
        "enumerate_or_count_facts": _ui_text(language, "枚举或统计事实", "Enumerate or count facts"),
    }
    return labels.get(goal, goal)


def _gap_label(gap: str, language: str) -> str:
    labels = {
        "missing_entity": _ui_text(language, "缺少实体名", "Missing entity name"),
        "missing_chapter_context": _ui_text(language, "缺少章节信息", "Missing chapter context"),
        "missing_build_context": _ui_text(language, "缺少流派信息", "Missing build context"),
        "limited_community_evidence": _ui_text(language, "社区资料有限", "Limited community evidence"),
    }
    return labels.get(gap, gap)


def _stop_reason_label(stop_reason: str | None, language: str) -> str:
    labels = {
        "answered": _ui_text(language, "已正常生成回答", "Answered normally"),
        "answered_via_secondary_provider": _ui_text(language, "已通过第二提供方生成回答", "Answered via the secondary provider"),
        "insufficient_evidence": _ui_text(language, "证据不足", "Insufficient evidence"),
        "workflow_completed": _ui_text(language, "工作流已完成", "Workflow completed"),
        "generation_rate_limited_fallback": _ui_text(language, "生成限流，已退回抽取式整理", "Generation was rate-limited, so extractive fallback was used"),
        "generation_failed_fallback": _ui_text(language, "生成失败，已退回抽取式整理", "Generation failed, so extractive fallback was used"),
    }
    if not stop_reason:
        return _ui_text(language, "未知", "unknown")
    return labels.get(stop_reason, stop_reason)
