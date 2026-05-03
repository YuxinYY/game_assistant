"""
Source panel: shows retrieved docs, consensus stats, and agent trace.
This is the demo killer — reviewers see exactly WHY the system gave that answer.
"""

import streamlit as st
from app.session import get_last_state


def render_source_panel():
    state = get_last_state()
    if state is None:
        st.info("先提问，这里会显示系统的推理过程。")
        return

    # --- Citations ---
    st.subheader("引用来源")
    if state.citations:
        for i, cite in enumerate(state.citations, 1):
            with st.expander(f"[{i}] {cite.source} — {cite.url}"):
                st.write(cite.excerpt)
                if cite.author:
                    st.caption(f"作者: {cite.author}")
    else:
        st.caption("无引用")

    # --- Consensus ---
    if state.consensus_analysis:
        st.subheader("共识分析")
        strategies = state.consensus_analysis.get("strategies", [])
        for s in strategies:
            label = s["label"]
            count = s["source_count"]
            contested = " ⚠️ 存在争议" if s.get("is_contested") else ""
            st.write(f"**{label}**: {count} 个来源支持{contested}")
            breakdown = s.get("sources", {})
            if breakdown:
                st.caption(" | ".join(f"{src}: {n}" for src, n in breakdown.items()))

        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            st.warning("争议点")
            for c in conflicts:
                st.write(f"- **{c['topic']}**: 支持({len(c['pro'])}) vs 反对({len(c['con'])})")

    # --- Agent Trace ---
    with st.expander("Agent 推理过程（Debug）"):
        for event in state.trace:
            st.code(
                f"[{event.agent}] step {event.step}\n"
                f"  action: {event.action}\n"
                f"  obs: {event.observation[:200]}",
                language="text",
            )

    st.caption(f"Workflow: {state.workflow} | Trace steps: {len(state.trace)}")
