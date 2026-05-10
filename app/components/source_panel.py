"""
Source panel: shows retrieved docs, consensus stats, and agent trace.
This is the demo killer — reviewers see exactly WHY the system gave that answer.
"""

import streamlit as st
from app.session import get_last_state


def render_source_panel():
    state = get_last_state()
    if state is None:
        st.info("Ask a question to see sources, consensus, and the agent trace.")
        return

    if state.profile_updates:
        st.subheader("Profile Updates")
        for update in state.profile_updates:
            st.write(
                f"**{update['field']}**: {update['old_value']} -> {update['new_value']}"
            )
            if update.get("source"):
                st.caption(f"Source: {update['source']}")

    # --- Citations ---
    st.subheader("Sources")
    if state.citations:
        for i, cite in enumerate(state.citations, 1):
            title = f"[{i}] {cite.source}"
            if cite.author:
                title += f" | {cite.author}"
            with st.expander(title):
                if cite.url:
                    st.markdown(f"[Open original source]({cite.url})")
                    st.caption(cite.url)
                else:
                    st.caption("This citation has no original URL.")
                st.write(cite.excerpt)
                if cite.author:
                    st.caption(f"Author / site: {cite.author}")
    else:
        st.caption("No citations")

    # --- Consensus ---
    if state.consensus_analysis:
        st.subheader("Consensus")
        strategies = state.consensus_analysis.get("strategies", [])
        for s in strategies:
            label = s["label"]
            count = s["source_count"]
            contested = " ⚠️ contested" if s.get("is_contested") else ""
            st.write(f"**{label}**: {count} supporting source(s){contested}")
            breakdown = s.get("sources", {})
            if breakdown:
                st.caption(" | ".join(f"{src}: {n}" for src, n in breakdown.items()))

        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            st.warning("Conflicts")
            for c in conflicts:
                st.write(f"- **{c['topic']}**: support({len(c['pro'])}) vs oppose({len(c['con'])})")

    # --- Agent Trace ---
    with st.expander("Agent Trace (Debug)"):
        for event in state.trace:
            st.code(
                f"[{event.agent}] step {event.step}\n"
                f"  action: {event.action}\n"
                f"  obs: {event.observation[:200]}",
                language="text",
            )

    st.caption(f"Workflow: {state.workflow} | Trace steps: {len(state.trace)}")
