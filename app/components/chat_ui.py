"""
Main chat interface component.
"""

import streamlit as st
from app.session import get_orchestrator, get_profile, get_history, add_message, set_last_state


def render_chat_ui():
    st.title("Black Myth: Wukong Guide Assistant")
    st.caption("Multi-agent answers with citations and build-aware suggestions")

    _render_history()

    # Query input
    query = st.chat_input("Ask in English, for example: How do I beat Tiger Vanguard's charged slam?")
    if not query:
        return

    _handle_query(query)
    st.rerun()


def _render_history() -> None:
    for msg in get_history():
        with st.chat_message(msg.role):
            st.markdown(msg.content)


def _handle_query(query: str) -> None:
    add_message("user", query)

    with st.spinner("Searching the wiki and community sources..."):
        state = get_orchestrator().run(
            query=query,
            profile=get_profile(),
            history=get_history(),
        )
        set_last_state(state)

    answer = state.final_answer or "The system did not produce an answer. Check the model configuration."
    add_message("assistant", answer)
