"""
Main chat interface component.
"""

import streamlit as st
from app.session import get_orchestrator, get_profile, get_history, add_message, set_last_state


def render_chat_ui():
    st.title("Black Myth: Wukong Guide Assistant")
    st.caption("Multi-agent answers with citations and build-aware suggestions")

    # Display conversation history
    for msg in get_history():
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # Screenshot upload
    screenshots = st.file_uploader(
        "Upload gameplay screenshots to update your build automatically",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    screenshot_payloads = [uploaded.getvalue() for uploaded in (screenshots or [])]

    parse_only = st.button(
        "Parse screenshots and update profile",
        disabled=not screenshot_payloads,
        use_container_width=True,
    )
    if parse_only:
        count = len(screenshot_payloads)
        synthetic_query = f"Update my profile from the {count} uploaded screenshot{'s' if count != 1 else ''}."
        with st.chat_message("user"):
            st.markdown(synthetic_query)
        add_message("user", synthetic_query)

        with st.chat_message("assistant"):
            with st.spinner("Reading screenshots and updating your player profile..."):
                state = get_orchestrator().run(
                    query="",
                    profile=get_profile(),
                    history=get_history(),
                    screenshots=screenshot_payloads,
                )
                set_last_state(state)

            answer = _build_profile_update_message(state)
            st.markdown(answer)

        add_message("assistant", answer)
        return

    # Query input
    query = st.chat_input("Ask in English, for example: How do I beat Tiger Vanguard's charged slam?")
    if not query:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    add_message("user", query)

    # Run agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching the wiki and community sources..."):
            state = get_orchestrator().run(
                query=query,
                profile=get_profile(),
                history=get_history(),
                screenshots=screenshot_payloads,
            )
            set_last_state(state)

        answer = state.final_answer or "The system did not produce an answer. Check the model configuration."
        st.markdown(answer)

    add_message("assistant", answer)


def _build_profile_update_message(state) -> str:
    if state.profile_updates:
        lines = ["## Screenshot Parsing"]
        for update in state.profile_updates:
            lines.append(
                f"- {update['field']}: {_format_profile_value(update['old_value'])} -> {_format_profile_value(update['new_value'])}"
            )
        lines.append("")
        lines.append("## Current Player Profile")
        lines.append(f"- {state.player_profile.to_context_string(language='en')}")
        return "\n".join(lines)

    if any(event.action == "vision_unavailable" for event in state.trace):
        return (
            "## Screenshot Parsing\n"
            "- No vision model is available in the current configuration, so the screenshots were not parsed.\n\n"
            "## Suggestion\n"
            "- If you use Groq for text, configure Anthropic separately as the vision provider."
        )

    return (
        "## Screenshot Parsing\n"
        "- No player-profile fields were extracted from these screenshots.\n\n"
        "## Suggestion\n"
        "- Upload clear HUD, inventory, or skill-tree screenshots."
    )


def _format_profile_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "None"
    return str(value)
