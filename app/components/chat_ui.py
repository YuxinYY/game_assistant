"""
Main chat interface component.
"""

import streamlit as st
from app.session import get_orchestrator, get_profile, get_history, add_message, set_last_state


def render_chat_ui():
    st.title("黑神话悟空攻略助手")
    st.caption("基于多 agent + 可引用来源的个性化攻略系统")

    # Display conversation history
    for msg in get_history():
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # Screenshot upload
    screenshots = st.file_uploader(
        "上传游戏截图（可选，自动识别你的 build）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    screenshot_payloads = [uploaded.getvalue() for uploaded in (screenshots or [])]

    parse_only = st.button(
        "识别截图并更新状态",
        disabled=not screenshot_payloads,
        use_container_width=True,
    )
    if parse_only:
        synthetic_query = f"请根据我上传的{len(screenshot_payloads)}张截图更新我的状态"
        with st.chat_message("user"):
            st.markdown(synthetic_query)
        add_message("user", synthetic_query)

        with st.chat_message("assistant"):
            with st.spinner("正在识别截图并更新玩家状态..."):
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
    query = st.chat_input("输入你的问题，例如：虎先锋那个蓄力的招怎么躲？")
    if not query:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    add_message("user", query)

    # Run agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("正在查询多个来源..."):
            state = get_orchestrator().run(
                query=query,
                profile=get_profile(),
                history=get_history(),
                screenshots=screenshot_payloads,
            )
            set_last_state(state)

        answer = state.final_answer or "（系统未生成回答，请检查配置）"
        st.markdown(answer)

    add_message("assistant", answer)


def _build_profile_update_message(state) -> str:
    if state.profile_updates:
        lines = ["## 截图识别结果"]
        for update in state.profile_updates:
            lines.append(
                f"- {update['field']}: {update['old_value']} → {update['new_value']}"
            )
        lines.append("")
        lines.append("## 当前玩家状态")
        lines.append(f"- {state.player_profile.to_context_string()}")
        return "\n".join(lines)

    if any(event.action == "vision_unavailable" for event in state.trace):
        return (
            "## 截图识别结果\n"
            "- 当前配置下没有可用的视觉模型，所以截图未被解析。\n\n"
            "## 建议\n"
            "- 如果主模型使用 Groq，可以单独配置 Anthropic 作为视觉解析 provider。"
        )

    return (
        "## 截图识别结果\n"
        "- 本次未识别出可合并到玩家画像的字段。\n\n"
        "## 建议\n"
        "- 尽量上传清晰的 HUD、背包或技能树截图。"
    )
