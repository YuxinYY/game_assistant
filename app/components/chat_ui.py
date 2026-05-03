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
    screenshot = st.file_uploader("上传游戏截图（可选，自动识别你的 build）",
                                  type=["png", "jpg", "jpeg"],
                                  label_visibility="collapsed")

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
            image_bytes = screenshot.read() if screenshot else None
            state = get_orchestrator().run(
                query=query,
                profile=get_profile(),
                history=get_history(),
                screenshot=image_bytes,
            )
            set_last_state(state)

        answer = state.final_answer or "（系统未生成回答，请检查配置）"
        st.markdown(answer)

    add_message("assistant", answer)
