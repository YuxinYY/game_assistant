"""
Streamlit entry point.
Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.session import init_session
from app.components.chat_ui import render_chat_ui
from app.components.profile_panel import render_profile_panel
from app.components.source_panel import render_source_panel
from src.utils.logging import configure_logging

st.set_page_config(
    page_title="黑神话攻略助手",
    page_icon="🐉",
    layout="wide",
)

configure_logging()
init_session()

# Layout: sidebar (profile) | main chat | right panel (sources)
render_profile_panel()

col_chat, col_sources = st.columns([3, 2])

with col_chat:
    render_chat_ui()

with col_sources:
    render_source_panel()
