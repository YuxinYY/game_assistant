"""
Streamlit entry point.
Run: streamlit run app/streamlit_app.py
"""

import base64
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from app.session import init_session
from app.components.chat_ui import render_chat_ui
from app.components.profile_panel import render_profile_panel
from app.components.source_panel import render_source_panel
from src.utils.logging import configure_logging


@st.cache_data(show_spinner=False)
def _background_video_data_url() -> str:
    video_path = Path(__file__).parent / "static" / "beat_vid1.mp4"
    if not video_path.exists():
        return ""
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"

st.set_page_config(
    page_title="Black Myth: Wukong Guide Assistant",
    page_icon="🐉",
    layout="wide",
)

background_video_url = _background_video_data_url()

st.markdown("""
<style>
/* ── Video background ── */
#bg-video-container {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1;
    overflow: hidden;
}
#bg-video-container video {
    width: 100%; height: 100%;
    object-fit: cover;
}
#bg-video-container::after {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.58);
}

/* ── Strip every Streamlit background layer ── */
html, body { background: transparent !important; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] > div,
.main, .block-container {
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar: dark glass ── */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 8, 0.72) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Global text color ── */
.stApp, .stMarkdown, h1, h2, h3, h4, p, li, label, span,
[data-testid="stText"], [data-testid="stMarkdown"],
[data-testid="stCaptionContainer"] {
    color: #e8e0d0 !important;
}

/* ── Chat message bubbles: glass cards ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    margin-bottom: 10px;
}

/* ── Expanders & info boxes ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px;
}

/* ── Input box ── */
[data-testid="stChatInputContainer"] {
    background: rgba(20, 20, 20, 0.80) !important;
    backdrop-filter: blur(10px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}
</style>
<div id="bg-video-container">
    <video id="bg-video" autoplay muted loop playsinline>
        <source src="__BACKGROUND_VIDEO_URL__" type="video/mp4">
    </video>
</div>
""".replace("__BACKGROUND_VIDEO_URL__", background_video_url), unsafe_allow_html=True)

# Mute button via iframe — persists across Streamlit rerenders
components.html("""
<style>
  #mute-btn {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    background: rgba(0,0,0,0.55);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 22px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    cursor: pointer;
    backdrop-filter: blur(8px);
    transition: background 0.2s;
  }
  #mute-btn:hover { background: rgba(0,0,0,0.80); }
</style>
<button id="mute-btn">🔇</button>
<script>
  var muted = true;
  document.getElementById("mute-btn").addEventListener("click", function() {
    var doc = window.parent.document;
    var video = doc.getElementById("bg-video");
    if (!video) return;
    muted = !muted;
    video.muted = muted;
    this.textContent = muted ? "🔇" : "🔊";
  });
</script>
""", height=80)

configure_logging()
init_session()

# Layout: sidebar (profile) | main chat | right panel (sources)
render_profile_panel()

col_chat, col_sources = st.columns([3, 2])

with col_chat:
    render_chat_ui()

with col_sources:
    render_source_panel()
