"""
Streamlit session state management.
Isolates all st.session_state access so components don't reach into global state directly.
"""

import streamlit as st
import yaml
from src.core.state import PlayerProfile, Message
from src.core.orchestrator import Orchestrator


def init_session():
    """Call once at app startup to initialize all session keys."""
    if "orchestrator" not in st.session_state:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        st.session_state.orchestrator = Orchestrator(config)

    if "profile" not in st.session_state:
        st.session_state.profile = PlayerProfile()

    if "history" not in st.session_state:
        st.session_state.history: list[Message] = []

    if "last_state" not in st.session_state:
        st.session_state.last_state = None


def get_orchestrator() -> Orchestrator:
    return st.session_state.orchestrator


def get_profile() -> PlayerProfile:
    return st.session_state.profile


def update_profile(**kwargs):
    for k, v in kwargs.items():
        setattr(st.session_state.profile, k, v)


def get_history() -> list[Message]:
    return st.session_state.history


def add_message(role: str, content: str):
    st.session_state.history.append(Message(role=role, content=content))


def set_last_state(state):
    st.session_state.last_state = state


def get_last_state():
    return st.session_state.last_state
