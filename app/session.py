"""
Streamlit session state management.
Isolates all st.session_state access so components don't reach into global state directly.
"""

import streamlit as st
from src.core.state import PlayerProfile, Message
from src.core.orchestrator import Orchestrator, load_config


def init_session():
    """Call once at app startup to initialize all session keys."""
    if "orchestrator" not in st.session_state:
        config = load_config()
        st.session_state.orchestrator = Orchestrator(config)

    if "profile" not in st.session_state:
        st.session_state.profile = PlayerProfile()
        st.session_state.profile_is_explicit = False
    elif "profile_is_explicit" not in st.session_state:
        st.session_state.profile_is_explicit = not _looks_like_legacy_default_profile(st.session_state.profile)
        if not st.session_state.profile_is_explicit:
            _clear_profile_filters(st.session_state.profile)

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
    st.session_state.profile_is_explicit = _has_any_profile_filter(st.session_state.profile)


def get_history() -> list[Message]:
    return st.session_state.history


def add_message(role: str, content: str):
    st.session_state.history.append(Message(role=role, content=content))


def set_last_state(state):
    st.session_state.last_state = state


def get_last_state():
    return st.session_state.last_state


def _has_any_profile_filter(profile: PlayerProfile) -> bool:
    return any(
        [
            profile.chapter is not None,
            profile.build is not None,
            profile.staff_level is not None,
            bool(profile.equipped_spirit),
            bool(profile.equipped_armor),
            bool(profile.equipped_spells),
            bool(profile.unlocked_skills),
            bool(profile.unlocked_spells),
            bool(profile.unlocked_transformations),
            profile.skills_explicit,
            profile.spells_explicit,
            profile.transformations_explicit,
        ]
    )


def _looks_like_legacy_default_profile(profile: PlayerProfile) -> bool:
    return (
        profile.chapter == 1
        and profile.build == "dodge"
        and profile.staff_level == 1
        and not profile.equipped_spirit
        and not profile.equipped_armor
        and not profile.equipped_spells
        and not profile.unlocked_skills
        and not profile.unlocked_spells
        and not profile.unlocked_transformations
    )


def _clear_profile_filters(profile: PlayerProfile) -> None:
    profile.chapter = None
    profile.build = None
    profile.staff_level = None
    profile.equipped_spirit = None
    profile.equipped_armor = []
    profile.equipped_spells = []
    profile.unlocked_skills = []
    profile.unlocked_spells = []
    profile.unlocked_transformations = []
    profile.skills_explicit = False
    profile.spells_explicit = False
    profile.transformations_explicit = False
