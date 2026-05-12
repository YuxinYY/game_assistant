"""
Profile panel: sidebar widget for viewing and editing player state.
"""

import streamlit as st
from app.session import get_profile, update_profile


BUILD_LABELS = {
    None: "Not set",
    "dodge": "Dodge",
    "parry": "Parry",
    "spell": "Spell",
    "hybrid": "Hybrid",
}


def render_profile_panel():
    st.sidebar.header("Player Profile")

    profile = get_profile()
    chapter_options = [None, 1, 2, 3, 4, 5, 6]
    build_options = [None, "dodge", "parry", "spell", "hybrid"]
    staff_level_options = [None, 1, 2, 3, 4, 5]

    chapter = st.sidebar.selectbox(
        "Current chapter",
        options=chapter_options,
        index=chapter_options.index(profile.chapter) if profile.chapter in chapter_options else 0,
        format_func=lambda x: "Not set" if x is None else f"Chapter {x}",
    )

    build = st.sidebar.selectbox(
        "Build",
        options=build_options,
        index=build_options.index(profile.build) if profile.build in build_options else 0,
        format_func=lambda x: BUILD_LABELS.get(x, x),
    )

    staff_level = st.sidebar.selectbox(
        "Staff level",
        options=staff_level_options,
        index=staff_level_options.index(profile.staff_level) if profile.staff_level in staff_level_options else 0,
        format_func=lambda x: "Not set" if x is None else f"Lv.{x}",
    )

    skills_input = st.sidebar.text_input(
        "Unlocked skills (comma-separated)",
        value=", ".join(profile.unlocked_skills),
    )

    spells_input = st.sidebar.text_input(
        "Unlocked spells (comma-separated)",
        value=", ".join(profile.unlocked_spells),
    )

    transforms_input = st.sidebar.text_input(
        "Unlocked transformations (comma-separated)",
        value=", ".join(profile.unlocked_transformations),
    )

    if st.sidebar.button("Update profile"):
        update_profile(
            chapter=chapter,
            build=build,
            staff_level=staff_level,
            unlocked_skills=[s.strip() for s in skills_input.split(",") if s.strip()],
            unlocked_spells=[s.strip() for s in spells_input.split(",") if s.strip()],
            unlocked_transformations=[s.strip() for s in transforms_input.split(",") if s.strip()],
            skills_explicit=True,
            spells_explicit=True,
            transformations_explicit=True,
        )
        st.sidebar.success("Profile updated")

    if st.sidebar.button("Clear filters"):
        update_profile(
            chapter=None,
            build=None,
            staff_level=None,
            equipped_spirit=None,
            equipped_armor=[],
            equipped_spells=[],
            unlocked_skills=[],
            unlocked_spells=[],
            unlocked_transformations=[],
            skills_explicit=False,
            spells_explicit=False,
            transformations_explicit=False,
        )
        st.sidebar.success("Filters cleared")

    st.sidebar.caption(get_profile().to_context_string(language="en"))
