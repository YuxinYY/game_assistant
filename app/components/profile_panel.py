"""
Profile panel: sidebar widget for viewing and editing player state.
"""

import streamlit as st
from app.session import get_profile, update_profile


CHAPTER_LABELS = {
    None: "Not set",
    1: "Chapter 1 - Black Wind Mountain",
    2: "Chapter 2 - Yellow Wind Ridge",
    3: "Chapter 3 - The New West",
    4: "Chapter 4 - Webbed Hollow",
    5: "Chapter 5 - Flaming Mountains",
    6: "Chapter 6 - Mount Huaguo",
}

BUILD_LABELS = {
    None: "Not set",
    "dodge": "Dodge",
    "parry": "Parry",
    "spell": "Spell",
    "hybrid": "Hybrid",
}

STANCE_LABELS = {
    None: "Not set",
    "smash": "Smash Stance (劈棍)",
    "pillar": "Pillar Stance (立棍)",
    "thrust": "Thrust Stance (戳棍)",
}

SPELL_LABELS = {
    None: "Not set",
    "定身术": "Immobilize (定身术)",
    "安身法": "Ring of Fire (安身法)",
    "铜头铁臂": "Rock Solid (铜头铁臂)",
    "聚形散气": "Cloud Step (聚形散气)",
    "身外身法": "Pluck of Many (身外身法)",
    "禁字法": "Spell Binder (禁字法)",
}

TRANSFORMATION_LABELS = {
    None: "Not set",
    "Red Tides": "Red Tides (赤潮)",
    "Azure Dust": "Azure Dust (碧尘)",
    "Ashen Slumber": "Ashen Slumber (灰蛰)",
    "Hoarfrost": "Hoarfrost (皓霜)",
    "Umbral Abyss": "Umbral Abyss (乌川)",
    "Golden Lining": "Golden Lining (金岚)",
    "Violet Hail": "Violet Hail (藕雹)",
    "Ebon Flow": "Ebon Flow",
    "Dark Thunder": "Dark Thunder (黯雷)",
}

MYSTICISM_OPTIONS = [None, "定身术", "安身法"]
BODY_OPTIONS = [None, "铜头铁臂", "聚形散气"]
STRAND_OPTIONS = [None, "身外身法", "禁字法"]
TRANSFORMATION_OPTIONS = [
    None,
    "Red Tides",
    "Azure Dust",
    "Ashen Slumber",
    "Hoarfrost",
    "Umbral Abyss",
    "Golden Lining",
    "Violet Hail",
    "Ebon Flow",
    "Dark Thunder",
]

TRANSFORMATION_ALIASES = {
    "赤潮": "Red Tides",
    "碧尘": "Azure Dust",
    "灰蛰": "Ashen Slumber",
    "皓霜": "Hoarfrost",
    "乌川": "Umbral Abyss",
    "金岚": "Golden Lining",
    "藕雹": "Violet Hail",
    "黯雷": "Dark Thunder",
}


def render_profile_panel():
    st.sidebar.header("Player Profile")

    profile = get_profile()
    chapter_options = [None, 1, 2, 3, 4, 5, 6]
    build_options = [None, "dodge", "parry", "spell", "hybrid"]
    stance_options = [None, "smash", "pillar", "thrust"]
    staff_level_options = [None, 1, 2, 3, 4, 5]
    equipped_spells = list(profile.equipped_spells or [])

    st.sidebar.subheader("Story Progress")
    chapter = st.sidebar.selectbox(
        "Current chapter",
        options=chapter_options,
        index=chapter_options.index(profile.chapter) if profile.chapter in chapter_options else 0,
        format_func=lambda x: CHAPTER_LABELS.get(x, f"Chapter {x}"),
    )
    current_boss = st.sidebar.text_input("Current boss (optional)", value=profile.current_boss or "")

    st.sidebar.subheader("Combat Loadout")
    primary_stance_default = profile.primary_stance if profile.primary_stance in stance_options else None
    primary_stance = st.sidebar.selectbox(
        "Primary stance",
        options=stance_options,
        index=stance_options.index(primary_stance_default),
        format_func=lambda x: STANCE_LABELS.get(x, x),
    )

    equipped_mysticism = st.sidebar.selectbox(
        "Mysticism",
        options=MYSTICISM_OPTIONS,
        index=MYSTICISM_OPTIONS.index(_default_select_value(profile.equipped_mysticism, equipped_spells, MYSTICISM_OPTIONS)),
        format_func=lambda x: SPELL_LABELS.get(x, x),
    )
    equipped_body = st.sidebar.selectbox(
        "Body",
        options=BODY_OPTIONS,
        index=BODY_OPTIONS.index(_default_select_value(profile.equipped_body, equipped_spells, BODY_OPTIONS)),
        format_func=lambda x: SPELL_LABELS.get(x, x),
    )
    equipped_strand = st.sidebar.selectbox(
        "Strand",
        options=STRAND_OPTIONS,
        index=STRAND_OPTIONS.index(_default_select_value(profile.equipped_strand, equipped_spells, STRAND_OPTIONS)),
        format_func=lambda x: SPELL_LABELS.get(x, x),
    )

    normalized_transformations = [
        _normalize_transformation_value(value) for value in (profile.unlocked_transformations or [])
    ]
    equipped_transformation_default = _normalize_transformation_value(profile.equipped_transformation)
    if equipped_transformation_default not in TRANSFORMATION_OPTIONS:
        equipped_transformation_default = None

    equipped_transformation = st.sidebar.selectbox(
        "Equipped transformation",
        options=TRANSFORMATION_OPTIONS,
        index=TRANSFORMATION_OPTIONS.index(equipped_transformation_default),
        format_func=lambda x: TRANSFORMATION_LABELS.get(x, x),
    )
    unlocked_transformations = st.sidebar.multiselect(
        "Unlocked transformations",
        options=TRANSFORMATION_OPTIONS[1:],
        default=_filter_supported_options(normalized_transformations, TRANSFORMATION_OPTIONS[1:]),
        format_func=lambda x: TRANSFORMATION_LABELS.get(x, x),
    )

    with st.sidebar.expander("Advanced assistant overrides"):
        build = st.selectbox(
            "Build hint",
            options=build_options,
            index=build_options.index(profile.build) if profile.build in build_options else 0,
            format_func=lambda x: BUILD_LABELS.get(x, x),
        )
        staff_level = st.selectbox(
            "Staff level",
            options=staff_level_options,
            index=staff_level_options.index(profile.staff_level) if profile.staff_level in staff_level_options else 0,
            format_func=lambda x: "Not set" if x is None else f"Lv.{x}",
        )
        equipped_spirit = st.text_input("Equipped spirit (optional)", value=profile.equipped_spirit or "")
        armor_input = st.text_input("Equipped armor (comma-separated)", value=", ".join(profile.equipped_armor))
        skills_input = st.text_input("Explicit unlocked skills (comma-separated)", value=", ".join(profile.unlocked_skills))
        spells_input = st.text_input("Explicit unlocked spells (comma-separated)", value=", ".join(profile.unlocked_spells))
        st.caption("Fill these only if you want the assistant to strictly filter unavailable skills or spells.")

    if st.sidebar.button("Update profile"):
        explicit_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        explicit_spells = [s.strip() for s in spells_input.split(",") if s.strip()]
        equipped_spell_values = _unique_preserving_order(
            [value for value in [equipped_mysticism, equipped_body, equipped_strand] if value]
        )
        transformation_values = _unique_preserving_order(
            [*unlocked_transformations, *([equipped_transformation] if equipped_transformation else [])]
        )
        update_profile(
            chapter=chapter,
            current_boss=current_boss.strip() or None,
            build=build,
            primary_stance=primary_stance,
            staff_level=staff_level,
            equipped_spirit=equipped_spirit.strip() or None,
            equipped_armor=[s.strip() for s in armor_input.split(",") if s.strip()],
            equipped_mysticism=equipped_mysticism,
            equipped_body=equipped_body,
            equipped_strand=equipped_strand,
            equipped_transformation=equipped_transformation,
            equipped_spells=equipped_spell_values,
            unlocked_skills=explicit_skills,
            unlocked_spells=explicit_spells,
            unlocked_transformations=transformation_values,
            skills_explicit=bool(explicit_skills),
            spells_explicit=bool(explicit_spells),
            transformations_explicit=bool(transformation_values),
        )
        st.sidebar.success("Profile updated")

    if st.sidebar.button("Clear filters"):
        update_profile(
            chapter=None,
            current_boss=None,
            build=None,
            primary_stance=None,
            staff_level=None,
            equipped_spirit=None,
            equipped_armor=[],
            equipped_mysticism=None,
            equipped_body=None,
            equipped_strand=None,
            equipped_transformation=None,
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


def _default_select_value(current_value, fallback_values, options):
    if current_value in options:
        return current_value
    for value in fallback_values:
        if value in options:
            return value
    return None


def _filter_supported_options(values, options):
    return [value for value in values if value in options]


def _normalize_transformation_value(value):
    if value in TRANSFORMATION_ALIASES:
        return TRANSFORMATION_ALIASES[value]
    return value


def _unique_preserving_order(values):
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values
