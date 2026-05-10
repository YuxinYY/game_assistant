"""
Profile panel: sidebar widget for viewing and editing player state.
"""

import streamlit as st
from app.session import get_profile, update_profile


BUILD_LABELS = {
    None: "未设置",
    "dodge": "闪身流",
    "parry": "棍反流",
    "spell": "法术流",
    "hybrid": "混合",
}


def render_profile_panel():
    st.sidebar.header("玩家状态")

    profile = get_profile()
    chapter_options = [None, 1, 2, 3, 4, 5, 6]
    build_options = [None, "dodge", "parry", "spell", "hybrid"]
    staff_level_options = [None, 1, 2, 3, 4, 5]

    chapter = st.sidebar.selectbox(
        "当前章节",
        options=chapter_options,
        index=chapter_options.index(profile.chapter) if profile.chapter in chapter_options else 0,
        format_func=lambda x: "未设置" if x is None else f"第 {x} 章",
    )

    build = st.sidebar.selectbox(
        "流派",
        options=build_options,
        index=build_options.index(profile.build) if profile.build in build_options else 0,
        format_func=lambda x: BUILD_LABELS.get(x, x),
    )

    staff_level = st.sidebar.selectbox(
        "棍法等级",
        options=staff_level_options,
        index=staff_level_options.index(profile.staff_level) if profile.staff_level in staff_level_options else 0,
        format_func=lambda x: "未设置" if x is None else f"Lv.{x}",
    )

    skills_input = st.sidebar.text_input(
        "已解锁技能（逗号分隔）",
        value=", ".join(profile.unlocked_skills),
    )

    spells_input = st.sidebar.text_input(
        "已解锁法术",
        value=", ".join(profile.unlocked_spells),
    )

    transforms_input = st.sidebar.text_input(
        "已解锁变身",
        value=", ".join(profile.unlocked_transformations),
    )

    if st.sidebar.button("更新状态"):
        update_profile(
            chapter=chapter,
            build=build,
            staff_level=staff_level,
            unlocked_skills=[s.strip() for s in skills_input.split(",") if s.strip()],
            unlocked_spells=[s.strip() for s in spells_input.split(",") if s.strip()],
            unlocked_transformations=[s.strip() for s in transforms_input.split(",") if s.strip()],
        )
        st.sidebar.success("已更新")

    if st.sidebar.button("清空筛选"):
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
        )
        st.sidebar.success("已清空")

    st.sidebar.caption(get_profile().to_context_string())
