"""
ProfileAgent: maintains and applies the player profile.
Responsibilities:
  - Parse VLM screenshot to extract build info
  - Filter retrieved_docs to remove content about locked skills/transformations
  - Annotate state with what is/isn't applicable for this player
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Document
from src.tools.screenshot_parser import parse_screenshot
from src.tools.spoiler_filter import apply_spoiler_filter

# Chapter gates: which chapter unlocks each item
CHAPTER_GATES: dict[str, int] = {
    "广智": 3,
    "定风珠": 2,
    "铜头铁臂": 1,
    "定身术": 1,
}


class _ParseScreenshot(Tool):
    name = "parse_screenshot"
    description = "从游戏截图（VLM）提取玩家装备、技能、等级信息"

    def __call__(self, image_bytes: bytes) -> dict:
        return parse_screenshot(image_bytes)


class _SpoilerFilter(Tool):
    name = "spoiler_filter"
    description = "过滤超出玩家当前章节的剧透内容"

    def __call__(self, docs: list, max_chapter: int) -> list:
        return apply_spoiler_filter(docs, max_chapter)


class ProfileAgent(BaseAgent):
    name = "profile_agent"
    prompt_file = "profile_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_ParseScreenshot(), _SpoilerFilter()]

    def execute(self, state: AgentState) -> AgentState:
        # 1. Optionally update profile from screenshot
        if state.user_screenshot:
            parsed = parse_screenshot(state.user_screenshot)
            _update_profile_from_parsed(state, parsed)

        # 2. Filter retrieved docs: remove chapter-gated content player can't access
        state.retrieved_docs = _filter_by_profile(state.retrieved_docs, state.player_profile)

        # 3. Apply spoiler filter if enabled
        if self.config.get("spoiler", {}).get("enable", True):
            state.retrieved_docs = apply_spoiler_filter(
                state.retrieved_docs, state.player_profile.chapter
            )
        return state


def _filter_by_profile(docs: list[Document], profile) -> list[Document]:
    """Remove docs that recommend items the player hasn't unlocked yet."""
    filtered = []
    for doc in docs:
        blocked = any(
            item in doc.text and profile.chapter < gate
            for item, gate in CHAPTER_GATES.items()
        )
        if not blocked:
            filtered.append(doc)
    return filtered


def _update_profile_from_parsed(state: AgentState, parsed: dict):
    if "chapter" in parsed:
        state.player_profile.chapter = parsed["chapter"]
    if "build" in parsed:
        state.player_profile.build = parsed["build"]
    if "staff_level" in parsed:
        state.player_profile.staff_level = parsed["staff_level"]
    if "unlocked_skills" in parsed:
        state.player_profile.unlocked_skills = parsed["unlocked_skills"]
