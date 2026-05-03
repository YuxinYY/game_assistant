"""
ProfileAgent: the single owner of player-state updates and screenshot parsing.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Document
from src.tools.parsers import (
    CombatHUDParser,
    InventoryParser,
    ScreenshotClassifier,
    SkillTreeParser,
)
from src.tools.profile_ops import load_knowledge_base, merge_to_profile, validate_extraction
from src.tools.spoiler_filter import apply_spoiler_filter

# Chapter gates: which chapter unlocks each item
CHAPTER_GATES: dict[str, int] = {
    "广智": 3,
    "定风珠": 2,
    "铜头铁臂": 1,
    "定身术": 1,
}


class _FunctionTool(Tool):
    def __init__(self, name: str, description: str, fn: Callable[..., Any]):
        self.name = name
        self.description = description
        self._fn = fn

    def __call__(self, **kwargs):
        return self._fn(**kwargs)


class ProfileAgent(BaseAgent):
    name = "profile_agent"
    prompt_file = "profile_agent.txt"

    def __init__(
        self,
        config: dict,
        vlm_client=None,
        knowledge_base: dict[str, list[str]] | None = None,
        parsers: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        if vlm_client is not None:
            self.llm = vlm_client
        self.vlm = self.llm
        self.kb = knowledge_base or load_knowledge_base()
        self.parsers = parsers or {
            "classifier": ScreenshotClassifier(self.vlm),
            "combat_hud": CombatHUDParser(self.vlm),
            "inventory": InventoryParser(self.vlm),
            "skill_tree": SkillTreeParser(self.vlm),
        }
        self.tools = self._register_tools()

    def _register_tools(self) -> list[Tool]:
        return [
            _FunctionTool(
                "classify_screenshot",
                "判断截图属于 combat_hud / inventory / skill_tree / save_screen / other",
                lambda image_bytes: self.parsers["classifier"].classify(image_bytes),
            ),
            _FunctionTool(
                "parse_combat_hud",
                "解析战斗 HUD 截图并提取玩家状态",
                lambda image_bytes: self.parsers["combat_hud"].extract(image_bytes),
            ),
            _FunctionTool(
                "parse_inventory",
                "解析背包/装备截图并提取玩家状态",
                lambda image_bytes: self.parsers["inventory"].extract(image_bytes),
            ),
            _FunctionTool(
                "parse_skill_tree",
                "解析技能树截图并提取玩家状态",
                lambda image_bytes: self.parsers["skill_tree"].extract(image_bytes),
            ),
            _FunctionTool(
                "validate_extraction",
                "用知识库校验截图提取字段",
                lambda payload: validate_extraction(payload, self.kb),
            ),
            _FunctionTool(
                "merge_to_profile",
                "把校验后的字段合并进 player profile",
                lambda payload, profile, source="screenshot": merge_to_profile(payload, profile, source),
            ),
        ]

    def execute(self, state: AgentState) -> AgentState:
        if state.screenshots():
            state = self._handle_screenshots(state)
        elif self._has_profile_signal_in_text(state.user_query):
            state = self._handle_conversational_update(state)
        else:
            self._trace(state, 0, "profile_context", "No new profile signal; keeping current player profile.")

        state.retrieved_docs = _filter_by_profile(state.retrieved_docs, state.player_profile)

        if self.config.get("spoiler", {}).get("enable", True):
            state.retrieved_docs = apply_spoiler_filter(
                state.retrieved_docs, state.player_profile.chapter
            )
        return state

    def _handle_screenshots(self, state: AgentState) -> AgentState:
        all_updates: list[dict[str, Any]] = []

        for index, screenshot in enumerate(state.screenshots()):
            screenshot_type = self.parsers["classifier"].classify(screenshot)
            parser = self.parsers.get(screenshot_type)
            if parser is None:
                self._trace(
                    state,
                    index,
                    "classify_screenshot",
                    f"Unknown or unsupported screenshot type: {screenshot_type}",
                )
                continue

            raw_extraction = parser.extract(screenshot)
            validated = validate_extraction(raw_extraction, self.kb)
            state.player_profile, updates = merge_to_profile(
                validated,
                state.player_profile,
                source=f"screenshot:{screenshot_type}",
            )
            all_updates.extend(updates)
            self._trace(
                state,
                index,
                f"parse_{screenshot_type}",
                str({"raw": raw_extraction, "validated": validated, "updates": updates}),
            )

        state.profile_updates = all_updates
        return state

    def _handle_conversational_update(self, state: AgentState) -> AgentState:
        extracted = self._extract_profile_from_text(state.user_query)
        if not extracted:
            self._trace(state, 0, "profile_text_update", "Detected profile signal but found no structured update.")
            return state

        validated = validate_extraction(extracted, self.kb)
        state.player_profile, updates = merge_to_profile(
            validated,
            state.player_profile,
            source="conversation",
        )
        state.profile_updates = updates
        self._trace(
            state,
            0,
            "profile_text_update",
            str({"raw": extracted, "validated": validated, "updates": updates}),
        )
        return state

    def _has_profile_signal_in_text(self, text: str) -> bool:
        keywords = [
            "我现在",
            "我用",
            "我带",
            "我装备",
            "我没装备",
            "我在第",
            "我的章节",
            "我的棍法",
            "技能点",
            "截图里",
            "识别错",
            "纠正",
        ]
        return any(keyword in text for keyword in keywords)

    def _extract_profile_from_text(self, text: str) -> dict[str, Any]:
        extracted: dict[str, Any] = {}

        chapter_match = re.search(r"第\s*([1-6])\s*章", text)
        if chapter_match:
            extracted["chapter"] = int(chapter_match.group(1))

        staff_match = re.search(r"棍法\s*(?:等级|lv\.?|LV\.?)?\s*([1-5])", text)
        if staff_match:
            extracted["staff_level"] = int(staff_match.group(1))

        if "闪身流" in text:
            extracted["build"] = "dodge"
        elif "棍反流" in text:
            extracted["build"] = "parry"
        elif "法术流" in text:
            extracted["build"] = "spell"
        elif "混合流" in text:
            extracted["build"] = "hybrid"

        for spirit in self.kb.get("all_spirits", []):
            if f"没装备{spirit}" in text or f"不是{spirit}" in text:
                if extracted.get("equipped_spirit") == spirit:
                    extracted.pop("equipped_spirit", None)
                continue
            if spirit in text and any(token in text for token in ["装备", "带的是", "带了", "用的是"]):
                extracted["equipped_spirit"] = spirit

        spells = [spell for spell in self.kb.get("all_spells", []) if spell in text]
        if spells:
            extracted["equipped_spells"] = spells
            extracted["unlocked_spells"] = spells

        skills = [skill for skill in self.kb.get("all_skills_tree", []) if skill in text]
        if skills:
            extracted["unlocked_skills"] = skills

        armors = [armor for armor in self.kb.get("all_armors", []) if armor in text]
        if armors:
            extracted["equipped_armor"] = armors

        return extracted


def _filter_by_profile(docs: list[Document], profile) -> list[Document]:
    """Remove docs that recommend items the player hasn't unlocked yet."""
    filtered = []
    owned_items = set(
        profile.unlocked_skills
        + profile.unlocked_spells
        + profile.unlocked_transformations
        + profile.equipped_spells
    )
    if profile.equipped_spirit:
        owned_items.add(profile.equipped_spirit)

    for doc in docs:
        blocked = False
        for item, gate in CHAPTER_GATES.items():
            if item not in doc.text:
                continue
            if item in owned_items:
                continue
            if profile.chapter < gate:
                blocked = True
                break
        if not blocked:
            filtered.append(doc)
    return filtered
