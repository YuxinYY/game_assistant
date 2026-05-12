"""
ProfileAgent: the single owner of player-state updates and screenshot parsing.
"""

from __future__ import annotations

from pathlib import Path
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
from src.tools.search import infer_wiki_entity
from src.utils.language import preferred_response_language

PROFILE_SIGNAL_KEYWORDS = (
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
    "已解锁",
    "当前技能",
    "当前法术",
    "当前变身",
    "unlocked skills",
    "unlocked spells",
    "unlocked transformations",
    "unlocked transformers",
    "my skills",
    "my spells",
    "my transformations",
    "my build",
    "i use",
    "i'm using",
    "i am using",
    "i have unlocked",
)

PROFILE_SIGNAL_PATTERNS = (
    re.compile(r"\bchapter\s*[1-6]\b", re.IGNORECASE),
    re.compile(r"\b(?:dodge|parry|spell|hybrid)\s+build\b", re.IGNORECASE),
    re.compile(r"\b(?:skills?|spells?|transformations?|transformers?)\s*:", re.IGNORECASE),
    re.compile(r"(?:已解锁|当前|目前).{0,4}(?:技能|法术|变身|化身)", re.IGNORECASE),
)

_EXPLICIT_LIST_FLAG_FIELDS = {
    "unlocked_skills": "skills_explicit",
    "unlocked_spells": "spells_explicit",
    "unlocked_transformations": "transformations_explicit",
}

_BUILD_PATTERNS = (
    (re.compile(r"\b(?:dodge|evasive?)\s+build\b", re.IGNORECASE), "dodge"),
    (re.compile(r"\bparry\s+build\b", re.IGNORECASE), "parry"),
    (re.compile(r"\bspell\s+build\b", re.IGNORECASE), "spell"),
    (re.compile(r"\bhybrid\s+build\b", re.IGNORECASE), "hybrid"),
)

_SKILL_SECTION_PATTERNS = (
    re.compile(r"(?:已解锁(?:的)?|当前(?:的)?|目前(?:的)?|现在(?:的)?)?技能(?:点)?(?:有|是|为|都在|点了)?\s*[:：]?\s*([^\n。！？!?;；]+)", re.IGNORECASE),
    re.compile(r"(?:unlocked|current(?:ly)?(?:\s+unlocked)?|my)\s+skills?\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
    re.compile(r"(?:i have|i've)\s+(?:currently\s+)?(?:unlocked\s+)?skills?\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
)

_SPELL_SECTION_PATTERNS = (
    re.compile(r"(?:已解锁(?:的)?|当前(?:的)?|目前(?:的)?|现在(?:的)?)?法术(?:有|是|为)?\s*[:：]?\s*([^\n。！？!?;；]+)", re.IGNORECASE),
    re.compile(r"(?:unlocked|current(?:ly)?(?:\s+unlocked)?|my)\s+spells?\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
    re.compile(r"(?:i have|i've)\s+(?:currently\s+)?(?:unlocked\s+)?spells?\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
)

_TRANSFORMATION_SECTION_PATTERNS = (
    re.compile(r"(?:已解锁(?:的)?|当前(?:的)?|目前(?:的)?|现在(?:的)?)?(?:变身|化身)(?:有|是|为)?\s*[:：]?\s*([^\n。！？!?;；]+)", re.IGNORECASE),
    re.compile(r"(?:unlocked|current(?:ly)?(?:\s+unlocked)?|my)\s+(?:transformations?|transformers?)\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
    re.compile(r"(?:i have|i've)\s+(?:currently\s+)?(?:unlocked\s+)?(?:transformations?|transformers?)\s*(?:are|:|=)\s*([^\n.!?;]+)", re.IGNORECASE),
)


def has_profile_signal_in_text(text: str) -> bool:
    content = text or ""
    return any(keyword in content for keyword in PROFILE_SIGNAL_KEYWORDS) or any(
        pattern.search(content) for pattern in PROFILE_SIGNAL_PATTERNS
    )


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
    visual_entity_prompt_file = Path(__file__).resolve().parents[1] / "llm" / "prompts" / "profile" / "visible_entity.txt"

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
            if not self._supports_vision():
                self._trace(
                    state,
                    0,
                    "vision_unavailable",
                    self._localized(
                        state,
                        "当前配置的 LLM/VLM 客户端不支持截图解析。",
                        "Configured LLM/VLM client does not support screenshot parsing.",
                    ),
                )
                return state
            self._trace(state, 0, "vision_context", str(self._vision_context()))
            state = self._handle_screenshots(state)
        elif self._has_profile_signal_in_text(state.user_query):
            state = self._handle_conversational_update(state)
        else:
            self._trace(
                state,
                0,
                "profile_context",
                self._localized(
                    state,
                    "没有检测到新的玩家状态信号，沿用当前玩家档案。",
                    "No new profile signal; keeping the current player profile.",
                ),
            )

        if not state.identified_entities and getattr(state.player_profile, "current_boss", None):
            profile_entity = infer_wiki_entity(state.player_profile.current_boss)
            if profile_entity:
                state.identified_entities = _merge_entities(state.identified_entities, [profile_entity])
                self._trace(state, 0, "profile_current_boss", profile_entity)

        state.retrieved_docs = _filter_by_profile(state.retrieved_docs, state.player_profile)

        if self.config.get("spoiler", {}).get("enable", True) and state.player_profile.chapter is not None:
            state.retrieved_docs = apply_spoiler_filter(
                state.retrieved_docs, state.player_profile.chapter
            )
        return state

    def _handle_screenshots(self, state: AgentState) -> AgentState:
        all_updates: list[dict[str, Any]] = []

        for index, screenshot in enumerate(state.screenshots()):
            try:
                screenshot_type = self.parsers["classifier"].classify(screenshot)
            except Exception as exc:
                self._trace(state, index, "classify_screenshot_error", str(exc))
                continue
            parser = self.parsers.get(screenshot_type)
            if parser is None:
                screenshot_entity = self._identify_visual_entity(screenshot)
                if screenshot_entity:
                    state.identified_entities = _merge_entities(state.identified_entities, [screenshot_entity])
                self._trace(
                    state,
                    index,
                    "classify_screenshot",
                    str(
                        {
                            "screenshot_type": screenshot_type,
                            "identified_entity": screenshot_entity,
                            "message": self._localized(
                                state,
                                "未知或暂不支持的截图类型。",
                                "Unknown or unsupported screenshot type.",
                            ),
                        }
                    ),
                )
                continue

            try:
                raw_extraction = parser.extract(screenshot)
            except Exception as exc:
                self._trace(state, index, f"parse_{screenshot_type}_error", str(exc))
                continue

            screenshot_entity = self._resolve_screenshot_entity(raw_extraction)
            if not screenshot_entity:
                screenshot_entity = self._identify_visual_entity(screenshot)
            if screenshot_entity:
                state.identified_entities = _merge_entities(state.identified_entities, [screenshot_entity])

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
                str(
                    {
                        "raw": raw_extraction,
                        "validated": validated,
                        "identified_entity": screenshot_entity,
                        "updates": updates,
                    }
                ),
            )

        state.profile_updates = all_updates
        return state

    def _supports_vision(self) -> bool:
        supports = getattr(self.vlm, "supports_vision", None)
        if callable(supports):
            try:
                return bool(supports())
            except Exception:
                return False
        return hasattr(self.vlm, "vision_json")

    def _vision_context(self) -> dict[str, Any]:
        return {
            "vision_provider": getattr(self.vlm, "vision_provider", None),
            "vision_model": getattr(self.vlm, "vision_model", None),
            "supports_vision": self._supports_vision(),
        }

    def _handle_conversational_update(self, state: AgentState) -> AgentState:
        extracted, explicit_fields = self._extract_profile_from_text(state.user_query)
        if not extracted:
            self._trace(
                state,
                0,
                "profile_text_update",
                self._localized(
                    state,
                    "检测到了玩家状态信号，但没有提取到结构化更新。",
                    "Detected profile signal but found no structured update.",
                ),
            )
            return state

        validated = validate_extraction(extracted, self.kb)
        merge_payload = {
            key: value
            for key, value in validated.items()
            if key not in explicit_fields
        }
        state.player_profile, updates = merge_to_profile(
            merge_payload,
            state.player_profile,
            source="conversation",
        )
        updates.extend(
            self._apply_explicit_list_updates(
                state.player_profile,
                validated,
                explicit_fields,
            )
        )
        state.profile_updates = updates
        self._trace(
            state,
            0,
            "profile_text_update",
            str({"raw": extracted, "validated": validated, "explicit_fields": sorted(explicit_fields), "updates": updates}),
        )
        return state

    def _has_profile_signal_in_text(self, text: str) -> bool:
        return has_profile_signal_in_text(text)

    def _extract_profile_from_text(self, text: str) -> tuple[dict[str, Any], set[str]]:
        extracted: dict[str, Any] = {}
        explicit_fields: set[str] = set()

        text = text or ""
        lowered = text.casefold()

        chapter_match = re.search(r"第\s*([1-6])\s*章", text) or re.search(r"\bchapter\s*([1-6])\b", text, re.IGNORECASE)
        if chapter_match:
            extracted["chapter"] = int(chapter_match.group(1))

        staff_match = re.search(r"棍法\s*(?:等级|lv\.?|LV\.?)?\s*([1-5])", text) or re.search(
            r"\b(?:staff(?:\s+level)?|staff\s*lv\.?|stance\s*lv\.?)\s*([1-5])\b",
            text,
            re.IGNORECASE,
        )
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
        else:
            for pattern, build in _BUILD_PATTERNS:
                if pattern.search(text):
                    extracted["build"] = build
                    break

        explicit_skills = _extract_labeled_items(text, _SKILL_SECTION_PATTERNS)
        if explicit_skills is not None:
            extracted["unlocked_skills"] = explicit_skills
            explicit_fields.add("unlocked_skills")

        explicit_spells = _extract_labeled_items(text, _SPELL_SECTION_PATTERNS)
        if explicit_spells is not None:
            extracted["unlocked_spells"] = explicit_spells
            explicit_fields.add("unlocked_spells")

        explicit_transformations = _extract_labeled_items(text, _TRANSFORMATION_SECTION_PATTERNS)
        if explicit_transformations is not None:
            extracted["unlocked_transformations"] = explicit_transformations
            explicit_fields.add("unlocked_transformations")

        for spirit in self.kb.get("all_spirits", []):
            if f"没装备{spirit}" in text or f"不是{spirit}" in text:
                if extracted.get("equipped_spirit") == spirit:
                    extracted.pop("equipped_spirit", None)
                continue
            if spirit in text and any(token in lowered for token in ["装备", "带的是", "带了", "用的是", "equipped", "using", "i use", "i'm using"]):
                extracted["equipped_spirit"] = spirit

        spells = [spell for spell in self.kb.get("all_spells", []) if spell in text]
        if spells and any(token in lowered for token in ["用的是", "带的是", "带了", "装备法术", "法术是", "using", "i use", "i'm using", "equipped spell"]):
            extracted["equipped_spells"] = spells
            if "unlocked_spells" not in explicit_fields:
                extracted["unlocked_spells"] = spells

        skills = [skill for skill in self.kb.get("all_skills_tree", []) if skill in text]
        if skills and "unlocked_skills" not in explicit_fields and any(
            token in lowered for token in ["技能", "技能点", "点了", "加点", "skills", "skill points"]
        ):
            extracted["unlocked_skills"] = skills

        armors = [armor for armor in self.kb.get("all_armors", []) if armor in text]
        if armors and any(token in lowered for token in ["装备", "带的是", "穿的是", "armor", "equipped"]):
            extracted["equipped_armor"] = armors

        return extracted, explicit_fields

    def _apply_explicit_list_updates(
        self,
        profile,
        validated: dict[str, Any],
        explicit_fields: set[str],
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        for field in sorted(explicit_fields):
            new_value = list(validated.get(field, []))
            old_value = list(getattr(profile, field, []) or [])
            flag_field = _EXPLICIT_LIST_FLAG_FIELDS.get(field)
            if flag_field:
                setattr(profile, flag_field, True)
            if old_value == new_value:
                continue
            setattr(profile, field, new_value)
            updates.append(
                {
                    "field": field,
                    "old_value": old_value,
                    "new_value": new_value,
                    "source": "conversation",
                    "confidence": validated.get("confidence"),
                }
            )
        return updates

    def _resolve_screenshot_entity(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        candidates = [
            str(payload.get("current_boss") or "").strip(),
            str(payload.get("boss_name") or "").strip(),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            entity = infer_wiki_entity(candidate)
            if entity:
                return entity
        return ""

    def _identify_visual_entity(self, image_bytes: bytes) -> str:
        if not self._supports_vision():
            return ""
        try:
            prompt = self.visual_entity_prompt_file.read_text(encoding="utf-8")
            payload = self.vlm.vision_json(image_bytes=image_bytes, prompt=prompt)
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        candidate = str(payload.get("visible_entity") or "").strip()
        if not candidate:
            return ""
        return infer_wiki_entity(candidate)

    def _localized(self, state: AgentState, zh_text: str, en_text: str) -> str:
        return en_text if preferred_response_language(state.user_query) == "en" else zh_text


def _filter_by_profile(docs: list[Document], profile) -> list[Document]:
    """Remove docs that recommend items the player hasn't unlocked yet."""
    if profile.chapter is None:
        return list(docs)

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


def _merge_entities(existing: list[str], new_entities: list[str]) -> list[str]:
    merged = list(existing)
    for entity in new_entities:
        if entity and entity not in merged:
            merged.append(entity)
    return merged


def _extract_labeled_items(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str] | None:
    for pattern in patterns:
        match = pattern.search(text or "")
        if not match:
            continue
        return _parse_profile_list_segment(match.group(1))
    return None


def _parse_profile_list_segment(segment: str) -> list[str]:
    cleaned = (segment or "").strip().strip("。.!?；;，,")
    if not cleaned:
        return []
    if re.fullmatch(r"(?:none|no(?:ne)?|n/?a|无|没有|暂无|未解锁|空)", cleaned, re.IGNORECASE):
        return []
    normalized = re.sub(r"\b(?:and|only|just|currently)\b", ",", cleaned, flags=re.IGNORECASE)
    normalized = normalized.replace("以及", ",").replace("和", ",")
    values = [part.strip().strip("。.!?；;，,") for part in re.split(r"[,，、/；;]", normalized)]
    return [value for value in values if value]
