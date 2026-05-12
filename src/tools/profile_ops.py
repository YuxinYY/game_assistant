"""
Validation and merge helpers for screenshot-derived player profile updates.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.core.state import PlayerProfile


VALUE_ALIASES = {
    "all_spells": {
        "immobilize": "定身术",
        "rock solid": "铜头铁臂",
        "ring of fire": "安身法",
        "cloud step": "聚形散气",
        "pluck of many": "身外身法",
        "spell binder": "禁字法",
    },
    "all_skills_tree": {
        "immobilize": "定身术",
        "rock solid": "铜头铁臂",
        "ring of fire": "安身法",
        "cloud step": "聚形散气",
        "pluck of many": "身外身法",
        "spell binder": "禁字法",
    },
}

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

KB_FILES = {
    "all_spells": "all_spells.json",
    "all_spirits": "all_spirits.json",
    "all_armors": "all_armors.json",
    "all_skills_tree": "all_skills_tree.json",
}

LIST_FIELDS = {
    "equipped_armor",
    "equipped_spells",
    "unlocked_skills",
    "unlocked_spells",
    "unlocked_transformations",
}

UNION_FIELDS = {
    "unlocked_skills",
    "unlocked_spells",
    "unlocked_transformations",
}

PROFILE_FIELDS = {
    "chapter",
    "current_boss",
    "build",
    "primary_stance",
    "staff_level",
    "equipped_spirit",
    "equipped_armor",
    "equipped_mysticism",
    "equipped_body",
    "equipped_strand",
    "equipped_transformation",
    "equipped_spells",
    "unlocked_skills",
    "unlocked_spells",
    "unlocked_transformations",
}


def load_knowledge_base(base_path: str | Path = "data/knowledge") -> dict[str, list[str]]:
    base_dir = Path(base_path)
    kb: dict[str, list[str]] = {}
    for key, filename in KB_FILES.items():
        path = base_dir / filename
        if not path.exists():
            kb[key] = []
            continue
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        kb[key] = payload if isinstance(payload, list) else []
    return kb


def validate_extraction(raw_extraction: dict[str, Any], kb: dict[str, list[str]]) -> dict[str, Any]:
    validated = deepcopy(raw_extraction)

    chapter = validated.get("chapter")
    if isinstance(chapter, int):
        if not 1 <= chapter <= 6:
            validated.pop("chapter", None)
    elif chapter is not None:
        validated.pop("chapter", None)

    staff_level = validated.get("staff_level")
    if isinstance(staff_level, int):
        if not 1 <= staff_level <= 5:
            validated.pop("staff_level", None)
    elif staff_level is not None:
        validated.pop("staff_level", None)

    build = validated.get("build")
    if build is not None and build not in {"dodge", "parry", "spell", "hybrid"}:
        validated.pop("build", None)

    primary_stance = validated.get("primary_stance")
    if primary_stance is not None and primary_stance not in {"smash", "pillar", "thrust"}:
        validated.pop("primary_stance", None)

    _validate_list(validated, "unlocked_spells", set(kb.get("all_spells", [])), VALUE_ALIASES.get("all_spells", {}))
    _validate_list(validated, "equipped_spells", set(kb.get("all_spells", [])), VALUE_ALIASES.get("all_spells", {}))
    _validate_scalar(validated, "equipped_spirit", set(kb.get("all_spirits", [])))
    _validate_list(validated, "equipped_armor", set(kb.get("all_armors", [])))
    _validate_list(validated, "unlocked_skills", set(kb.get("all_skills_tree", [])), VALUE_ALIASES.get("all_skills_tree", {}))
    _validate_list(validated, "unlocked_transformations", set(), TRANSFORMATION_ALIASES)

    return validated


def merge_to_profile(
    extracted: dict[str, Any],
    current_profile: PlayerProfile,
    source: str = "screenshot",
) -> tuple[PlayerProfile, list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []

    for field, value in extracted.items():
        if field not in PROFILE_FIELDS or value in (None, [], ""):
            continue

        old_value = getattr(current_profile, field, None)
        if field in LIST_FIELDS:
            if field in UNION_FIELDS:
                new_value = _merge_unique(old_value or [], value)
            else:
                new_value = list(value)
        else:
            new_value = value

        if old_value == new_value:
            continue

        setattr(current_profile, field, new_value)
        updates.append(
            {
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
                "source": source,
                "confidence": extracted.get("confidence"),
            }
        )

    return current_profile, updates


def _validate_list(
    payload: dict[str, Any],
    field: str,
    allowed_values: set[str],
    aliases: dict[str, str] | None = None,
) -> None:
    values = payload.get(field)
    if values is None:
        return
    if not isinstance(values, list):
        payload.pop(field, None)
        return
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        normalized_value = _normalize_alias(value, aliases)
        if normalized_value:
            normalized.append(normalized_value)
    if allowed_values:
        normalized = [value for value in normalized if value in allowed_values]
    payload[field] = _merge_unique([], normalized)


def _validate_scalar(payload: dict[str, Any], field: str, allowed_values: set[str]) -> None:
    value = payload.get(field)
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        payload.pop(field, None)
        return
    if allowed_values and value not in allowed_values:
        payload.pop(field, None)


def _normalize_alias(value: str, aliases: dict[str, str] | None) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    if not aliases:
        return stripped
    return aliases.get(stripped.casefold(), stripped)


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *incoming]:
        if value in seen:
            continue
        merged.append(value)
        seen.add(value)
    return merged
