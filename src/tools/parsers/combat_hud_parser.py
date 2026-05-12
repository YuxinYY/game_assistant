from __future__ import annotations

from typing import List, Optional

try:
    from pydantic import BaseModel, field_validator
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

from src.tools.parsers import BaseParser

if _PYDANTIC:
    class CombatHUDSchema(BaseModel):
        chapter: Optional[int] = None
        build: Optional[str] = None
        staff_level: Optional[int] = None
        equipped_spirit: Optional[str] = None
        equipped_spells: List[str] = []
        unlocked_transformations: List[str] = []
        confidence: float = 0.0

        @field_validator("chapter")
        @classmethod
        def validate_chapter(cls, v):
            if v is not None and not (1 <= v <= 6):
                return None
            return v

        @field_validator("staff_level")
        @classmethod
        def validate_staff_level(cls, v):
            if v is not None and not (1 <= v <= 5):
                return None
            return v

        @field_validator("build")
        @classmethod
        def validate_build(cls, v):
            if v not in {"dodge", "parry", "spell", "hybrid", None}:
                return None
            return v
else:
    CombatHUDSchema = None  # type: ignore[assignment,misc]


class CombatHUDParser(BaseParser):
    screenshot_type = "combat_hud"
    prompt_path = "profile/combat_hud.txt"
    schema = CombatHUDSchema
