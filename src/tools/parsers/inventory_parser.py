from __future__ import annotations

from typing import List, Optional

try:
    from pydantic import BaseModel, field_validator
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

from src.tools.parsers import BaseParser

if _PYDANTIC:
    class InventorySchema(BaseModel):
        chapter: Optional[int] = None
        equipped_spirit: Optional[str] = None
        equipped_armor: List[str] = []
        equipped_spells: List[str] = []
        confidence: float = 0.0

        @field_validator("chapter")
        @classmethod
        def validate_chapter(cls, v):
            if v is not None and not (1 <= v <= 6):
                return None
            return v
else:
    InventorySchema = None  # type: ignore[assignment,misc]


class InventoryParser(BaseParser):
    screenshot_type = "inventory"
    prompt_path = "profile/inventory.txt"
    schema = InventorySchema
