"""
Screenshot parsers used by ProfileAgent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BaseParser:
    screenshot_type = "other"
    prompt_path = ""

    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def extract(self, image_bytes: bytes) -> dict[str, Any]:
        prompt = self._load_prompt()
        if not _supports_vision(self.vlm):
            return {}
        try:
            payload = self.vlm.vision_json(image_bytes=image_bytes, prompt=prompt)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        payload.setdefault("screenshot_type", self.screenshot_type)
        return payload

    def _load_prompt(self) -> str:
        prompt_file = Path(__file__).resolve().parents[2] / "llm" / "prompts" / self.prompt_path
        return prompt_file.read_text(encoding="utf-8")


def _supports_vision(vlm_client) -> bool:
    supports = getattr(vlm_client, "supports_vision", None)
    if callable(supports):
        try:
            return bool(supports())
        except Exception:
            return False
    return hasattr(vlm_client, "vision_json")


from src.tools.parsers.classifier import ScreenshotClassifier
from src.tools.parsers.combat_hud_parser import CombatHUDParser
from src.tools.parsers.inventory_parser import InventoryParser
from src.tools.parsers.skill_tree_parser import SkillTreeParser

__all__ = [
    "BaseParser",
    "CombatHUDParser",
    "InventoryParser",
    "ScreenshotClassifier",
    "SkillTreeParser",
]
