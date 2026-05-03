"""
Backward-compatible wrapper around the new ProfileAgent parser stack.
"""

from __future__ import annotations

from src.tools.parsers import CombatHUDParser, InventoryParser, ScreenshotClassifier, SkillTreeParser
from src.tools.profile_ops import load_knowledge_base, validate_extraction


def parse_screenshot(image_bytes: bytes, llm_client=None) -> dict:
    if llm_client is None:
        raise NotImplementedError("A VLM-capable llm_client is required for screenshot parsing.")

    classifier = ScreenshotClassifier(llm_client)
    screenshot_type = classifier.classify(image_bytes)
    parsers = {
        "combat_hud": CombatHUDParser(llm_client),
        "inventory": InventoryParser(llm_client),
        "skill_tree": SkillTreeParser(llm_client),
    }
    parser = parsers.get(screenshot_type)
    if parser is None:
        return {"screenshot_type": screenshot_type}

    raw = parser.extract(image_bytes)
    return validate_extraction(raw, load_knowledge_base())
