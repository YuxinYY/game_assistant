from __future__ import annotations

from pathlib import Path


class ScreenshotClassifier:
    def __init__(self, vlm_client):
        self.vlm = vlm_client
        self.prompt_path = Path(__file__).resolve().parents[2] / "llm" / "prompts" / "profile" / "classifier.txt"

    def classify(self, image_bytes: bytes) -> str:
        if not hasattr(self.vlm, "vision_json"):
            return "other"
        prompt = self.prompt_path.read_text(encoding="utf-8")
        payload = self.vlm.vision_json(image_bytes=image_bytes, prompt=prompt)
        screenshot_type = payload.get("screenshot_type") if isinstance(payload, dict) else None
        if screenshot_type in {"combat_hud", "inventory", "skill_tree", "save_screen", "other"}:
            return screenshot_type
        return "other"
