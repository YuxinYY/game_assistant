"""
Screenshot parser: calls a VLM (Claude) to extract player build info from a game screenshot.
Returns a dict that ProfileAgent uses to update PlayerProfile.
"""

import base64
from typing import Optional


def parse_screenshot(image_bytes: bytes, llm_client=None) -> dict:
    """
    Send screenshot to Claude vision and extract:
      - chapter (inferred from location/UI elements)
      - build type (from equipped skills/staff)
      - staff_level
      - unlocked_skills, unlocked_spells, unlocked_transformations

    Returns partial dict; keys absent if not detectable from image.
    """
    if llm_client is None:
        # Lazy import to avoid circular deps
        import anthropic
        import os
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    else:
        client = llm_client

    b64 = base64.standard_b64encode(image_bytes).decode()

    # TODO: implement actual VLM call
    # response = client.messages.create(
    #     model="claude-opus-4-7",
    #     max_tokens=512,
    #     messages=[{
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
    #             {"type": "text", "text": SCREENSHOT_PARSE_PROMPT},
    #         ]
    #     }]
    # )
    # return _parse_vlm_response(response.content[0].text)
    raise NotImplementedError("VLM screenshot parsing not yet implemented")


SCREENSHOT_PARSE_PROMPT = """
Look at this Black Myth: Wukong screenshot and extract:
1. Current chapter (1-6) if visible
2. Build type: dodge/parry/spell/hybrid based on equipped skills
3. Staff mastery level if visible
4. List of unlocked skills/spells/transformations visible in the skill tree

Respond in JSON:
{"chapter": <int or null>, "build": <str or null>, "staff_level": <int or null>,
 "unlocked_skills": [...], "unlocked_spells": [...], "unlocked_transformations": [...]}
"""
