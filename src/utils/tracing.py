"""
Lightweight tracer. All agents/tools call Tracer.log() so every
LLM call, tool call, and routing decision is recorded.
Trace is stored in AgentState.trace AND optionally written to a JSONL file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("wukong.trace")


class Tracer:
    def __init__(self, log_to_file: bool = False, path: str = "logs/trace.jsonl"):
        self.log_to_file = log_to_file
        self.path = Path(path)
        if log_to_file:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, **kwargs):
        entry = {
            "ts": datetime.now().isoformat(),
            "msg": message,
            **kwargs,
        }
        logger.info(message)
        if self.log_to_file:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_agent_step(self, agent: str, step: int, action: str, observation: str):
        self.log(
            f"[{agent}] step {step}: {action}",
            agent=agent,
            step=step,
            action=action,
            observation=observation[:300],
        )

    def log_llm_call(self, agent: str, model: str, prompt_tokens: int = 0):
        self.log(f"[{agent}] LLM call → {model}", model=model, tokens=prompt_tokens)
