"""
Abstract base for all agents. Implements the ReAct loop once;
subclasses only need to declare their tools and prompt file.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from src.core.state import AgentState, TraceEvent
from src.llm.client import LLMClient
from datetime import datetime


class Tool:
    """Lightweight wrapper so agents can declare their tool roster declaratively."""
    name: str
    description: str

    def __call__(self, **kwargs):
        raise NotImplementedError


class BaseAgent(ABC):
    name: str = "base"
    prompt_file: str = ""   # filename under src/llm/prompts/

    def __init__(self, config: dict):
        self.config = config
        self.llm = LLMClient(config)
        self.max_iterations = config.get("agents", {}).get("max_react_iterations", 3)
        self.tools: list[Tool] = self._register_tools()

    @abstractmethod
    def _register_tools(self) -> list[Tool]:
        """Return the list of Tool instances this agent can use."""
        ...

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Entry point called by the orchestrator. Must return updated state."""
        ...

    def react_loop(self, state: AgentState, initial_context: str) -> AgentState:
        """
        Generic ReAct: Think → Choose tool → Execute → Observe → repeat.
        Agents call this from execute() after building initial_context.
        """
        context = initial_context
        for step in range(self.max_iterations):
            thought, action_name, action_args = self._decide(context, state)
            if action_name == "FINISH":
                self._trace(state, step, "FINISH", thought)
                break
            tool = self._get_tool(action_name)
            if tool is None:
                self._trace(state, step, f"ERROR: unknown tool '{action_name}'", "")
                break
            observation = str(tool(**action_args))
            context += f"\nObservation [{step}]: {observation}"
            self._trace(state, step, f"{action_name}({action_args})", observation)
        return state

    def _decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        """
        Ask LLM which tool to call next.
        Returns (thought, action_name, action_args).
        action_name == "FINISH" signals loop exit.
        TODO: implement full JSON-structured tool-use call.
        """
        raise NotImplementedError

    def _get_tool(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)

    def _trace(self, state: AgentState, step: int, action: str, observation: str):
        state.trace.append(TraceEvent(
            agent=self.name,
            step=step,
            action=action,
            observation=observation[:500],  # truncate long observations
        ))

    def _load_prompt(self) -> str:
        path = Path(__file__).parent.parent / "llm" / "prompts" / self.prompt_file
        return path.read_text(encoding="utf-8") if path.exists() else ""
