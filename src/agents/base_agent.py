"""
Abstract base for all agents. Implements the ReAct loop once;
subclasses only need to declare their tools and prompt file.
"""

from abc import ABC, abstractmethod
import inspect
import json
import re
from typing import Any, Optional
from pathlib import Path
from src.core.state import AgentState, Document, TraceEvent
from src.llm.client import LLMClient


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
            safe_args = self._sanitize_tool_args(tool, action_args)
            try:
                tool_result = tool(**safe_args)
            except Exception as exc:
                tool_result = {"error": str(exc)}
            self._apply_tool_result(state, action_name, safe_args, tool_result)
            observation = self._summarize_tool_result(tool_result)
            context += f"\nObservation [{step}]: {observation}"
            self._trace(state, step, f"{action_name}({safe_args})", observation)
        return state

    def _decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        """
        Ask LLM which tool to call next.
        Returns (thought, action_name, action_args).
        action_name == "FINISH" signals loop exit.
        """
        messages = [
            {
                "role": "user",
                "content": self._build_decision_prompt(context, state),
            }
        ]
        system = self._build_decision_system_prompt()

        try:
            response = self.llm.complete(messages, system=system)
            parsed = self._parse_decision_response(response)
            if parsed is not None:
                thought, action_name, action_args = parsed
                if action_name == "FINISH" or self._get_tool(action_name) is not None:
                    return thought, action_name, action_args
        except Exception:
            pass

        return self._fallback_decide(context, state)

    def _get_tool(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)

    def _apply_tool_result(self, state: AgentState, action_name: str, action_args: dict, tool_result: Any) -> None:
        """Optional hook for subclasses to write tool results into AgentState."""
        return None

    def _fallback_decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        return "无法可靠解析下一步，结束循环", "FINISH", {}

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

    def _build_decision_system_prompt(self) -> str:
        tool_names = [tool.name for tool in self.tools]
        return (
            f"{self._load_prompt()}\n\n"
            "你正在执行一个 ReAct 决策循环。每次只决定下一步动作。\n"
            "只输出一个 JSON 对象，不要输出 Markdown，不要输出代码块。\n"
            '固定格式: {"thought": "简短思考", "action": "工具名或FINISH", "action_args": {}}\n'
            f"可用 action: {tool_names} 和 FINISH。\n"
            "当已有足够信息可写入状态时，选择 FINISH。"
        )

    def _build_decision_prompt(self, context: str, state: AgentState) -> str:
        tool_lines = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ) or "- 无工具"
        return (
            f"工具列表:\n{tool_lines}\n\n"
            f"当前共享状态摘要:\n{self._state_snapshot(state)}\n\n"
            f"当前工作记忆:\n{context}"
        )

    def _state_snapshot(self, state: AgentState) -> str:
        doc_lines = []
        for doc in state.retrieved_docs[:3]:
            excerpt = self._truncate_text(doc.text, 80)
            doc_lines.append(f"- {doc.source} | {doc.url} | {doc.entity or '未知实体'} | {excerpt}")
        docs_text = "\n".join(doc_lines) if doc_lines else "- 暂无检索文档"
        return (
            f"用户问题: {state.user_query}\n"
            f"玩家状态: {state.player_profile.to_context_string()}\n"
            f"已识别实体: {state.identified_entities or '无'}\n"
            f"当前文档数: {len(state.retrieved_docs)}\n"
            f"文档预览:\n{docs_text}"
        )

    def _parse_decision_response(self, response: str) -> tuple[str, str, dict] | None:
        cleaned = response.strip()
        if not cleaned:
            return None
        if cleaned.upper() == "FINISH":
            return "模型认为信息已足够", "FINISH", {}

        payload = self._extract_json_object(cleaned)
        if not payload:
            return None

        thought = str(payload.get("thought") or "")
        action_name = str(payload.get("action") or "").strip()
        action_args = payload.get("action_args") or {}
        if not isinstance(action_args, dict) or not action_name:
            return None
        return thought, action_name, action_args

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

    @staticmethod
    def _sanitize_tool_args(tool: Tool, action_args: dict) -> dict:
        if not isinstance(action_args, dict):
            return {}
        try:
            signature = inspect.signature(tool.__call__)
        except (TypeError, ValueError):
            return action_args
        allowed = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in action_args.items() if key in allowed}

    def _summarize_tool_result(self, tool_result: Any) -> str:
        if isinstance(tool_result, dict):
            return self._truncate_text(json.dumps(tool_result, ensure_ascii=False), 500)
        if isinstance(tool_result, list):
            if tool_result and isinstance(tool_result[0], Document):
                preview = [
                    {
                        "source": doc.source,
                        "entity": doc.entity,
                        "url": doc.url,
                        "text": self._truncate_text(doc.text, 100),
                    }
                    for doc in tool_result[:3]
                ]
                return self._truncate_text(json.dumps(preview, ensure_ascii=False), 500)
            return self._truncate_text(json.dumps(tool_result[:5], ensure_ascii=False), 500)
        return self._truncate_text(str(tool_result), 500)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"
