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
from src.utils.language import preferred_response_language


_UNSET = object()


class Tool:
    """Lightweight wrapper so agents can declare their tool roster declaratively."""
    name: str
    description: str
    description_en: str = ""

    def __call__(self, **kwargs):
        raise NotImplementedError


class BaseAgent(ABC):
    name: str = "base"
    prompt_file: str = ""   # filename under src/llm/prompts/
    prompt_file_en: str = ""

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
        observation_label = "Observation" if self._is_english(state) else "观察"
        for step in range(self.max_iterations):
            thought, action_name, action_args = self._decide(context, state)
            if action_name == "FINISH":
                self._trace(state, step, "FINISH", thought)
                break
            tool = self._get_tool(action_name)
            if tool is None:
                self._trace(
                    state,
                    step,
                    self._localize(
                        state,
                        f"错误：未知工具 '{action_name}'",
                        f"ERROR: unknown tool '{action_name}'",
                    ),
                    "",
                )
                break
            safe_args = self._sanitize_tool_args(tool, action_args)
            safe_args = self._bind_tool_args(tool, safe_args, state)
            try:
                tool_result = tool(**safe_args)
            except Exception as exc:
                tool_result = {"error": str(exc)}
            self._apply_tool_result(state, action_name, safe_args, tool_result)
            observation = self._summarize_tool_result(tool_result)
            context += f"\n{observation_label} [{step}]: {observation}"
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
        system = self._build_decision_system_prompt(state)

        try:
            response = self.llm.complete(messages, system=system)
            parsed = self._parse_decision_response(response, state)
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
        return self._localize(
            state,
            "无法可靠解析下一步，结束循环",
            "Could not reliably determine the next step, so the loop will stop.",
        ), "FINISH", {}

    def _trace(self, state: AgentState, step: int, action: str, observation: str):
        state.trace.append(TraceEvent(
            agent=self.name,
            step=step,
            action=action,
            observation=observation[:500],  # truncate long observations
        ))

    def _load_prompt(self, language: str = "zh") -> str:
        path = self._resolve_prompt_path(language)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _resolve_prompt_path(self, language: str) -> Path:
        base_dir = Path(__file__).parent.parent / "llm" / "prompts"
        if language == "en":
            candidates = []
            if self.prompt_file_en:
                candidates.append(base_dir / self.prompt_file_en)
            elif self.prompt_file:
                prompt_path = Path(self.prompt_file)
                candidates.append(base_dir / f"{prompt_path.stem}_en{prompt_path.suffix}")
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return base_dir / self.prompt_file

    def _build_decision_system_prompt(self, state: AgentState) -> str:
        language = self._language(state)
        tool_names = [tool.name for tool in self.tools]
        if language == "en":
            return (
                f"{self._load_prompt(language)}\n\n"
                "You are running a ReAct decision loop. Decide only the next action.\n"
                "Return exactly one JSON object. Do not output Markdown or code fences.\n"
                'Format: {"thought": "brief thought in English", "action": "tool name or FINISH", "action_args": {}}\n'
                f"Available actions: {tool_names} and FINISH.\n"
                "When there is already enough information to write back into state, choose FINISH."
            )
        return (
            f"{self._load_prompt(language)}\n\n"
            "你正在执行一个 ReAct 决策循环。每次只决定下一步动作。\n"
            "只输出一个 JSON 对象，不要输出 Markdown，不要输出代码块。\n"
            '固定格式: {"thought": "简短思考", "action": "工具名或FINISH", "action_args": {}}\n'
            f"可用 action: {tool_names} 和 FINISH。\n"
            "当已有足够信息可写入状态时，选择 FINISH。"
        )

    def _build_decision_prompt(self, context: str, state: AgentState) -> str:
        language = self._language(state)
        tool_lines = "\n".join(
            f"- {tool.name}: {self._tool_description(tool, language)}" for tool in self.tools
        ) or ("- No tools available" if language == "en" else "- 无工具")
        if language == "en":
            return (
                f"Tools:\n{tool_lines}\n\n"
                f"Shared state snapshot:\n{self._state_snapshot(state)}\n\n"
                f"Working memory:\n{context}"
            )
        return (
            f"工具列表:\n{tool_lines}\n\n"
            f"当前共享状态摘要:\n{self._state_snapshot(state)}\n\n"
            f"当前工作记忆:\n{context}"
        )

    def _state_snapshot(self, state: AgentState) -> str:
        language = self._language(state)
        doc_lines = []
        for doc in state.retrieved_docs[:3]:
            excerpt = self._truncate_text(doc.text, 80)
            doc_lines.append(
                f"- {doc.source} | {doc.url} | {doc.entity or ('Unknown entity' if language == 'en' else '未知实体')} | {excerpt}"
            )
        docs_text = "\n".join(doc_lines) if doc_lines else ("- No retrieved docs yet" if language == "en" else "- 暂无检索文档")
        if language == "en":
            return (
                f"User question: {state.user_query}\n"
                f"Player profile: {state.player_profile.to_context_string(language='en')}\n"
                f"Identified entities: {state.identified_entities or 'None'}\n"
                f"Current doc count: {len(state.retrieved_docs)}\n"
                f"Document preview:\n{docs_text}"
            )
        return (
            f"用户问题: {state.user_query}\n"
            f"玩家状态: {state.player_profile.to_context_string(language='zh')}\n"
            f"已识别实体: {state.identified_entities or '无'}\n"
            f"当前文档数: {len(state.retrieved_docs)}\n"
            f"文档预览:\n{docs_text}"
        )

    def _parse_decision_response(self, response: str, state: AgentState) -> tuple[str, str, dict] | None:
        cleaned = response.strip()
        if not cleaned:
            return None
        if cleaned.upper() == "FINISH":
            return self._localize(
                state,
                "模型认为信息已足够",
                "The model determined that the current evidence is sufficient.",
            ), "FINISH", {}

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

    def _bind_tool_args(self, tool: Tool, action_args: dict, state: AgentState) -> dict:
        try:
            signature = inspect.signature(tool.__call__)
        except (TypeError, ValueError):
            return action_args

        bound_args = dict(action_args)
        for name, param in signature.parameters.items():
            if name == "self" or param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue

            if name in bound_args:
                coerced = self._coerce_tool_arg(name, bound_args[name], state)
                if coerced is not _UNSET:
                    bound_args[name] = coerced
                continue

            if param.default is not inspect.Parameter.empty:
                continue

            default_value = self._default_tool_arg(name, state)
            if default_value is not _UNSET:
                bound_args[name] = default_value

        return bound_args

    def _coerce_tool_arg(self, name: str, value: Any, state: AgentState) -> Any:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        references = {
            "user_query": state.user_query,
            "state.user_query": state.user_query,
            "retrieved_docs": state.retrieved_docs,
            "state.retrieved_docs": state.retrieved_docs,
            "identified_entities": state.identified_entities,
            "state.identified_entities": state.identified_entities,
            "player_profile": state.player_profile,
            "state.player_profile": state.player_profile,
        }

        if normalized in references:
            return references[normalized]
        if name == "docs" and normalized in {"docs", "retrieved_docs", "state.retrieved_docs"}:
            return state.retrieved_docs
        if name == "query" and normalized in {"query", "user_query", "state.user_query"}:
            return state.user_query
        if name == "topic" and normalized in {"topic", "user_query", "state.user_query"}:
            return state.user_query
        if name == "entities" and normalized in {"entities", "identified_entities", "state.identified_entities"}:
            return state.identified_entities
        return value

    def _default_tool_arg(self, name: str, state: AgentState) -> Any:
        if name == "query":
            return state.user_query
        if name == "topic":
            return state.user_query
        if name == "docs":
            return state.retrieved_docs
        if name == "entities":
            return state.identified_entities
        if name == "profile":
            return state.player_profile
        if name == "entity" and len(state.identified_entities) == 1:
            return state.identified_entities[0]
        return _UNSET

    def _language(self, state: AgentState) -> str:
        return preferred_response_language(state.user_query)

    def _is_english(self, state: AgentState) -> bool:
        return self._language(state) == "en"

    def _localize(self, state: AgentState, zh_text: str, en_text: str) -> str:
        return en_text if self._is_english(state) else zh_text

    @staticmethod
    def _tool_description(tool: Tool, language: str) -> str:
        if language == "en" and getattr(tool, "description_en", ""):
            return tool.description_en
        return tool.description

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"
