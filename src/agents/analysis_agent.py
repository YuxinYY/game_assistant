"""
AnalysisAgent: aggregates retrieved docs into consensus/conflict structure.
Produces state.consensus_analysis used by SynthesisAgent for honest uncertainty.
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState
from src.tools.consensus import count_source_consensus, detect_conflicts


class _ConsensusCount(Tool):
    name = "consensus_count"
    description = "统计 docs 中各策略建议的出现频次（按来源去重）"

    def __call__(self, docs: list, topic: str) -> dict:
        return count_source_consensus(docs, topic)


class _ConflictDetect(Tool):
    name = "conflict_detect"
    description = "检测 docs 中互相矛盾的观点"

    def __call__(self, docs: list) -> list[dict]:
        return detect_conflicts(docs)


class AnalysisAgent(BaseAgent):
    name = "analysis_agent"
    prompt_file = "analysis_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return [_ConsensusCount(), _ConflictDetect()]

    def execute(self, state: AgentState) -> AgentState:
        docs = state.retrieved_docs
        if not docs:
            state.consensus_analysis = {"strategies": [], "conflicts": []}
            return state

        self._strategies = None
        self._conflicts = None
        initial_context = (
            "目标: 统计 retrieved_docs 中的共识策略，并识别冲突观点。\n"
            "先做共识统计，再补冲突检测。"
        )
        return self.react_loop(state, initial_context)

    def _apply_tool_result(self, state: AgentState, action_name: str, action_args: dict, tool_result) -> None:
        if action_name == "consensus_count" and isinstance(tool_result, list):
            self._strategies = tool_result
        elif action_name == "conflict_detect" and isinstance(tool_result, list):
            self._conflicts = tool_result

        state.consensus_analysis = {
            "strategies": self._strategies or [],
            "conflicts": self._conflicts or [],
        }

    def _fallback_decide(self, context: str, state: AgentState) -> tuple[str, str, dict]:
        if self._strategies is None:
            return (
                "先统计各策略的来源共识",
                "consensus_count",
                {"docs": state.retrieved_docs, "topic": state.user_query},
            )
        if self._conflicts is None:
            return "再检测互相矛盾的观点", "conflict_detect", {"docs": state.retrieved_docs}
        return "共识分析已完成", "FINISH", {}
