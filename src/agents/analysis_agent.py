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
        """
        Build consensus_analysis dict from retrieved_docs.
        Structure written to state:
        {
          "strategies": [
            {"label": "侧向闪避", "source_count": 8, "sources": {...}, "is_contested": False},
            ...
          ],
          "conflicts": [{"topic": "棍反", "pro": [...], "con": [...]}]
        }
        """
        docs = state.retrieved_docs
        if not docs:
            state.consensus_analysis = {"strategies": [], "conflicts": []}
            return state

        strategies = count_source_consensus(docs, topic=state.user_query)
        conflicts = detect_conflicts(docs)
        state.consensus_analysis = {
            "strategies": strategies,
            "conflicts": conflicts,
        }
        return state
