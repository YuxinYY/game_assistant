"""
SynthesisAgent: writes the final answer.
Enforces: every claim must cite a source, contested items must be flagged,
unsupported claims must be admitted as "不确定".
"""

from src.agents.base_agent import BaseAgent, Tool
from src.core.state import AgentState, Citation
from src.tools.spoiler_filter import apply_spoiler_filter


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    prompt_file = "synthesis_agent.txt"

    def _register_tools(self) -> list[Tool]:
        return []  # synthesis is a single LLM call, no tool loop needed

    def execute(self, state: AgentState) -> AgentState:
        prompt = self._load_prompt()
        context = _build_synthesis_context(state)
        messages = [{"role": "user", "content": context}]

        # TODO: call self.llm.complete(messages, system=prompt)
        # state.final_answer = self.llm.complete(messages, system=prompt)
        state.final_answer = "[TODO: synthesis LLM call not yet implemented]"

        state.citations = _extract_citations(state.retrieved_docs)
        return state


def _build_synthesis_context(state: AgentState) -> str:
    docs_text = "\n\n".join(
        f"[来源 {i+1}] {d.source} | {d.url}\n{d.text}"
        for i, d in enumerate(state.retrieved_docs)
    )
    consensus_text = ""
    if state.consensus_analysis:
        strategies = state.consensus_analysis.get("strategies", [])
        consensus_text = "\n".join(
            f"- {s['label']}: {s['source_count']} 个来源支持"
            + (" ⚠️ 存在争议" if s.get("is_contested") else "")
            for s in strategies
        )
        conflicts = state.consensus_analysis.get("conflicts", [])
        if conflicts:
            consensus_text += "\n\n争议点:\n" + "\n".join(
                f"- {c['topic']}: 支持({len(c['pro'])}) vs 反对({len(c['con'])})"
                for c in conflicts
            )

    return (
        f"玩家状态:\n{state.player_profile.to_context_string()}\n\n"
        f"用户问题:\n{state.user_query}\n\n"
        f"检索到的内容:\n{docs_text}\n\n"
        f"共识分析:\n{consensus_text or '无'}"
    )


def _extract_citations(docs) -> list[Citation]:
    seen = set()
    citations = []
    for doc in docs:
        if doc.url not in seen:
            seen.add(doc.url)
            citations.append(Citation(
                source=doc.source,
                url=doc.url,
                excerpt=doc.text[:120] + "…",
                author=doc.metadata.get("author", ""),
            ))
    return citations
