"""
Bounded execution planner.

The planner only selects from the existing workflow steps, so the system keeps
its current stability characteristics while gaining explicit runtime goals,
evidence-gap tracking, and conditional step skipping.
"""

from __future__ import annotations

from src.agents.profile_agent import has_profile_signal_in_text
from src.core.state import AgentState, ExecutionPlan, PlanStep
from src.tools.search import has_indexed_source_documents, infer_wiki_entity
from src.utils.language import detect_query_language


class ExecutionPlanner:
    def __init__(self, config: dict):
        self.config = config

    def build_plan(self, state: AgentState, workflow_name: str, agent_sequence: list[type]) -> ExecutionPlan:
        goals = self._infer_goals(state, workflow_name)
        evidence_gaps = self._infer_evidence_gaps(state, workflow_name)
        steps = [
            PlanStep(agent=agent_class.__name__, goal=self._step_goal(agent_class.__name__, workflow_name, goals))
            for agent_class in agent_sequence
        ]
        plan = ExecutionPlan(
            workflow=workflow_name,
            goals=goals,
            evidence_gaps=evidence_gaps,
            stop_conditions=self._stop_conditions(workflow_name),
            steps=steps,
        )
        state.execution_plan = plan
        state.evidence_gaps = list(evidence_gaps)
        state.completed_steps = []
        state.skipped_steps = []
        state.need_user_clarification = any(
            gap in {"missing_entity", "missing_build_context"}
            for gap in evidence_gaps
        )
        state.stop_reason = None
        return plan

    def should_execute(self, state: AgentState, workflow_name: str, agent_class: type) -> tuple[bool, str]:
        agent_name = agent_class.__name__

        if agent_name == "ProfileAgent":
            if state.screenshots() or has_profile_signal_in_text(state.user_query):
                return True, ""
            return False, "No new screenshot or text-based profile signal was detected."

        if agent_name == "CommunityAgent":
            if workflow_name not in {"boss_strategy", "decision_making"}:
                return False, "Community retrieval is not part of the selected workflow."
            if not self._available_community_sources(state.user_query):
                return False, "No indexed community sources are available for the current query language."
            return True, ""

        if agent_name == "AnalysisAgent":
            unique_sources = {doc.source for doc in state.retrieved_docs}
            if len(state.retrieved_docs) < 2 or len(unique_sources) < 2:
                if state.consensus_analysis is None:
                    state.consensus_analysis = {"strategies": [], "conflicts": []}
                return False, "Consensus analysis needs at least two documents from two distinct sources."

        return True, ""

    def mark_step_completed(self, state: AgentState, agent_name: str) -> None:
        if agent_name not in state.completed_steps:
            state.completed_steps.append(agent_name)
        self._update_step_status(state, agent_name, status="completed", reason="")

    def mark_step_skipped(self, state: AgentState, agent_name: str, reason: str) -> None:
        state.skipped_steps.append({"agent": agent_name, "reason": reason})
        self._update_step_status(state, agent_name, status="skipped", reason=reason)

    def describe_plan(self, plan: ExecutionPlan | None) -> str:
        if plan is None:
            return "no plan"
        step_labels = [step.agent for step in plan.steps]
        return f"goals={plan.goals}; steps={step_labels}; evidence_gaps={plan.evidence_gaps}"

    def _update_step_status(self, state: AgentState, agent_name: str, status: str, reason: str) -> None:
        if state.execution_plan is None:
            return
        for step in state.execution_plan.steps:
            if step.agent == agent_name:
                step.status = status
                step.reason = reason
                break

    def _infer_goals(self, state: AgentState, workflow_name: str) -> list[str]:
        query = state.user_query or ""
        normalized_query = query.lower()
        goals: list[str] = []

        if workflow_name == "boss_strategy":
            goals.extend([
                "identify_entity",
                "collect_official_evidence",
                "collect_community_counterplay",
                "ground_answer_with_citations",
            ])
            if any(keyword in query for keyword in ["怎么躲", "闪避", "躲", "侧闪"]) or any(
                keyword in normalized_query for keyword in ["dodge", "avoid", "evade"]
            ):
                goals.append("resolve_dodge_timing")
            if any(keyword in query for keyword in ["怎么打", "反打", "后摇", "输出"]) or any(
                keyword in normalized_query for keyword in ["beat", "punish", "counter", "recovery"]
            ):
                goals.append("resolve_punish_window")
            if self._needs_build_specific_advice(state):
                goals.append("collect_build_specific_advice")

        elif workflow_name == "decision_making":
            goals.extend([
                "collect_official_evidence",
                "collect_community_counterplay",
                "compare_build_options",
                "ground_answer_with_citations",
            ])
            if self._needs_build_specific_advice(state):
                goals.append("collect_build_specific_advice")

        elif workflow_name == "navigation":
            goals.extend([
                "identify_entity",
                "collect_location_evidence",
                "ground_answer_with_citations",
            ])

        else:
            goals.extend([
                "identify_entity",
                "collect_exact_fact",
                "ground_answer_with_citations",
            ])
            if any(keyword in query for keyword in ["几个", "多少", "哪些", "叫什么"]) or any(
                keyword in normalized_query for keyword in ["how many", "what are", "which moves", "name the"]
            ):
                goals.append("enumerate_or_count_facts")

        return _dedupe(goals)

    def _infer_evidence_gaps(self, state: AgentState, workflow_name: str) -> list[str]:
        gaps: list[str] = []
        query = state.user_query or ""
        inferred_entity = state.identified_entities[0] if state.identified_entities else infer_wiki_entity(query)

        if workflow_name in {"boss_strategy", "navigation", "fact_lookup"} and not inferred_entity:
            gaps.append("missing_entity")

        if workflow_name in {"boss_strategy", "decision_making"} and state.player_profile.chapter is None:
            gaps.append("missing_chapter_context")

        if workflow_name == "decision_making" and state.player_profile.build is None:
            gaps.append("missing_build_context")

        if workflow_name in {"boss_strategy", "decision_making"} and not self._available_community_sources(query):
            gaps.append("limited_community_evidence")

        return _dedupe(gaps)

    def _step_goal(self, agent_name: str, workflow_name: str, plan_goals: list[str]) -> str:
        if agent_name == "ProfileAgent":
            return "Update player context only when there is new profile evidence."
        if agent_name == "WikiAgent":
            return "Resolve entities and gather official evidence from the local wiki index."
        if agent_name == "CommunityAgent":
            if "resolve_dodge_timing" in plan_goals:
                return "Search community sources for dodge timing and counterplay details."
            if "compare_build_options" in plan_goals:
                return "Search community sources for build comparisons and tradeoffs."
            return "Search community sources for grounded player strategy evidence."
        if agent_name == "AnalysisAgent":
            return "Aggregate cross-source consensus only when multi-source evidence exists."
        if agent_name == "SynthesisAgent":
            return "Produce the final grounded answer with citations and uncertainty handling."
        return f"Execute {agent_name} for workflow '{workflow_name}'."

    def _stop_conditions(self, workflow_name: str) -> list[str]:
        if workflow_name in {"navigation", "fact_lookup"}:
            return [
                "Stop after official evidence is synthesized.",
                "If no evidence is found, return a grounded no-results answer instead of guessing.",
            ]
        return [
            "Stop after the bounded workflow finishes.",
            "If community evidence is unavailable, continue with wiki-only synthesis instead of failing.",
        ]

    def _available_community_sources(self, query: str) -> list[str]:
        query_language = detect_query_language(query)
        if query_language == "en":
            return [
                source
                for source in ["reddit", "nga", "bilibili"]
                if has_indexed_source_documents(source, language="en")
            ]

        available_sources: list[str] = []
        if has_indexed_source_documents("nga"):
            available_sources.append("nga")
        if has_indexed_source_documents("bilibili"):
            available_sources.append("bilibili")
        if has_indexed_source_documents("reddit", language="en"):
            available_sources.append("reddit")
        return available_sources

    @staticmethod
    def _has_build_keyword(query: str) -> bool:
        normalized_query = query.lower()
        return any(keyword in query for keyword in ["流派", "技能", "法术", "配装"]) or any(
            keyword in normalized_query
            for keyword in ["build", "spell", "skill", "weapon", "staff", "armor"]
        )

    def _needs_build_specific_advice(self, state: AgentState) -> bool:
        if state.player_profile.build is not None:
            return True
        return self._has_build_keyword(state.user_query or "")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered