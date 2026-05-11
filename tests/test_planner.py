"""
Tests for bounded execution planning and conditional step execution.
"""

from unittest.mock import patch

from src.agents.analysis_agent import AnalysisAgent
from src.agents.community_agent import CommunityAgent
from src.core.planner import ExecutionPlanner
from src.core.state import AgentState, Document, PlayerProfile
from src.core.workflows import build_workflows


DUMMY_CONFIG = {
    "llm": {"model": "claude-sonnet-4-7", "temperature": 0.3},
    "agents": {"max_react_iterations": 3},
}


def make_state(query: str, chapter=None, build=None) -> AgentState:
    return AgentState(user_query=query, player_profile=PlayerProfile(chapter=chapter, build=build))


def make_doc(text: str, source: str = "wiki") -> Document:
    return Document(text=text, source=source, url=f"http://{source}/1")


def test_planner_builds_goal_driven_plan_for_boss_strategy_query():
    planner = ExecutionPlanner(DUMMY_CONFIG)
    workflows = build_workflows()
    state = make_state("虎先锋那个招怎么躲？", chapter=2)

    with patch("src.core.planner.has_indexed_source_documents", return_value=True):
        plan = planner.build_plan(state, "boss_strategy", workflows["boss_strategy"])

    assert plan.workflow == "boss_strategy"
    assert "collect_official_evidence" in plan.goals
    assert "collect_community_counterplay" in plan.goals
    assert "resolve_dodge_timing" in plan.goals
    assert [step.agent for step in plan.steps][:2] == ["ProfileAgent", "WikiAgent"]


def test_planner_marks_missing_build_context_for_decision_query_without_profile():
    planner = ExecutionPlanner(DUMMY_CONFIG)
    workflows = build_workflows()
    state = make_state("Which build is better for Yellow Wind Sage?", chapter=None, build=None)

    with patch("src.core.planner.has_indexed_source_documents", return_value=True):
        plan = planner.build_plan(state, "decision_making", workflows["decision_making"])

    assert "compare_build_options" in plan.goals
    assert "missing_build_context" in plan.evidence_gaps
    assert "missing_chapter_context" in plan.evidence_gaps


def test_planner_skips_community_agent_when_no_indexed_sources_are_available():
    planner = ExecutionPlanner(DUMMY_CONFIG)
    state = make_state("How do I beat Tiger Vanguard?", chapter=2)

    with patch("src.core.planner.has_indexed_source_documents", return_value=False):
        should_run, reason = planner.should_execute(state, "boss_strategy", CommunityAgent)

    assert should_run is False
    assert "No indexed community sources" in reason


def test_planner_skips_analysis_when_only_single_source_evidence_exists():
    planner = ExecutionPlanner(DUMMY_CONFIG)
    state = make_state("How do I beat Tiger Vanguard?", chapter=2)
    state.retrieved_docs = [make_doc("wiki answer", source="wiki")]

    should_run, reason = planner.should_execute(state, "boss_strategy", AnalysisAgent)

    assert should_run is False
    assert "Consensus analysis" in reason
    assert state.consensus_analysis == {"strategies": [], "conflicts": []}