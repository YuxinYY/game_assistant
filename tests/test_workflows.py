"""
Tests for routing and workflow selection.
"""

import pytest
from src.core.router import Router
from src.core.state import AgentState, PlayerProfile
from src.core.workflows import build_workflows


DUMMY_CONFIG = {
    "llm": {"model": "claude-sonnet-4-7", "temperature": 0.3},
}


def make_state(query: str) -> AgentState:
    return AgentState(user_query=query, player_profile=PlayerProfile())


class TestRouter:
    def setup_method(self):
        self.router = Router(DUMMY_CONFIG, llm_client=None)  # uses heuristic fallback

    def test_boss_query_routes_correctly(self):
        state = make_state("虎先锋那个招怎么躲？")
        assert self.router.route(state) == "boss_strategy"

    def test_decision_query_routes_correctly(self):
        state = make_state("闪身流和棍反流哪个好？")
        assert self.router.route(state) == "decision_making"

    def test_navigation_query_routes_correctly(self):
        state = make_state("广智在哪里？")
        assert self.router.route(state) == "navigation"

    def test_llm_route_uses_valid_model_output(self):
        class FakeLLM:
            def complete(self, messages, system=""):
                return "decision_making"

        router = Router(DUMMY_CONFIG, llm_client=FakeLLM())
        state = make_state("闪身流和棍反流哪个好？")
        assert router.route(state) == "decision_making"

    def test_llm_route_parses_json_output(self):
        class FakeLLM:
            def complete(self, messages, system=""):
                return '{"workflow": "navigation"}'

        router = Router(DUMMY_CONFIG, llm_client=FakeLLM())
        state = make_state("广智在哪里？")
        assert router.route(state) == "navigation"

    def test_llm_path_falls_back_to_heuristic_during_mvp_stage(self):
        class FakeLLM:
            def complete(self, messages, system=""):
                raise RuntimeError("API unavailable")

        router = Router(DUMMY_CONFIG, llm_client=FakeLLM())
        state = make_state("虎先锋那个招怎么躲？")
        assert router.route(state) == "boss_strategy"


class TestWorkflows:
    def test_all_workflows_defined(self):
        workflows = build_workflows()
        assert "boss_strategy" in workflows
        assert "decision_making" in workflows
        assert "navigation" in workflows
        assert "fact_lookup" in workflows

    def test_boss_strategy_has_five_agents(self):
        workflows = build_workflows()
        assert len(workflows["boss_strategy"]) == 5

    def test_boss_strategy_starts_with_profile_agent(self):
        workflows = build_workflows()
        assert workflows["boss_strategy"][0].__name__ == "ProfileAgent"
