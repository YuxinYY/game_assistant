"""
Main controller: receives a query + profile, runs the appropriate workflow,
returns a fully populated AgentState.
"""

import yaml
from pathlib import Path
from src.core.state import AgentState, PlayerProfile, Message
from src.core.router import Router
from src.core.workflows import build_workflows
from src.llm.client import LLMClient
from src.utils.tracing import Tracer


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.llm = LLMClient(config)
        self.router = Router(config, llm_client=self.llm)
        self.workflows = build_workflows()
        self.tracer = Tracer()

    def run(
        self,
        query: str,
        profile: PlayerProfile,
        history: list[Message] | None = None,
        screenshot: bytes | None = None,
    ) -> AgentState:
        state = AgentState(
            user_query=query,
            player_profile=profile,
            conversation_history=history or [],
            user_screenshot=screenshot,
        )

        # 1. Classify intent → select workflow
        workflow_name = self.router.route(state)
        state.workflow = workflow_name
        self.tracer.log(f"[orchestrator] route → '{workflow_name}'")

        # 2. Execute each agent in the workflow sequentially
        agent_sequence = self.workflows.get(workflow_name, self.workflows["boss_strategy"])
        for AgentClass in agent_sequence:
            agent = AgentClass(self.config)
            state = agent.execute(state)

        return state
