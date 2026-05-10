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
from src.agents.profile_agent import ProfileAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH_KEYS = {
    ("retrieval", "chroma_persist_dir"),
    ("retrieval", "bm25_index_path"),
    ("data", "chunks_path"),
    ("data", "raw_wiki_dir"),
    ("data", "raw_nga_dir"),
    ("data", "raw_bilibili_dir"),
    ("data", "raw_reddit_dir"),
    ("logging", "trace_path"),
}


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _resolve_config_paths(config, config_path.parent)


def _resolve_config_paths(config: dict, base_dir: Path) -> dict:
    for section, key in _CONFIG_PATH_KEYS:
        section_map = config.get(section)
        value = section_map.get(key) if isinstance(section_map, dict) else None
        if not value:
            continue
        path_value = Path(value)
        if not path_value.is_absolute():
            section_map[key] = str((base_dir / path_value).resolve())
    return config


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
        screenshots: list[bytes] | None = None,
    ) -> AgentState:
        state = AgentState(
            user_query=query,
            player_profile=profile,
            conversation_history=history or [],
            user_screenshot=screenshot,
            user_screenshots=screenshots or ([] if screenshot is None else [screenshot]),
        )

        did_preprocess_profile = False
        if state.screenshots():
            state = ProfileAgent(self.config).execute(state)
            did_preprocess_profile = True
            self.tracer.log("[orchestrator] preprocessed screenshots via ProfileAgent")
            if not query.strip():
                state.workflow = "profile_update"
                return state

        # 1. Classify intent → select workflow
        workflow_name = self.router.route(state)
        state.workflow = workflow_name
        self.tracer.log(f"[orchestrator] route → '{workflow_name}'")

        # 2. Execute each agent in the workflow sequentially
        agent_sequence = self.workflows.get(workflow_name, self.workflows["boss_strategy"])
        for AgentClass in agent_sequence:
            if did_preprocess_profile and AgentClass is ProfileAgent:
                continue
            agent = AgentClass(self.config)
            state = agent.execute(state)

        return state
