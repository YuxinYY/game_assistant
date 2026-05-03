"""
Workflow definitions: each workflow is an ordered list of agent classes.
Adding a new workflow = adding one entry here, no orchestrator changes needed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent

# Import lazily to avoid circular deps at module load
def _get_agents():
    from src.agents.wiki_agent import WikiAgent
    from src.agents.community_agent import CommunityAgent
    from src.agents.profile_agent import ProfileAgent
    from src.agents.analysis_agent import AnalysisAgent
    from src.agents.synthesis_agent import SynthesisAgent
    return WikiAgent, CommunityAgent, ProfileAgent, AnalysisAgent, SynthesisAgent


def build_workflows() -> dict[str, list]:
    WikiAgent, CommunityAgent, ProfileAgent, AnalysisAgent, SynthesisAgent = _get_agents()
    return {
        "boss_strategy": [
            ProfileAgent,     # update player state before downstream retrieval/personalization
            WikiAgent,        # identify move name from wiki
            CommunityAgent,   # find player counter-strategies
            AnalysisAgent,    # consensus + conflict detection
            SynthesisAgent,   # write final cited answer
        ],
        "decision_making": [
            ProfileAgent,     # load build context first
            WikiAgent,
            CommunityAgent,
            AnalysisAgent,
            SynthesisAgent,
        ],
        "navigation": [
            WikiAgent,
            SynthesisAgent,
        ],
        "fact_lookup": [
            WikiAgent,
            SynthesisAgent,
        ],
    }
