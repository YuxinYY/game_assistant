from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Document:
    text: str
    source: str        # "wiki" | "nga" | "bilibili" | "reddit"
    url: str
    chapter: Optional[int] = None    # chapter gate for spoiler filter
    entity: Optional[str] = None     # e.g. "虎先锋"
    credibility: float = 0.8
    post_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    source: str
    url: str
    excerpt: str
    author: str = ""


@dataclass
class TraceEvent:
    agent: str
    step: int
    action: str
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PlayerProfile:
    chapter: int = 1
    build: str = "dodge"   # "dodge" | "parry" | "spell" | "hybrid"
    staff_level: int = 1
    unlocked_skills: List[str] = field(default_factory=list)
    unlocked_spells: List[str] = field(default_factory=list)
    unlocked_transformations: List[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        return (
            f"章节: 第{self.chapter}章 | 流派: {self.build} | 棍法: Lv.{self.staff_level} | "
            f"技能: {self.unlocked_skills or '无'} | "
            f"法术: {self.unlocked_spells or '无'} | "
            f"变身: {self.unlocked_transformations or '无'}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentState:
    # Input
    user_query: str
    user_screenshot: Optional[bytes] = None

    # User context
    player_profile: PlayerProfile = field(default_factory=PlayerProfile)
    conversation_history: List[Message] = field(default_factory=list)

    # Intermediate results (written by each agent)
    detected_intent: Optional[str] = None
    workflow: Optional[str] = None
    identified_entities: List[str] = field(default_factory=list)
    retrieved_docs: List[Document] = field(default_factory=list)
    consensus_analysis: Optional[Dict] = None

    # Output
    final_answer: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)

    # Debug / tracing
    trace: List[TraceEvent] = field(default_factory=list)
