from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


BUILD_LABELS_ZH = {
    None: "未设置",
    "dodge": "闪身流",
    "parry": "棍反流",
    "spell": "法术流",
    "hybrid": "混合",
}

BUILD_LABELS_EN = {
    None: "Not set",
    "dodge": "Dodge",
    "parry": "Parry",
    "spell": "Spell",
    "hybrid": "Hybrid",
}

STANCE_LABELS_ZH = {
    None: "未设置",
    "smash": "劈棍",
    "pillar": "立棍",
    "thrust": "戳棍",
}

STANCE_LABELS_EN = {
    None: "Not set",
    "smash": "Smash",
    "pillar": "Pillar",
    "thrust": "Thrust",
}


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
class PlanStep:
    agent: str
    goal: str = ""
    status: str = "pending"   # pending | completed | skipped
    reason: str = ""


@dataclass
class ExecutionPlan:
    workflow: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    stop_conditions: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)


@dataclass
class PlayerProfile:
    chapter: Optional[int] = None
    current_boss: Optional[str] = None
    build: Optional[str] = None   # "dodge" | "parry" | "spell" | "hybrid"
    primary_stance: Optional[str] = None  # "smash" | "pillar" | "thrust"
    staff_level: Optional[int] = None
    equipped_spirit: Optional[str] = None
    equipped_armor: List[str] = field(default_factory=list)
    equipped_mysticism: Optional[str] = None
    equipped_body: Optional[str] = None
    equipped_strand: Optional[str] = None
    equipped_transformation: Optional[str] = None
    equipped_spells: List[str] = field(default_factory=list)
    unlocked_skills: List[str] = field(default_factory=list)
    unlocked_spells: List[str] = field(default_factory=list)
    unlocked_transformations: List[str] = field(default_factory=list)
    skills_explicit: bool = False
    spells_explicit: bool = False
    transformations_explicit: bool = False

    def to_context_string(self, language: str = "zh") -> str:
        if language == "en":
            current_boss = self.current_boss or "Not set"
            chapter = f"Chapter {self.chapter}" if self.chapter is not None else "Not set"
            build = BUILD_LABELS_EN.get(self.build, self.build or "Not set")
            stance = STANCE_LABELS_EN.get(self.primary_stance, self.primary_stance or "Not set")
            staff_level = f"Lv.{self.staff_level}" if self.staff_level is not None else "Not set"
            spirit = self.equipped_spirit or "None"
            return (
                f"Current boss: {current_boss} | Chapter: {chapter} | Build: {build} | "
                f"Stance: {stance} | Staff: {staff_level} | Spirit: {spirit} | "
                f"Spell slots: {_format_spell_slots(self, language)} | "
                f"Transformation slot: {_format_scalar_value(self.equipped_transformation, language)} | "
                f"Armor: {_format_profile_values(self.equipped_armor, language)} | "
                f"Equipped spells: {_format_profile_values(self.equipped_spells, language)} | "
                f"Skills: {_format_profile_values(self.unlocked_skills, language)} | "
                f"Unlocked spells: {_format_profile_values(self.unlocked_spells, language)} | "
                f"Transformations: {_format_profile_values(self.unlocked_transformations, language)}"
            )

        current_boss = self.current_boss or "未设置"
        chapter = f"第{self.chapter}章" if self.chapter is not None else "未设置"
        build = BUILD_LABELS_ZH.get(self.build, self.build or "未设置")
        stance = STANCE_LABELS_ZH.get(self.primary_stance, self.primary_stance or "未设置")
        staff_level = f"Lv.{self.staff_level}" if self.staff_level is not None else "未设置"
        spirit = self.equipped_spirit or "无"
        return (
            f"当前 boss: {current_boss} | 章节: {chapter} | 流派: {build} | "
            f"棍势: {stance} | 棍法: {staff_level} | 精魄: {spirit} | "
            f"法术槽: {_format_spell_slots(self, language)} | "
            f"当前变化: {_format_scalar_value(self.equipped_transformation, language)} | "
            f"披挂: {_format_profile_values(self.equipped_armor, language)} | "
            f"装备法术: {_format_profile_values(self.equipped_spells, language)} | "
            f"技能: {_format_profile_values(self.unlocked_skills, language)} | "
            f"法术: {_format_profile_values(self.unlocked_spells, language)} | "
            f"变身: {_format_profile_values(self.unlocked_transformations, language)}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentState:
    # Input
    user_query: str
    user_screenshot: Optional[bytes] = None
    user_screenshots: List[bytes] = field(default_factory=list)

    # User context
    player_profile: PlayerProfile = field(default_factory=PlayerProfile)
    conversation_history: List[Message] = field(default_factory=list)

    # Intermediate results (written by each agent)
    detected_intent: Optional[str] = None
    workflow: Optional[str] = None
    identified_entities: List[str] = field(default_factory=list)
    retrieved_docs: List[Document] = field(default_factory=list)
    consensus_analysis: Optional[Dict] = None
    execution_plan: Optional[ExecutionPlan] = None
    evidence_gaps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[Dict[str, str]] = field(default_factory=list)
    need_user_clarification: bool = False
    answer_confidence: float = 0.0
    stop_reason: Optional[str] = None

    # Output
    final_answer: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    profile_updates: List[Dict[str, Any]] = field(default_factory=list)

    # Debug / tracing
    trace: List[TraceEvent] = field(default_factory=list)

    def screenshots(self) -> List[bytes]:
        if self.user_screenshots:
            return self.user_screenshots
        return [self.user_screenshot] if self.user_screenshot else []


def _format_profile_values(values: List[str], language: str) -> str:
    if not values:
        return "None" if language == "en" else "无"
    return ", ".join(str(value) for value in values)


def _format_scalar_value(value: Optional[str], language: str) -> str:
    if not value:
        return "None" if language == "en" else "无"
    return str(value)


def _format_spell_slots(profile: PlayerProfile, language: str) -> str:
    if language == "en":
        labels = (
            ("Mysticism", profile.equipped_mysticism),
            ("Body", profile.equipped_body),
            ("Strand", profile.equipped_strand),
        )
    else:
        labels = (
            ("奇术", profile.equipped_mysticism),
            ("身法", profile.equipped_body),
            ("毫毛", profile.equipped_strand),
        )
    return ", ".join(
        f"{label}={_format_scalar_value(value, language)}" for label, value in labels
    )
