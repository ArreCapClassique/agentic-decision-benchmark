from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import field_validator

ModeName = Literal["single", "supervisor", "self_organizing"]
StrategyId = Literal["A", "B", "C", "D"]
BlackboardItemType = Literal[
    "claim",
    "risk",
    "opportunity",
    "assumption",
    "question",
    "critique",
    "belief_update",
    "scorecard",
    "minority_concern",
    "new_information",
    "consensus",
]
BlackboardStance = Literal["supports", "opposes", "neutral", "challenges"]
TopicTag = Literal[
    "financial_viability",
    "investment_capacity",
    "cash_flow",
    "revenue_exposure",
    "operational_feasibility",
    "retooling_risk",
    "production_disruption",
    "data_readiness",
    "ai_feasibility",
    "implementation_timeline",
    "workforce_feasibility",
    "hiring_timeline",
    "skill_gap",
    "compliance",
    "auditability",
    "contract_risk",
    "strategic_value",
    "diversification",
    "market_uncertainty",
    "customer_concentration",
    "uncategorized",
]
ConflictType = Literal["direct_critique", "opposing_stances", "risk_cluster", "score_divergence"]
ConflictStatus = Literal["unresolved", "partially_resolved", "resolved"]

STRATEGY_IDS: tuple[StrategyId, ...] = ("A", "B", "C", "D")
MODE_NAMES: tuple[str, ...] = ("single", "supervisor", "self_organizing")
MCDA_CRITERIA: tuple[str, ...] = (
    "financial_viability",
    "operational_feasibility",
    "strategic_value",
    "workforce_feasibility",
    "legal_compliance_safety",
)
TOPIC_TAGS: tuple[str, ...] = (
    "financial_viability",
    "investment_capacity",
    "cash_flow",
    "revenue_exposure",
    "operational_feasibility",
    "retooling_risk",
    "production_disruption",
    "data_readiness",
    "ai_feasibility",
    "implementation_timeline",
    "workforce_feasibility",
    "hiring_timeline",
    "skill_gap",
    "compliance",
    "auditability",
    "contract_risk",
    "strategic_value",
    "diversification",
    "market_uncertainty",
    "customer_concentration",
    "uncategorized",
)
VALID_TOPIC_TAGS = set(TOPIC_TAGS)
VALID_BLACKBOARD_STANCES = {"supports", "opposes", "neutral", "challenges"}


def normalize_topic_tags(value: Any) -> list[TopicTag]:
    raw_values = value if isinstance(value, list) else [value] if value else []
    normalized: list[str] = []
    for raw in raw_values:
        tag = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if tag in VALID_TOPIC_TAGS and tag not in normalized:
            normalized.append(tag)
    if not normalized:
        normalized.append("uncategorized")
    return normalized  # type: ignore[return-value]


def normalize_blackboard_stance(value: Any) -> BlackboardStance | None:
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in VALID_BLACKBOARD_STANCES:
        raise ValueError(f"Invalid blackboard stance {value!r}. Expected one of {sorted(VALID_BLACKBOARD_STANCES)}")
    return normalized  # type: ignore[return-value]


def deterministic_blackboard_id(parts: list[Any]) -> str:
    payload = "|".join("" if item is None else str(item) for item in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"bb-{digest}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateStrategy(StrictModel):
    name: str
    description: str
    core_tradeoff: str


class AgentDefinition(StrictModel):
    name: str
    domain: str
    responsibilities: list[str]
    constraints: list[str]


class PrivateBriefItem(StrictModel):
    id: str
    source_label: str
    synthetic: bool = True
    statement: str
    certainty: Literal["low", "medium", "high"]
    relevance: list[str]


class RolePrivateBrief(StrictModel):
    visibility: str
    synthetic: bool = True
    brief_items: list[PrivateBriefItem]


class PrivateRoleBriefs(StrictModel):
    finance: RolePrivateBrief
    operations: RolePrivateBrief
    hr: RolePrivateBrief
    legal: RolePrivateBrief
    strategy: RolePrivateBrief
    technology: RolePrivateBrief


class BlackboardContribution(StrictModel):
    item_type: BlackboardItemType
    content: str
    topic_tags: list[TopicTag] = Field(default_factory=lambda: ["uncategorized"])
    related_strategy: StrategyId | None = None
    criterion: str | None = None
    stance: BlackboardStance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: int | None = Field(default=None, ge=1, le=5)

    @field_validator("topic_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[TopicTag]:
        return normalize_topic_tags(value)

    @field_validator("stance", mode="before")
    @classmethod
    def normalize_stance(cls, value: Any) -> BlackboardStance | None:
        return normalize_blackboard_stance(value)


class BlackboardItem(StrictModel):
    id: str = ""
    round_id: int = Field(ge=0)
    cycle_id: int = Field(default=0, ge=0)
    author: str
    item_type: BlackboardItemType
    content: str
    topic_tags: list[TopicTag] = Field(default_factory=lambda: ["uncategorized"])
    related_strategy: StrategyId | None = None
    criterion: str | None = None
    stance: BlackboardStance | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    target_agent: str | None = None
    target_claim: str | None = None
    target_item_id: str | None = None
    severity: int | None = Field(default=None, ge=1, le=5)

    @field_validator("topic_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[TopicTag]:
        return normalize_topic_tags(value)

    @field_validator("stance", mode="before")
    @classmethod
    def normalize_stance(cls, value: Any) -> BlackboardStance | None:
        return normalize_blackboard_stance(value)

    @model_validator(mode="after")
    def populate_id(self) -> "BlackboardItem":
        if not self.id:
            self.id = deterministic_blackboard_id(
                [
                    self.round_id,
                    self.cycle_id,
                    self.author,
                    self.item_type,
                    self.content,
                    self.related_strategy,
                    self.criterion,
                    self.target_item_id,
                    self.target_agent,
                    self.target_claim,
                ]
            )
        return self


class AgentOutput(StrictModel):
    mode: ModeName
    phase: str
    agent: str
    round_id: int = Field(ge=0)
    output: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)


class SingleRecommendation(StrictModel):
    recommended_strategy: StrategyId
    recommendation: str
    reasoning: list[str]
    risks: list[str]
    assumptions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class SupervisorPlan(StrictModel):
    plan: list[str]
    domains_to_consult: list[str]
    decision_criteria: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class DomainAnalysis(StrictModel):
    claims: list[str]
    risks: list[str]
    opportunities: list[str]
    assumptions: list[str]
    recommendation: StrategyId
    confidence: float = Field(ge=0.0, le=1.0)


class SupervisorRecommendation(StrictModel):
    recommended_strategy: StrategyId
    recommendation: str
    synthesis: list[str]
    tradeoffs: list[str]
    risks: list[str]
    assumptions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class Round1Analysis(StrictModel):
    claims: list[str]
    risks: list[str]
    opportunities: list[str]
    assumptions: list[str]
    questions_for_others: list[str]
    initial_recommendation: StrategyId
    blackboard_items: list[BlackboardContribution] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Critique(StrictModel):
    target_item_id: str | None = None
    target_agent: str
    target_claim: str
    critique: str
    missing_assumption: str
    risk_introduced: str
    topic_tags: list[TopicTag] = Field(default_factory=lambda: ["uncategorized"])
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    related_strategy: StrategyId | None = None
    criterion: str | None = None

    @field_validator("topic_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[TopicTag]:
        return normalize_topic_tags(value)


class Round2CritiqueOutput(StrictModel):
    agent_name: str = ""
    selected_items: list[str] = Field(default_factory=list)
    critiques: list[Critique]


class Round3BeliefUpdate(StrictModel):
    agent_name: str = ""
    updated_recommendation: StrategyId
    changed_beliefs: list[str]
    accepted_critiques: list[str]
    rejected_critiques: list[str]
    remaining_concerns: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class CriteriaScores(StrictModel):
    financial_viability: int = Field(ge=1, le=5)
    operational_feasibility: int = Field(ge=1, le=5)
    strategic_value: int = Field(ge=1, le=5)
    workforce_feasibility: int = Field(ge=1, le=5)
    legal_compliance_safety: int = Field(ge=1, le=5)

    def mean(self) -> float:
        values = [
            self.financial_viability,
            self.operational_feasibility,
            self.strategic_value,
            self.workforce_feasibility,
            self.legal_compliance_safety,
        ]
        return sum(values) / len(values)


class StrategyScore(StrictModel):
    criteria: CriteriaScores
    overall: float = Field(ge=1.0, le=5.0)
    rationale: str


class Scorecard(StrictModel):
    agent: str
    round_id: int = 4
    scores: dict[StrategyId, StrategyScore]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_strategy_keys(self) -> "Scorecard":
        missing = set(STRATEGY_IDS) - set(self.scores)
        extra = set(self.scores) - set(STRATEGY_IDS)
        if missing or extra:
            raise ValueError(f"Scorecard must contain exactly A/B/C/D; missing={missing}, extra={extra}")
        return self


class Round4ScorecardOutput(StrictModel):
    scores: dict[StrategyId, StrategyScore]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_strategy_keys(self) -> "Round4ScorecardOutput":
        missing = set(STRATEGY_IDS) - set(self.scores)
        extra = set(self.scores) - set(STRATEGY_IDS)
        if missing or extra:
            raise ValueError(f"Scorecard output must contain exactly A/B/C/D; missing={missing}, extra={extra}")
        return self


class AggregateStrategyScore(StrictModel):
    mean: float
    stdev: float
    top_votes: int
    criteria_means: dict[str, float] = Field(default_factory=dict)


class ConsensusResult(StrictModel):
    recommended_strategy: StrategyId | None
    strategy_description: str
    aggregate_scores: dict[StrategyId, AggregateStrategyScore]
    agreement_ratio: float = Field(ge=0.0, le=1.0)
    minority_concerns: list[str]
    unresolved_risks: list[str]
    deliberation_summary: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    result_status: Literal["selected", "unresolved"] = "selected"
    unresolved_strategies: list[StrategyId] = Field(default_factory=list)


class SalienceRecord(StrictModel):
    item_id: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    critique_count: int = Field(ge=0)
    support_count: int = Field(ge=0)
    salience: float = Field(ge=0.0, le=1.0)
    domain_relevance: dict[str, float] = Field(default_factory=dict)


class ConflictItem(StrictModel):
    id: str
    cycle_id: int = Field(ge=0)
    conflict_type: ConflictType
    topic: str
    related_strategy: StrategyId | None = None
    criterion: str | None = None
    central_item_id: str | None = None
    supporting_items: list[str] = Field(default_factory=list)
    challenging_items: list[str] = Field(default_factory=list)
    agents_involved: list[str] = Field(default_factory=list)
    severity: int = Field(ge=1, le=5)
    status: ConflictStatus = "unresolved"
    summary: str


class ConvergenceStatus(StrictModel):
    cycle_id: int = Field(ge=0)
    converged: bool
    leading_strategy: StrategyId | None = None
    agreement_ratio: float = Field(ge=0.0, le=1.0)
    unresolved_high_severity_conflicts: int = Field(ge=0)
    low_confidence_agents: list[str] = Field(default_factory=list)
    reason: str


def _round_score_one_decimal(value: Any) -> float:
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid evaluator score {value!r}") from exc
    if not score.is_finite():
        raise ValueError(f"Evaluator score must be finite, got {value!r}")
    if score < Decimal("1.0") or score > Decimal("5.0"):
        raise ValueError(f"Evaluator score must be between 1.0 and 5.0, got {value!r}")
    return float(score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class EvaluationResult(StrictModel):
    decision_quality: float = Field(ge=1.0, le=5.0)
    convergence: float = Field(ge=1.0, le=5.0)
    resilience: float = Field(ge=1.0, le=5.0)
    explainability: float = Field(ge=1.0, le=5.0)
    adaptability: float = Field(ge=1.0, le=5.0)
    cost_efficiency: float = Field(ge=1.0, le=5.0)
    runtime_efficiency: float = Field(ge=1.0, le=5.0)
    overall: float = Field(ge=1.0, le=5.0)
    rationale: str

    @field_validator(
        "decision_quality",
        "convergence",
        "resilience",
        "explainability",
        "adaptability",
        "cost_efficiency",
        "runtime_efficiency",
        "overall",
        mode="before",
    )
    @classmethod
    def normalize_score(cls, value: Any) -> float:
        return _round_score_one_decimal(value)


class Metrics(StrictModel):
    runtime_seconds: float
    model_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    total_estimated_tokens: int
    token_estimate_method: str = "approximate whitespace token estimate"
    number_of_rounds: int
    number_of_blackboard_items: int
    number_of_claim_items: int
    number_of_critique_items: int
    number_of_belief_update_items: int
    number_of_scorecards: int
    agreement_ratio: float | None = None
    score_variance: float | None = None
    final_selected_strategy: StrategyId | None = None
    fault_correction_detected: bool = False
    new_information_incorporated: bool = False
    deliberation_cycles_used: int = 0
    max_deliberation_cycles: int = 0
    converged_before_max_cycles: bool = False
    number_of_conflicts: int = 0
    number_of_direct_critique_conflicts: int = 0
    number_of_opposing_stance_conflicts: int = 0
    number_of_risk_cluster_conflicts: int = 0
    unresolved_high_severity_conflicts: int = 0
    agreement_ratio_after_each_cycle: list[float] = Field(default_factory=list)
    low_confidence_agents_after_each_cycle: list[list[str]] = Field(default_factory=list)
    average_salience: float = 0.0
    top_salience_items: list[dict[str, Any]] = Field(default_factory=list)


class ProviderStats(StrictModel):
    model_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @property
    def total_estimated_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens

