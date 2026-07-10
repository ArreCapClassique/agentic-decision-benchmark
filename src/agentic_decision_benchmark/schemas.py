from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import field_validator

ModeName = Literal["single", "supervisor", "self_organizing"]
StrategyId = str
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

DEFAULT_STRATEGY_IDS: tuple[StrategyId, ...] = ("A", "B", "C", "D")
STRATEGY_IDS: tuple[StrategyId, ...] = DEFAULT_STRATEGY_IDS
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
VALID_BLACKBOARD_ITEM_TYPES = {
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
}


def strategy_ids_from_candidates(candidate_strategies: dict[str, Any]) -> tuple[StrategyId, ...]:
    strategy_ids = tuple(str(key).strip() for key in candidate_strategies if str(key).strip())
    if not strategy_ids:
        raise ValueError("candidate_strategies must contain at least one strategy.")
    return strategy_ids


def strategy_id_label(strategy_ids: tuple[StrategyId, ...] | list[StrategyId] | set[StrategyId]) -> str:
    return "/".join(str(strategy_id) for strategy_id in strategy_ids)


def normalize_strategy_id(value: Any) -> StrategyId | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    strategy_match = re.match(r"^Strategy\s+([A-Za-z0-9_-]+)\b", text, flags=re.IGNORECASE)
    if strategy_match:
        return strategy_match.group(1)
    strategy_mentions = {
        match.group(1)
        for match in re.finditer(r"\bStrategy\s+([A-Za-z0-9_-]+)\b", text, flags=re.IGNORECASE)
    }
    if len(strategy_mentions) == 1:
        return next(iter(strategy_mentions))
    compact_match = re.match(r"^([A-Za-z0-9_-]+)\s*(?::|-|–|—|\()", text)
    if compact_match:
        return compact_match.group(1)
    return text


def normalize_strategy_id_list(value: Any) -> list[StrategyId]:
    if value is None or value == "":
        return []
    raw_values = value if isinstance(value, list | tuple | set) else [value]
    normalized: list[StrategyId] = []
    for raw in raw_values:
        if isinstance(raw, dict):
            raw = raw.get("strategy_id") or raw.get("strategy") or raw.get("id") or raw
        candidates = re.split(r"\s*(?:,|>|;|\||\r?\n)+\s*", raw.strip()) if isinstance(raw, str) else [raw]
        for candidate in candidates:
            strategy_id = normalize_strategy_id(candidate)
            if strategy_id is not None and strategy_id not in normalized:
                normalized.append(strategy_id)
    return normalized


def normalize_strategy_score_map(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized: dict[StrategyId, Any] = {}
    for key, item in value.items():
        strategy_id = normalize_strategy_id(key)
        if strategy_id is not None:
            normalized[strategy_id] = item
    return normalized


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


def normalize_string_list(value: Any) -> list[str]:
    raw_values = value if isinstance(value, list) else [value] if value else []
    normalized: list[str] = []
    for raw in raw_values:
        if isinstance(raw, dict):
            for key in ("text", "content", "statement", "claim", "risk", "opportunity", "assumption", "question"):
                item = raw.get(key)
                if item:
                    normalized.append(str(item))
                    break
            else:
                normalized.append(str(raw))
        else:
            normalized.append(str(raw))
    return normalized


def normalize_strategy_condition_map(value: Any) -> dict[StrategyId, list[str]]:
    if not value:
        return {}
    normalized: dict[StrategyId, list[str]] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            strategy_id = normalize_strategy_id(key)
            if strategy_id is not None:
                normalized[strategy_id] = normalize_string_list(item)
        return normalized
    if isinstance(value, list):
        for raw in value:
            if not isinstance(raw, dict):
                continue
            strategy_id = normalize_strategy_id(raw.get("strategy_id") or raw.get("strategy") or raw.get("id"))
            if strategy_id is None:
                continue
            conditions = raw.get("conditions") or raw.get("condition") or raw.get("text") or raw.get("content")
            normalized[strategy_id] = normalize_string_list(conditions)
    return normalized


def normalize_blackboard_item_type(value: Any) -> BlackboardItemType:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in VALID_BLACKBOARD_ITEM_TYPES:
        return normalized  # type: ignore[return-value]
    if any(word in normalized for word in ("risk", "constraint", "threat", "liquidity")):
        return "risk"
    if any(word in normalized for word in ("opportunity", "upside", "option")):
        return "opportunity"
    if "assumption" in normalized:
        return "assumption"
    if "question" in normalized:
        return "question"
    if any(word in normalized for word in ("critique", "challenge")):
        return "critique"
    if "score" in normalized:
        return "scorecard"
    if "concern" in normalized:
        return "minority_concern"
    if "information" in normalized:
        return "new_information"
    if "consensus" in normalized:
        return "consensus"
    return "claim"


def normalize_severity(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"low", "minor"}:
            return 1
        if normalized in {"medium", "moderate"}:
            return 3
        if normalized in {"high", "severe"}:
            return 5
    return int(value)


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
    strategies_supported: list[StrategyId] = Field(default_factory=list)
    strategies_challenged: list[StrategyId] = Field(default_factory=list)
    strategies_conditioned: list[StrategyId] = Field(default_factory=list)
    criterion: str | None = None
    stance: BlackboardStance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: int | None = Field(default=None, ge=1, le=5)

    @field_validator("item_type", mode="before")
    @classmethod
    def normalize_item_type(cls, value: Any) -> BlackboardItemType:
        return normalize_blackboard_item_type(value)

    @field_validator("topic_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[TopicTag]:
        return normalize_topic_tags(value)

    @field_validator("stance", mode="before")
    @classmethod
    def normalize_stance(cls, value: Any) -> BlackboardStance | None:
        return normalize_blackboard_stance(value)

    @field_validator("related_strategy", mode="before")
    @classmethod
    def normalize_related_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)

    @field_validator("strategies_supported", "strategies_challenged", "strategies_conditioned", mode="before")
    @classmethod
    def normalize_strategy_links(cls, value: Any) -> list[StrategyId]:
        return normalize_strategy_id_list(value)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity_value(cls, value: Any) -> int | None:
        return normalize_severity(value)

    @model_validator(mode="after")
    def populate_strategy_links(self) -> "BlackboardContribution":
        if self.related_strategy:
            if self.stance == "supports" and self.related_strategy not in self.strategies_supported:
                self.strategies_supported.append(self.related_strategy)
            if self.stance in {"opposes", "challenges"} and self.related_strategy not in self.strategies_challenged:
                self.strategies_challenged.append(self.related_strategy)
        return self


class BlackboardItem(StrictModel):
    id: str = ""
    round_id: int = Field(ge=0)
    cycle_id: int = Field(default=0, ge=0)
    author: str
    item_type: BlackboardItemType
    content: str
    topic_tags: list[TopicTag] = Field(default_factory=lambda: ["uncategorized"])
    related_strategy: StrategyId | None = None
    strategies_supported: list[StrategyId] = Field(default_factory=list)
    strategies_challenged: list[StrategyId] = Field(default_factory=list)
    strategies_conditioned: list[StrategyId] = Field(default_factory=list)
    criterion: str | None = None
    stance: BlackboardStance | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    target_agent: str | None = None
    target_claim: str | None = None
    target_item_id: str | None = None
    severity: int | None = Field(default=None, ge=1, le=5)

    @field_validator("item_type", mode="before")
    @classmethod
    def normalize_item_type(cls, value: Any) -> BlackboardItemType:
        return normalize_blackboard_item_type(value)

    @field_validator("topic_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[TopicTag]:
        return normalize_topic_tags(value)

    @field_validator("stance", mode="before")
    @classmethod
    def normalize_stance(cls, value: Any) -> BlackboardStance | None:
        return normalize_blackboard_stance(value)

    @field_validator("related_strategy", mode="before")
    @classmethod
    def normalize_related_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)

    @field_validator("strategies_supported", "strategies_challenged", "strategies_conditioned", mode="before")
    @classmethod
    def normalize_strategy_links(cls, value: Any) -> list[StrategyId]:
        return normalize_strategy_id_list(value)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity_value(cls, value: Any) -> int | None:
        return normalize_severity(value)

    @model_validator(mode="after")
    def populate_id(self) -> "BlackboardItem":
        if self.related_strategy:
            if self.stance == "supports" and self.related_strategy not in self.strategies_supported:
                self.strategies_supported.append(self.related_strategy)
            if self.stance in {"opposes", "challenges"} and self.related_strategy not in self.strategies_challenged:
                self.strategies_challenged.append(self.related_strategy)
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

    @field_validator("reasoning", "risks", "assumptions", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("recommended_strategy", mode="before")
    @classmethod
    def normalize_recommended_strategy(cls, value: Any) -> StrategyId:
        normalized = normalize_strategy_id(value)
        if normalized is None:
            raise ValueError("recommended_strategy is required.")
        return normalized


class SupervisorPlan(StrictModel):
    plan: list[str]
    domains_to_consult: list[str]
    decision_criteria: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("plan", "domains_to_consult", "decision_criteria", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class DomainAnalysis(StrictModel):
    claims: list[str]
    risks: list[str]
    opportunities: list[str]
    assumptions: list[str]
    recommendation: StrategyId
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("claims", "risks", "opportunities", "assumptions", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, value: Any) -> StrategyId:
        normalized = normalize_strategy_id(value)
        if normalized is None:
            raise ValueError("recommendation is required.")
        return normalized


class SupervisorRecommendation(StrictModel):
    recommended_strategy: StrategyId
    recommendation: str
    synthesis: list[str]
    tradeoffs: list[str]
    risks: list[str]
    assumptions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("synthesis", "tradeoffs", "risks", "assumptions", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("recommended_strategy", mode="before")
    @classmethod
    def normalize_recommended_strategy(cls, value: Any) -> StrategyId:
        normalized = normalize_strategy_id(value)
        if normalized is None:
            raise ValueError("recommended_strategy is required.")
        return normalized


class Round1Analysis(StrictModel):
    claims: list[str]
    risks: list[str]
    opportunities: list[str]
    assumptions: list[str]
    questions_for_others: list[str]
    domain_preferred_strategy: StrategyId
    organization_preferred_strategy: StrategyId
    domain_ranking: list[StrategyId]
    organization_ranking: list[StrategyId]
    strategies_supported: list[StrategyId] = Field(default_factory=list)
    strategies_challenged: list[StrategyId] = Field(default_factory=list)
    non_negotiable_constraints: list[str] = Field(default_factory=list)
    conditions_for_accepting_other_strategy: dict[StrategyId, list[str]] = Field(default_factory=dict)
    blackboard_items: list[BlackboardContribution] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "claims",
        "risks",
        "opportunities",
        "assumptions",
        "questions_for_others",
        "non_negotiable_constraints",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("domain_preferred_strategy", "organization_preferred_strategy", mode="before")
    @classmethod
    def normalize_preferred_strategy(cls, value: Any) -> StrategyId:
        normalized = normalize_strategy_id(value)
        if normalized is None:
            raise ValueError("Round 1 preferred strategy fields are required.")
        return normalized

    @field_validator("domain_ranking", "organization_ranking", "strategies_supported", "strategies_challenged", mode="before")
    @classmethod
    def normalize_strategy_lists(cls, value: Any) -> list[StrategyId]:
        return normalize_strategy_id_list(value)

    @field_validator("conditions_for_accepting_other_strategy", mode="before")
    @classmethod
    def normalize_conditions(cls, value: Any) -> dict[StrategyId, list[str]]:
        return normalize_strategy_condition_map(value)

    @model_validator(mode="after")
    def populate_rankings_and_links(self) -> "Round1Analysis":
        if self.domain_preferred_strategy not in self.domain_ranking:
            self.domain_ranking.insert(0, self.domain_preferred_strategy)
        if self.organization_preferred_strategy not in self.organization_ranking:
            self.organization_ranking.insert(0, self.organization_preferred_strategy)
        for strategy_id in (self.domain_preferred_strategy, self.organization_preferred_strategy):
            if strategy_id not in self.strategies_supported:
                self.strategies_supported.append(strategy_id)
        return self


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

    @field_validator("related_strategy", mode="before")
    @classmethod
    def normalize_related_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)


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

    @field_validator(
        "changed_beliefs",
        "accepted_critiques",
        "rejected_critiques",
        "remaining_concerns",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("updated_recommendation", mode="before")
    @classmethod
    def normalize_updated_recommendation(cls, value: Any) -> StrategyId:
        normalized = normalize_strategy_id(value)
        if normalized is None:
            raise ValueError("updated_recommendation is required.")
        return normalized


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

    @field_validator("scores", mode="before")
    @classmethod
    def normalize_scores(cls, value: Any) -> Any:
        return normalize_strategy_score_map(value)

    @model_validator(mode="after")
    def validate_strategy_keys(self) -> "Scorecard":
        if not self.scores:
            raise ValueError("Scorecard must contain at least one strategy score.")
        return self


class Round4ScorecardOutput(StrictModel):
    scores: dict[StrategyId, StrategyScore]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("scores", mode="before")
    @classmethod
    def normalize_scores(cls, value: Any) -> Any:
        return normalize_strategy_score_map(value)

    @model_validator(mode="after")
    def validate_strategy_keys(self) -> "Round4ScorecardOutput":
        if not self.scores:
            raise ValueError("Scorecard output must contain at least one strategy score.")
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

    @field_validator("recommended_strategy", mode="before")
    @classmethod
    def normalize_recommended_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)

    @field_validator("aggregate_scores", mode="before")
    @classmethod
    def normalize_aggregate_scores(cls, value: Any) -> Any:
        return normalize_strategy_score_map(value)

    @field_validator("unresolved_strategies", mode="before")
    @classmethod
    def normalize_unresolved_strategies(cls, value: Any) -> list[StrategyId]:
        raw_values = value if isinstance(value, list) else [value] if value else []
        return [item for raw in raw_values if (item := normalize_strategy_id(raw)) is not None]


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

    @field_validator("related_strategy", mode="before")
    @classmethod
    def normalize_related_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)


class ConvergenceStatus(StrictModel):
    cycle_id: int = Field(ge=0)
    converged: bool
    leading_strategy: StrategyId | None = None
    agreement_ratio: float = Field(ge=0.0, le=1.0)
    unresolved_high_severity_conflicts: int = Field(ge=0)
    low_confidence_agents: list[str] = Field(default_factory=list)
    reason: str

    @field_validator("leading_strategy", mode="before")
    @classmethod
    def normalize_leading_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)


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

    @field_validator("final_selected_strategy", mode="before")
    @classmethod
    def normalize_final_selected_strategy(cls, value: Any) -> StrategyId | None:
        return normalize_strategy_id(value)


class ProviderStats(StrictModel):
    model_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @property
    def total_estimated_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens

