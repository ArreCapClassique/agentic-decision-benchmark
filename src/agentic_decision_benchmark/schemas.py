from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

STRATEGY_IDS: tuple[StrategyId, ...] = ("A", "B", "C", "D")
MODE_NAMES: tuple[str, ...] = ("single", "supervisor", "self_organizing")
MCDA_CRITERIA: tuple[str, ...] = (
    "financial_viability",
    "operational_feasibility",
    "strategic_value",
    "workforce_feasibility",
    "legal_compliance_safety",
)


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


class BlackboardItem(StrictModel):
    round_id: int = Field(ge=0)
    author: str
    item_type: BlackboardItemType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    target_agent: str | None = None
    target_claim: str | None = None
    related_strategy: StrategyId | None = None
    severity: int | None = Field(default=None, ge=1, le=5)


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
    confidence: float = Field(ge=0.0, le=1.0)


class Critique(StrictModel):
    target_agent: str
    target_claim: str
    critique: str
    missing_assumption: str
    risk_introduced: str
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    related_strategy: StrategyId | None = None


class Round2CritiqueOutput(StrictModel):
    critiques: list[Critique]


class Round3BeliefUpdate(StrictModel):
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


class EvaluationResult(StrictModel):
    decision_quality: int = Field(ge=1, le=5)
    convergence: int = Field(ge=1, le=5)
    resilience: int = Field(ge=1, le=5)
    explainability: int = Field(ge=1, le=5)
    adaptability: int = Field(ge=1, le=5)
    cost_efficiency: int = Field(ge=1, le=5)
    runtime_efficiency: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    rationale: str


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


class ProviderStats(StrictModel):
    model_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @property
    def total_estimated_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens

