from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, get_args

from agentic_decision_benchmark.schemas import AgentOutput, BlackboardItem, ModeName, Scorecard


class BenchmarkState(TypedDict):
    mode: ModeName
    scenario: str
    candidate_strategies: dict[str, Any]
    metadata: dict[str, Any]
    round_id: int
    blackboard: Annotated[list[BlackboardItem], operator.add]
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    scorecards: Annotated[list[Scorecard], operator.add]
    final_recommendation: dict[str, Any]
    metrics: dict[str, Any]
    evaluation: dict[str, Any]
    fault_injection: bool
    new_info_injection: bool


def validate_mode(mode: str) -> ModeName:
    valid_modes = set(get_args(ModeName))
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode {mode!r}. Expected one of: {sorted(valid_modes)}")
    return mode  # type: ignore[return-value]


def create_initial_state(
    *,
    mode: str,
    scenario: str,
    candidate_strategies: dict[str, Any],
    fault_injection: bool = False,
    new_info_injection: bool = False,
) -> BenchmarkState:
    return {
        "mode": validate_mode(mode),
        "scenario": scenario,
        "candidate_strategies": candidate_strategies,
        "metadata": {
            "information_setting": "equal_total",
            "private_role_briefs_enabled": True,
        },
        "round_id": 0,
        "blackboard": [],
        "agent_outputs": [],
        "scorecards": [],
        "final_recommendation": {},
        "metrics": {},
        "evaluation": {},
        "fault_injection": fault_injection,
        "new_info_injection": new_info_injection,
    }


def append_reducer(existing: list[Any], new: list[Any]) -> list[Any]:
    return operator.add(existing, new)

