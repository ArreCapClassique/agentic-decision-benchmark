from __future__ import annotations

from agentic_decision_benchmark.schemas import ConflictItem
from agentic_decision_benchmark.self_organization.convergence import evaluate_convergence, route_after_convergence


def _round3(agent: str, strategy: str, confidence: float = 0.8) -> dict:
    return {
        "agent": agent,
        "phase": "round_3_belief_update_cycle_1",
        "confidence": confidence,
        "output": {
            "updated_recommendation": strategy,
            "confidence": confidence,
        },
    }


def test_convergence_true_when_agreement_high_and_severe_conflicts_low() -> None:
    status = evaluate_convergence(
        agent_outputs=[
            _round3("Finance Agent", "C"),
            _round3("Operations Agent", "C"),
            _round3("Technology Agent", "C"),
        ],
        conflict_map=[],
        cycle_id=1,
        agent_names=["Finance Agent", "Operations Agent", "Technology Agent"],
    )
    assert status.converged is True
    assert status.agreement_ratio == 1.0


def test_convergence_false_when_agreement_ratio_is_low() -> None:
    status = evaluate_convergence(
        agent_outputs=[
            _round3("Finance Agent", "A"),
            _round3("Operations Agent", "B"),
            _round3("Technology Agent", "C"),
        ],
        conflict_map=[],
        cycle_id=1,
        agent_names=["Finance Agent", "Operations Agent", "Technology Agent"],
    )
    assert status.converged is False
    assert status.agreement_ratio == 0.3333


def test_convergence_false_when_unresolved_high_severity_conflicts_are_too_many() -> None:
    conflicts = [
        ConflictItem(
            id=f"conflict-{index}",
            cycle_id=1,
            conflict_type="risk_cluster",
            topic="cash_flow",
            severity=4,
            summary="test",
        )
        for index in range(2)
    ]
    status = evaluate_convergence(
        agent_outputs=[
            _round3("Finance Agent", "C"),
            _round3("Operations Agent", "C"),
            _round3("Technology Agent", "C"),
        ],
        conflict_map=conflicts,
        cycle_id=1,
        agent_names=["Finance Agent", "Operations Agent", "Technology Agent"],
    )
    assert status.converged is False
    assert status.unresolved_high_severity_conflicts == 2


def test_max_cycle_logic_routes_to_round4_when_limit_reached() -> None:
    route = route_after_convergence(
        convergence={"converged": False},
        deliberation_cycle=2,
        max_deliberation_cycles=2,
    )
    assert route == "round_4_mcda_scorecard"

