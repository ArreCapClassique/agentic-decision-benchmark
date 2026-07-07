from __future__ import annotations

import pytest

from agentic_decision_benchmark.schemas import BlackboardItem
from agentic_decision_benchmark.state import append_reducer, create_initial_state, validate_mode


def test_state_initializes_correctly() -> None:
    state = create_initial_state(
        mode="self_organizing",
        scenario="case",
        candidate_strategies={"A": {}},
    )
    assert state["mode"] == "self_organizing"
    assert state["blackboard"] == []
    assert state["agent_outputs"] == []
    assert state["scorecards"] == []
    assert state["fault_injection"] is False
    assert state["metadata"]["information_setting"] == "equal_total"
    assert state["metadata"]["private_role_briefs_enabled"] is True


def test_append_reducer_does_not_overwrite_blackboard_items() -> None:
    first = [BlackboardItem(round_id=1, author="A", item_type="claim", content="one", confidence=0.7)]
    second = [BlackboardItem(round_id=2, author="B", item_type="critique", content="two", confidence=0.8)]
    merged = append_reducer(first, second)
    assert [item.content for item in merged] == ["one", "two"]


def test_mode_validation() -> None:
    assert validate_mode("single") == "single"
    assert validate_mode("supervisor") == "supervisor"
    assert validate_mode("self_organizing") == "self_organizing"
    with pytest.raises(ValueError):
        validate_mode("swarm")

