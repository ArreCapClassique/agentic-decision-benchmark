from __future__ import annotations

import pytest

from agentic_decision_benchmark.schemas import BlackboardItem


def test_blackboard_item_schema_validates() -> None:
    item = BlackboardItem(
        round_id=2,
        author="Finance Agent",
        item_type="critique",
        target_agent="Technology Agent",
        target_claim="Deployment is easy.",
        content="This underestimates integration cost.",
        confidence=0.78,
        related_strategy="A",
        severity=4,
        topic_tags=["AI feasibility", "invalid tag"],
        criterion="operational_feasibility",
        stance="challenges",
        target_item_id="bb-target",
    )
    assert item.id
    assert item.round_id == 2
    assert item.cycle_id == 0
    assert item.author == "Finance Agent"
    assert item.item_type == "critique"
    assert item.target_agent == "Technology Agent"
    assert item.target_claim == "Deployment is easy."
    assert item.target_item_id == "bb-target"
    assert item.content == "This underestimates integration cost."
    assert item.confidence == 0.78
    assert item.related_strategy == "A"
    assert item.severity == 4
    assert item.topic_tags == ["ai_feasibility"]
    assert item.criterion == "operational_feasibility"
    assert item.stance == "challenges"


def test_blackboard_item_falls_back_to_uncategorized_when_no_valid_tags() -> None:
    item = BlackboardItem(
        round_id=1,
        author="Strategy Agent",
        item_type="claim",
        content="test",
        confidence=0.7,
        topic_tags=["not a valid tag"],
    )
    assert item.topic_tags == ["uncategorized"]


def test_blackboard_item_rejects_invalid_stance() -> None:
    with pytest.raises(ValueError):
        BlackboardItem(
            round_id=1,
            author="Strategy Agent",
            item_type="claim",
            content="test",
            confidence=0.7,
            stance="maybe",
        )
