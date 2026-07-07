from __future__ import annotations

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
    )
    assert item.round_id == 2
    assert item.author == "Finance Agent"
    assert item.item_type == "critique"
    assert item.target_agent == "Technology Agent"
    assert item.target_claim == "Deployment is easy."
    assert item.content == "This underestimates integration cost."
    assert item.confidence == 0.78
    assert item.related_strategy == "A"
    assert item.severity == 4

