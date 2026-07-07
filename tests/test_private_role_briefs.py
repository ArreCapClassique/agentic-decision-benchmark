from __future__ import annotations

from agentic_decision_benchmark.settings import ROLE_PRIVATE_BRIEF_KEYS, load_private_role_briefs


def test_private_role_briefs_load_with_all_roles() -> None:
    briefs = load_private_role_briefs()
    assert set(briefs.model_dump()) == set(ROLE_PRIVATE_BRIEF_KEYS)


def test_private_role_brief_items_have_required_fields() -> None:
    briefs = load_private_role_briefs()
    for role_brief in briefs.model_dump().values():
        assert role_brief["visibility"]
        assert role_brief["synthetic"] is True
        for item in role_brief["brief_items"]:
            assert item["id"]
            assert item["source_label"]
            assert item["synthetic"] is True
            assert item["statement"]
            assert item["certainty"] in {"low", "medium", "high"}
            assert item["relevance"]
