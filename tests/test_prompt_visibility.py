from __future__ import annotations

from agentic_decision_benchmark.agents.registry import load_agents
from agentic_decision_benchmark.prompts import (
    build_domain_analysis_prompt,
    build_generalist_prompt,
    build_self_round1_prompt,
    build_self_round2_prompt,
    build_supervisor_synthesis_prompt,
)
from agentic_decision_benchmark.schemas import AgentDefinition, BlackboardItem
from agentic_decision_benchmark.settings import (
    build_consolidated_private_brief_pack,
    format_role_private_brief,
    get_private_brief_for_role,
    load_candidate_strategies,
    load_private_role_briefs,
    load_scenario,
)


def _agent(name: str) -> AgentDefinition:
    for agent in load_agents():
        if agent.name == name:
            return agent.definition
    raise AssertionError(f"Missing agent {name}")


def _private_brief_for(agent: AgentDefinition) -> str:
    briefs = load_private_role_briefs()
    role_brief = get_private_brief_for_role(briefs, agent.domain)
    return format_role_private_brief(agent.domain, role_brief)


def test_single_prompt_contains_all_private_role_sections() -> None:
    briefs = load_private_role_briefs()
    prompt = build_generalist_prompt(
        load_scenario(),
        load_candidate_strategies(),
        consolidated_private_brief_pack=build_consolidated_private_brief_pack(briefs),
    )
    for section in ["[Finance]", "[Operations]", "[Human Resources]", "[Legal / Compliance]", "[Strategy]", "[Technology]"]:
        assert section in prompt
    for item_id in ["FIN-001", "OPS-001", "HR-001", "LEG-001", "STR-001", "TECH-001"]:
        assert item_id in prompt


def test_single_prompt_can_receive_new_information() -> None:
    prompt = build_generalist_prompt(
        load_scenario(),
        load_candidate_strategies(),
        consolidated_private_brief_pack="brief pack",
        new_information="The automotive OEM extends the compliance deadline from 18 months to 24 months.",
    )

    assert "NEW_INFORMATION" in prompt
    assert "24 months" in prompt


def test_supervisor_finance_prompt_contains_only_finance_private_brief() -> None:
    finance = _agent("Finance Agent")
    prompt = build_domain_analysis_prompt(
        finance,
        load_scenario(),
        load_candidate_strategies(),
        private_brief=_private_brief_for(finance),
    )
    assert "FIN-001" in prompt
    assert "FIN-002" in prompt
    for excluded_id in ["OPS-001", "HR-001", "LEG-001", "STR-001", "TECH-001"]:
        assert excluded_id not in prompt


def test_supervisor_synthesis_prompt_can_receive_new_information() -> None:
    prompt = build_supervisor_synthesis_prompt(
        load_scenario(),
        load_candidate_strategies(),
        [{"agent": "Finance Agent", "output": {"recommendation": "C"}}],
        new_information="The automotive OEM extends the compliance deadline from 18 months to 24 months.",
    )

    assert "NEW_INFORMATION" in prompt
    assert "24 months" in prompt


def test_self_round1_technology_prompt_contains_only_technology_private_brief() -> None:
    technology = _agent("Technology Agent")
    prompt = build_self_round1_prompt(
        technology,
        load_scenario(),
        load_candidate_strategies(),
        _private_brief_for(technology),
    )
    assert "TECH-001" in prompt
    assert "TECH-002" in prompt
    for excluded_id in ["FIN-001", "OPS-001", "HR-001", "LEG-001", "STR-001"]:
        assert excluded_id not in prompt


def test_self_round2_prompt_contains_blackboard_and_own_private_brief() -> None:
    technology = _agent("Technology Agent")
    blackboard_item = BlackboardItem(
        round_id=1,
        author="Finance Agent",
        item_type="claim",
        content="Visible blackboard content from a prior round.",
        confidence=0.8,
    )
    prompt = build_self_round2_prompt(
        technology,
        load_scenario(),
        load_candidate_strategies(),
        _private_brief_for(technology),
        [blackboard_item],
    )
    assert "TECH-001" in prompt
    assert "Visible blackboard content from a prior round." in prompt
    assert "Select up to 3 blackboard items to critique." in prompt
    assert "No supervisor assigns your critique targets." in prompt
    assert "FIN-001" not in prompt
