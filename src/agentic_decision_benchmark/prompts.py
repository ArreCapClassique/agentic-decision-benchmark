from __future__ import annotations

import json
from typing import Any

from agentic_decision_benchmark.schemas import AgentDefinition, BlackboardItem
from agentic_decision_benchmark.utils.json_utils import to_jsonable


STRICT_JSON_RULES = (
    "Output valid JSON only. Do not include Markdown. Do not include explanations outside JSON. "
    "Stay within the assigned role. Use only the provided scenario, candidate strategies, and visible blackboard data. "
    "Calibrate confidence between 0 and 1. Avoid inventing facts not present in the scenario."
)


def _payload(data: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(data), indent=2, sort_keys=True)


def _base_context(scenario: str, candidate_strategies: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "candidate_strategies": candidate_strategies,
    }


def build_generalist_prompt(
    scenario: str,
    candidate_strategies: dict[str, Any],
    *,
    faulty_claim: str | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: GENERALIST_RECOMMENDATION",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Generalist strategy consultant.",
            "Return schema keys: recommended_strategy, recommendation, reasoning, risks, assumptions, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"FAULT_CONTEXT: {faulty_claim or 'None'}",
        ]
    )


def build_supervisor_plan_prompt(scenario: str, candidate_strategies: dict[str, Any]) -> str:
    return "\n".join(
        [
            "TASK: SUPERVISOR_PLAN",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Supervisor. Plan isolated expert work; do not decide yet.",
            "Return schema keys: plan, domains_to_consult, decision_criteria, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
        ]
    )


def build_domain_analysis_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    *,
    faulty_claim: str | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: ISOLATED_DOMAIN_ANALYSIS",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Isolated analysis. You cannot see other agents and must not critique them.",
            f"AGENT_PROFILE: {_payload(agent.model_dump())}",
            "Return schema keys: claims, risks, opportunities, assumptions, recommendation, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"FAULT_CONTEXT: {faulty_claim or 'None'}",
        ]
    )


def build_supervisor_synthesis_prompt(
    scenario: str,
    candidate_strategies: dict[str, Any],
    isolated_outputs: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "TASK: SUPERVISOR_SYNTHESIS",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Supervisor. You are the central decision-maker for this mode.",
            "Return schema keys: recommended_strategy, recommendation, synthesis, tradeoffs, risks, assumptions, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"ISOLATED_OUTPUTS: {_payload({'outputs': isolated_outputs})}",
        ]
    )


def build_self_round1_prompt(agent: AgentDefinition, scenario: str, candidate_strategies: dict[str, Any]) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_1",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 1 independent analysis. Do not reference other agents because you have not seen their outputs.",
            f"AGENT_PROFILE: {_payload(agent.model_dump())}",
            "Return schema keys: claims, risks, opportunities, assumptions, questions_for_others, initial_recommendation, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
        ]
    )


def build_self_round2_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    blackboard: list[BlackboardItem],
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_2",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 2 cross-critique. Critique claims, risks, assumptions, or recommendations from other agents.",
            "Return schema key: critiques. Each critique needs target_agent, target_claim, critique, missing_assumption, risk_introduced, severity, confidence, related_strategy.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"VISIBLE_BLACKBOARD: {_payload({'blackboard': blackboard})}",
        ]
    )


def build_self_round3_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    blackboard: list[BlackboardItem],
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_3",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 3 belief update. Read all previous blackboard items, critiques, and new information. Update your position without restarting from scratch.",
            "Return schema keys: updated_recommendation, changed_beliefs, accepted_critiques, rejected_critiques, remaining_concerns, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"VISIBLE_BLACKBOARD: {_payload({'blackboard': blackboard})}",
        ]
    )


def build_self_round4_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    blackboard: list[BlackboardItem],
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_4",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 4 MCDA scorecard. Score A, B, C, and D using the shared criteria.",
            "Scores must be integers from 1 to 5. Criteria: financial_viability, operational_feasibility, strategic_value, workforce_feasibility, legal_compliance_safety.",
            "Return schema keys: scores, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"VISIBLE_BLACKBOARD: {_payload({'blackboard': blackboard})}",
        ]
    )


def build_evaluator_prompt(
    final_recommendation: dict[str, Any],
    deterministic_metrics: dict[str, Any],
    rubric: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "TASK: EVALUATOR",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Neutral evaluator. Do not infer or reveal which architecture produced the answer.",
            "Return schema keys: decision_quality, convergence, resilience, explainability, adaptability, cost_efficiency, runtime_efficiency, overall, rationale.",
            f"RUBRIC: {_payload(rubric)}",
            f"FINAL_RECOMMENDATION: {_payload(final_recommendation)}",
            f"DETERMINISTIC_METRICS: {_payload(deterministic_metrics)}",
        ]
    )

