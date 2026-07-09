from __future__ import annotations

import json
from typing import Any

from agentic_decision_benchmark.schemas import AgentDefinition, BlackboardItem, TOPIC_TAGS
from agentic_decision_benchmark.utils.json_utils import to_jsonable


STRICT_JSON_RULES = (
    "Output valid JSON only. Do not include Markdown. Do not include explanations outside JSON. "
    "When a return field expects a list, output a JSON array of strings or objects, not a single string. "
    "For recommended_strategy, initial_recommendation, and updated_recommendation, output exactly one strategy ID: A, B, C, or D. "
    "Stay within the assigned role. Use only the provided scenario, candidate strategies, private brief or consolidated private pack when present, visible blackboard data, fault context, and new information. "
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
    consolidated_private_brief_pack: str | None = None,
    faulty_claim: str | None = None,
    new_information: str | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: GENERALIST_RECOMMENDATION",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Generalist strategy consultant.",
            "Return schema keys: recommended_strategy, recommendation, reasoning, risks, assumptions, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            "CONSOLIDATED PRIVATE ROLE-SPECIFIC INFORMATION PACK",
            "The following synthetic information combines all role-specific private briefs. Use it as the full information pack available to the generalist baseline.",
            consolidated_private_brief_pack or "None",
            f"FAULT_CONTEXT: {faulty_claim or 'None'}",
            "NEW_INFORMATION",
            "Late-arriving information visible to this mode. If present, update reasoning, risks, assumptions, or confidence to reflect it.",
            new_information or "None",
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
    private_brief: str | None = None,
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
            "PRIVATE ROLE-SPECIFIC BRIEF",
            "The following information is synthetic private case knowledge visible to your role only.",
            "Use it as partial internal organizational knowledge.",
            "Do not assume the supervisor or other agents know it unless you include it in your isolated output.",
            "Preserve uncertainty and source labels where relevant.",
            private_brief or "None",
            f"FAULT_CONTEXT: {faulty_claim or 'None'}",
        ]
    )


def build_supervisor_synthesis_prompt(
    scenario: str,
    candidate_strategies: dict[str, Any],
    isolated_outputs: list[dict[str, Any]],
    *,
    new_information: str | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: SUPERVISOR_SYNTHESIS",
            f"RULES: {STRICT_JSON_RULES}",
            "ROLE: Supervisor. You are the central decision-maker for this mode.",
            "Return schema keys: recommended_strategy, recommendation, synthesis, tradeoffs, risks, assumptions, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            f"ISOLATED_OUTPUTS: {_payload({'outputs': isolated_outputs})}",
            "NEW_INFORMATION",
            "Late-arriving information visible to the supervisor during final synthesis. If present, update the synthesis without rerunning isolated expert work.",
            new_information or "None",
        ]
    )


def _private_brief_section(private_brief: str | None) -> list[str]:
    return [
        "PRIVATE ROLE-SPECIFIC BRIEF",
        "The following information is synthetic private case knowledge visible to your role only.",
        "Use it as partial internal organizational knowledge.",
        "You only know your own private brief directly.",
        "You may learn other roles' private information only if they have posted it to the blackboard.",
        "Preserve uncertainty and source labels where relevant.",
        private_brief or "None",
    ]


def build_self_round1_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    private_brief: str | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_1",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 1 independent analysis. Do not reference other agents because you have not seen their outputs.",
            f"AGENT_PROFILE: {_payload(agent.model_dump())}",
            "Return schema keys: claims, risks, opportunities, assumptions, questions_for_others, initial_recommendation, blackboard_items, confidence.",
            "Optional blackboard_items entries use keys: item_type, content, topic_tags, related_strategy, criterion, stance, confidence, severity.",
            f"Allowed topic_tags: {', '.join(TOPIC_TAGS)}.",
            "Allowed stance values: supports, opposes, neutral, challenges.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            *_private_brief_section(private_brief),
        ]
    )


def build_self_round2_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    private_brief: str | None,
    blackboard: list[BlackboardItem],
    salience_ranked_items: list[dict[str, Any]] | None = None,
    conflict_history: list[dict[str, Any]] | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_2",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 2 agent-selected critique. Critique claims, risks, assumptions, or recommendations from other agents.",
            "Select up to 3 blackboard items to critique.",
            "Prioritize items that are:",
            "1. relevant to your domain;",
            "2. high salience;",
            "3. high severity;",
            "4. high confidence but weakly supported;",
            "5. contradictory to your private role brief;",
            "6. important to one or more candidate strategies.",
            "No supervisor assigns your critique targets. You choose them based on local domain rules and blackboard signals.",
            "Return schema keys: agent_name, selected_items, critiques.",
            "Each critique needs target_item_id, target_agent, target_claim, critique, missing_assumption, risk_introduced, topic_tags, related_strategy, criterion, severity, confidence.",
            f"Allowed topic_tags: {', '.join(TOPIC_TAGS)}.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            *_private_brief_section(private_brief),
            f"VISIBLE_BLACKBOARD: {_payload({'blackboard': blackboard})}",
            f"SALIENCE_RANKED_ITEMS: {_payload({'items': salience_ranked_items or []})}",
            f"CONFLICT_HISTORY: {_payload({'conflicts': conflict_history or []})}",
        ]
    )


def build_self_round3_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    private_brief: str | None,
    blackboard: list[BlackboardItem],
    conflict_map: list[dict[str, Any]] | None = None,
    previous_recommendation: str | None = None,
    critiques_targeting_own_claims: list[dict[str, Any]] | None = None,
    unresolved_high_severity_conflicts: list[dict[str, Any]] | None = None,
) -> str:
    return "\n".join(
        [
            "TASK: SELF_ORGANIZING_ROUND_3",
            f"AGENT_NAME: {agent.name}",
            f"AGENT_DOMAIN: {agent.domain}",
            f"RULES: {STRICT_JSON_RULES}",
            "PHASE: Round 3 belief update. Read all previous blackboard items, critiques, conflict map, and new information. Update your position without restarting from scratch.",
            "Update your position based on accepted critiques, rejected critiques, conflict map, newly revealed private information from other agents through the blackboard, and unresolved high-severity risks.",
            "Return schema keys: agent_name, updated_recommendation, changed_beliefs, accepted_critiques, rejected_critiques, remaining_concerns, confidence.",
            f"CONTEXT: {_payload(_base_context(scenario, candidate_strategies))}",
            *_private_brief_section(private_brief),
            f"PREVIOUS_OWN_RECOMMENDATION: {previous_recommendation or 'None'}",
            f"VISIBLE_BLACKBOARD: {_payload({'blackboard': blackboard})}",
            f"CURRENT_CONFLICT_MAP: {_payload({'conflicts': conflict_map or []})}",
            f"CRITIQUES_TARGETING_OWN_CLAIMS: {_payload({'critiques': critiques_targeting_own_claims or []})}",
            f"UNRESOLVED_HIGH_SEVERITY_CONFLICTS: {_payload({'conflicts': unresolved_high_severity_conflicts or []})}",
        ]
    )


def build_self_round4_prompt(
    agent: AgentDefinition,
    scenario: str,
    candidate_strategies: dict[str, Any],
    private_brief: str | None,
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
            *_private_brief_section(private_brief),
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
            "Use decimal scores from 1.0 to 5.0 for every score field.",
            "You may use one decimal place, for example 3.4, 4.2, or 4.7.",
            "Do not use only whole numbers unless the output exactly matches a category boundary.",
            "A score of 4.0 means strong. A score of 4.5 means between strong and excellent. A score of 3.5 means between adequate and strong.",
            "Use DETERMINISTIC_METRICS as scoring evidence, not as background decoration.",
            "Cost efficiency must reflect model_calls and total_estimated_tokens.",
            "Runtime efficiency must reflect runtime_seconds when it is nonzero; otherwise use model_calls and token counts as a proxy.",
            "Explainability should reflect audit artifacts such as blackboard items, conflicts, scorecards, and salience records.",
            "Return valid JSON only.",
            f"RUBRIC: {_payload(rubric)}",
            f"FINAL_RECOMMENDATION: {_payload(final_recommendation)}",
            f"DETERMINISTIC_METRICS: {_payload(deterministic_metrics)}",
        ]
    )

