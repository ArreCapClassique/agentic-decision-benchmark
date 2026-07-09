from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agentic_decision_benchmark.agents.registry import load_agents
from agentic_decision_benchmark.consensus.mcda import run_mcda_consensus
from agentic_decision_benchmark.graphs.common_nodes import make_agent_output, make_evaluator_node, provider_json
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.prompts import (
    build_self_round1_prompt,
    build_self_round2_prompt,
    build_self_round3_prompt,
    build_self_round4_prompt,
)
from agentic_decision_benchmark.schemas import (
    BlackboardItem,
    BlackboardItemType,
    BlackboardStance,
    BlackboardContribution,
    Critique,
    SalienceRecord,
    Round1Analysis,
    Round2CritiqueOutput,
    Round3BeliefUpdate,
    Round4ScorecardOutput,
    Scorecard,
    StrategyId,
    deterministic_blackboard_id,
)
from agentic_decision_benchmark.self_organization.conflict_map import build_conflict_map
from agentic_decision_benchmark.self_organization.convergence import evaluate_convergence, route_after_convergence
from agentic_decision_benchmark.self_organization.salience import compute_salience_map, rank_salient_items_for_agent
from agentic_decision_benchmark.settings import (
    BenchmarkSettings,
    format_role_private_brief,
    get_private_brief_for_role,
    load_private_role_briefs,
)
from agentic_decision_benchmark.state import BenchmarkState
from agentic_decision_benchmark.utils.json_utils import to_jsonable


def _item_id(round_id: int, cycle_id: int, author: str, item_type: str, index: int, content: str) -> str:
    return deterministic_blackboard_id([round_id, cycle_id, author, item_type, index, content])


def _default_topic_tags(content: str, item_type: BlackboardItemType) -> list[str]:
    text = content.lower()
    if any(word in text for word in ["cash", "capex", "investment", "revenue", "financial"]):
        return ["financial_viability"]
    if any(word in text for word in ["retool", "production", "operations", "capacity"]):
        return ["operational_feasibility"]
    if any(word in text for word in ["ai", "data", "sensor", "model", "technology"]):
        return ["ai_feasibility"]
    if any(word in text for word in ["hiring", "skill", "workforce", "training"]):
        return ["workforce_feasibility"]
    if any(word in text for word in ["contract", "compliance", "audit", "legal"]):
        return ["compliance"]
    if any(word in text for word in ["market", "customer", "wind", "diversification", "strategy"]):
        return ["strategic_value"]
    if item_type == "risk":
        return ["market_uncertainty"]
    return ["uncategorized"]


def _default_stance(item_type: BlackboardItemType) -> BlackboardStance:
    if item_type in {"claim", "opportunity", "scorecard", "consensus"}:
        return "supports"
    if item_type == "risk":
        return "opposes"
    if item_type == "critique":
        return "challenges"
    return "neutral"


def _make_blackboard_item(
    *,
    round_id: int,
    cycle_id: int,
    author: str,
    item_type: BlackboardItemType,
    content: str,
    index: int,
    confidence: float,
    topic_tags: list[str] | None = None,
    related_strategy: StrategyId | None = None,
    criterion: str | None = None,
    stance: BlackboardStance | None = None,
    severity: int | None = None,
    target_agent: str | None = None,
    target_claim: str | None = None,
    target_item_id: str | None = None,
) -> BlackboardItem:
    return BlackboardItem(
        id=_item_id(round_id, cycle_id, author, item_type, index, content),
        round_id=round_id,
        cycle_id=cycle_id,
        author=author,
        item_type=item_type,
        content=content,
        topic_tags=topic_tags or _default_topic_tags(content, item_type),
        related_strategy=related_strategy,
        criterion=criterion,
        stance=stance or _default_stance(item_type),
        confidence=confidence,
        target_agent=target_agent,
        target_claim=target_claim,
        target_item_id=target_item_id,
        severity=severity,
    )


def _strategy_label(candidate_strategies: dict[str, Any], strategy_id: str | None) -> str:
    if not strategy_id:
        return "unresolved"
    details = candidate_strategies.get(strategy_id, {})
    name = details.get("name") if isinstance(details, dict) else None
    return f"Strategy {strategy_id} ({name})" if name else f"Strategy {strategy_id}"


def _round1_to_blackboard(
    agent_name: str,
    output: Round1Analysis,
    cycle_id: int,
    candidate_strategies: dict[str, Any],
) -> list[BlackboardItem]:
    items: list[BlackboardItem] = []
    if output.blackboard_items:
        for index, contribution in enumerate(output.blackboard_items, start=1):
            items.append(
                _make_blackboard_item(
                    round_id=1,
                    cycle_id=cycle_id,
                    author=agent_name,
                    item_type=contribution.item_type,
                    content=contribution.content,
                    index=index,
                    confidence=contribution.confidence if contribution.confidence is not None else output.confidence,
                    topic_tags=list(contribution.topic_tags),
                    related_strategy=contribution.related_strategy,
                    criterion=contribution.criterion,
                    stance=contribution.stance,
                    severity=contribution.severity,
                )
            )
    else:
        index = 1
        for item_type, contents in [
            ("claim", output.claims),
            ("risk", output.risks),
            ("opportunity", output.opportunities),
            ("assumption", output.assumptions),
            ("question", output.questions_for_others),
        ]:
            for content in contents:
                items.append(
                    _make_blackboard_item(
                        round_id=1,
                        cycle_id=cycle_id,
                        author=agent_name,
                        item_type=item_type,  # type: ignore[arg-type]
                        content=content,
                        index=index,
                        confidence=output.confidence,
                        severity=3 if item_type == "risk" else None,
                    )
                )
                index += 1
    items.append(
        _make_blackboard_item(
            round_id=1,
            cycle_id=cycle_id,
            author=agent_name,
            item_type="claim",
            content=f"Initial recommendation is {_strategy_label(candidate_strategies, output.initial_recommendation)}.",
            index=999,
            confidence=output.confidence,
            topic_tags=["strategic_value"],
            related_strategy=output.initial_recommendation,
            criterion="strategic_value",
            stance="supports",
        )
    )
    return items


def _find_target_item_id(blackboard: list[BlackboardItem], critique: Critique) -> str | None:
    if critique.target_item_id:
        return critique.target_item_id
    for item in blackboard:
        if item.author == critique.target_agent and item.content == critique.target_claim:
            return item.id
    for item in blackboard:
        if item.author == critique.target_agent and critique.target_claim and critique.target_claim in item.content:
            return item.id
    return None


def _critique_to_blackboard(
    agent_name: str,
    critique: Critique,
    cycle_id: int,
    index: int,
    blackboard: list[BlackboardItem],
) -> BlackboardItem:
    return _make_blackboard_item(
        round_id=2,
        cycle_id=cycle_id,
        author=agent_name,
        item_type="critique",
        content=critique.critique,
        index=index,
        topic_tags=list(critique.topic_tags),
        criterion=critique.criterion,
        stance="challenges",
        target_agent=critique.target_agent,
        target_claim=critique.target_claim,
        target_item_id=_find_target_item_id(blackboard, critique),
        confidence=critique.confidence,
        related_strategy=critique.related_strategy,
        severity=critique.severity,
    )


def _belief_update_to_blackboard(agent_name: str, output: Round3BeliefUpdate, cycle_id: int) -> list[BlackboardItem]:
    items: list[BlackboardItem] = []
    index = 1
    for content in output.changed_beliefs + output.accepted_critiques + output.rejected_critiques:
        items.append(
            _make_blackboard_item(
                round_id=3,
                cycle_id=cycle_id,
                author=agent_name,
                item_type="belief_update",
                content=content,
                index=index,
                confidence=output.confidence,
                topic_tags=_default_topic_tags(content, "belief_update"),
                related_strategy=output.updated_recommendation,
                stance="neutral",
            )
        )
        index += 1
    for content in output.remaining_concerns:
        items.append(
            _make_blackboard_item(
                round_id=3,
                cycle_id=cycle_id,
                author=agent_name,
                item_type="minority_concern",
                content=content,
                index=index,
                confidence=output.confidence,
                topic_tags=_default_topic_tags(content, "minority_concern"),
                related_strategy=output.updated_recommendation,
                stance="challenges",
                severity=3,
            )
        )
        index += 1
    return items


def _scorecard_blackboard_item(
    agent_name: str,
    scorecard: Scorecard,
    cycle_id: int,
    candidate_strategies: dict[str, Any],
) -> BlackboardItem:
    top_strategy, top_score = sorted(scorecard.scores.items(), key=lambda item: (-item[1].overall, item[0]))[0]
    return _make_blackboard_item(
        round_id=4,
        cycle_id=cycle_id,
        author=agent_name,
        item_type="scorecard",
        content=(
            f"{agent_name} top-scored {_strategy_label(candidate_strategies, top_strategy)} "
            f"with overall score {top_score.overall}."
        ),
        index=1,
        confidence=scorecard.confidence,
        topic_tags=["strategic_value"],
        related_strategy=top_strategy,
        criterion="strategic_value",
        stance="supports",
    )


def _previous_recommendation_for_agent(agent_outputs: list[Any], agent_name: str) -> str | None:
    previous: str | None = None
    for raw_output in agent_outputs:
        output = to_jsonable(raw_output)
        if output.get("agent") != agent_name or not isinstance(output.get("output"), dict):
            continue
        payload = output["output"]
        if output.get("phase") == "round_1_independent_analysis":
            previous = payload.get("initial_recommendation")
        if str(output.get("phase", "")).startswith("round_3_belief_update"):
            previous = payload.get("updated_recommendation")
    return previous


def _critiques_targeting_agent_claims(blackboard: list[BlackboardItem], agent_name: str) -> list[dict[str, Any]]:
    own_item_ids = {item.id for item in blackboard if item.author == agent_name}
    return [
        to_jsonable(item)
        for item in blackboard
        if item.item_type == "critique" and item.target_item_id in own_item_ids
    ]


def _unresolved_high_severity_conflicts(conflict_map: list[Any]) -> list[dict[str, Any]]:
    conflicts = [to_jsonable(item) for item in conflict_map]
    return [
        item
        for item in conflicts
        if int(item.get("severity", 0)) >= 4 and item.get("status") == "unresolved"
    ]


def _graph_route_after_convergence(state: dict[str, Any]) -> str:
    return route_after_convergence(
        convergence=state.get("convergence", {}),
        deliberation_cycle=int(state.get("deliberation_cycle", 0)),
        max_deliberation_cycles=int(state.get("max_deliberation_cycles", 2)),
    )


def build_self_organizing_graph(provider: LLMProvider, settings: BenchmarkSettings):
    agents = load_agents()
    private_briefs = load_private_role_briefs()
    agent_domains = [agent.domain for agent in agents]
    agent_names = [agent.name for agent in agents]

    def private_brief_for_agent(agent_domain: str) -> str:
        role_brief = get_private_brief_for_role(private_briefs, agent_domain)
        return format_role_private_brief(agent_domain, role_brief)

    def round_1_independent_analysis(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        cycle_id = 1
        for agent in agents:
            prompt = build_self_round1_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
            )
            analysis = provider_json(
                provider,
                prompt,
                Round1Analysis,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            if (
                state.get("fault_injection")
                and agent.name == "Technology Agent"
                and settings.faulty_claim not in analysis.claims
            ):
                analysis.claims.append(settings.faulty_claim)
                analysis.blackboard_items.append(
                    BlackboardContribution(
                        item_type="claim",
                        content=settings.faulty_claim,
                        topic_tags=["ai_feasibility", "implementation_timeline"],
                        related_strategy="A",
                        criterion="operational_feasibility",
                        stance="supports",
                        confidence=analysis.confidence,
                        severity=5,
                    )
                )
            outputs.append(
                make_agent_output(
                    mode="self_organizing",
                    phase="round_1_independent_analysis",
                    agent=agent.name,
                    round_id=1,
                    output=analysis,
                    confidence=analysis.confidence,
                )
            )
            blackboard_items.extend(_round1_to_blackboard(agent.name, analysis, cycle_id, state["candidate_strategies"]))
        return {
            "agent_outputs": outputs,
            "blackboard": blackboard_items,
            "round_id": 1,
            "deliberation_cycle": cycle_id,
            "max_deliberation_cycles": settings.max_deliberation_cycles,
        }

    def update_blackboard_salience(state: dict[str, Any]) -> dict[str, Any]:
        salience_map = compute_salience_map(state.get("blackboard", []), agent_domains)
        return {"salience_map": salience_map}

    def round_2_agent_selected_critique(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        cycle_id = int(state.get("deliberation_cycle", 1))
        blackboard = state.get("blackboard", [])
        salience_map: dict[str, SalienceRecord] = state.get("salience_map", {})
        for agent in agents:
            salience_ranked_items = rank_salient_items_for_agent(
                blackboard,
                salience_map,
                agent.domain,
                limit=8,
            )
            prompt = build_self_round2_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
                blackboard,
                salience_ranked_items,
                [to_jsonable(item) for item in state.get("conflict_map", [])],
            )
            critique_output = provider_json(
                provider,
                prompt,
                Round2CritiqueOutput,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            outputs.append(
                make_agent_output(
                    mode="self_organizing",
                    phase=f"round_2_agent_selected_critique_cycle_{cycle_id}",
                    agent=agent.name,
                    round_id=2,
                    output=critique_output,
                    confidence=max([item.confidence for item in critique_output.critiques], default=0.6),
                )
            )
            blackboard_items.extend(
                _critique_to_blackboard(agent.name, item, cycle_id, index, blackboard)
                for index, item in enumerate(critique_output.critiques, start=1)
            )
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "round_id": 2}

    def optional_new_information_injection(state: dict[str, Any]) -> dict[str, Any]:
        if (
            not state.get("new_info_injection")
            or state.get("new_information_injected")
            or int(state.get("deliberation_cycle", 1)) != 1
        ):
            return {"round_id": 2}
        cycle_id = int(state.get("deliberation_cycle", 1))
        return {
            "blackboard": [
                _make_blackboard_item(
                    round_id=2,
                    cycle_id=cycle_id,
                    author="System",
                    item_type="new_information",
                    content=settings.new_information,
                    index=1,
                    confidence=1.0,
                    topic_tags=["implementation_timeline", "compliance"],
                    stance="neutral",
                )
            ],
            "new_information_injected": True,
            "round_id": 2,
        }

    def build_deterministic_conflict_map(state: dict[str, Any]) -> dict[str, Any]:
        cycle_id = int(state.get("deliberation_cycle", 1))
        return {"conflict_map": build_conflict_map(state.get("blackboard", []), cycle_id)}

    def round_3_belief_update(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        cycle_id = int(state.get("deliberation_cycle", 1))
        conflict_map = [to_jsonable(item) for item in state.get("conflict_map", [])]
        unresolved_conflicts = _unresolved_high_severity_conflicts(state.get("conflict_map", []))
        for agent in agents:
            prompt = build_self_round3_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
                state.get("blackboard", []),
                conflict_map,
                _previous_recommendation_for_agent(state.get("agent_outputs", []), agent.name),
                _critiques_targeting_agent_claims(state.get("blackboard", []), agent.name),
                unresolved_conflicts,
            )
            update = provider_json(
                provider,
                prompt,
                Round3BeliefUpdate,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            outputs.append(
                make_agent_output(
                    mode="self_organizing",
                    phase=f"round_3_belief_update_cycle_{cycle_id}",
                    agent=agent.name,
                    round_id=3,
                    output=update,
                    confidence=update.confidence,
                )
            )
            blackboard_items.extend(_belief_update_to_blackboard(agent.name, update, cycle_id))
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "round_id": 3}

    def check_convergence(state: dict[str, Any]) -> dict[str, Any]:
        cycle_id = int(state.get("deliberation_cycle", 1))
        convergence = evaluate_convergence(
            agent_outputs=state.get("agent_outputs", []),
            conflict_map=state.get("conflict_map", []),
            cycle_id=cycle_id,
            agent_names=agent_names,
        )
        return {
            "convergence": convergence,
            "convergence_history": [convergence],
        }

    def increment_deliberation_cycle(state: dict[str, Any]) -> dict[str, Any]:
        return {"deliberation_cycle": int(state.get("deliberation_cycle", 1)) + 1}

    def round_4_mcda_scorecard(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        scorecards: list[Scorecard] = []
        cycle_id = int(state.get("deliberation_cycle", 1))
        for agent in agents:
            prompt = build_self_round4_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
                state.get("blackboard", []),
            )
            card_output = provider_json(
                provider,
                prompt,
                Round4ScorecardOutput,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            scorecard = Scorecard(agent=agent.name, round_id=4, scores=card_output.scores, confidence=card_output.confidence)
            scorecards.append(scorecard)
            outputs.append(
                make_agent_output(
                    mode="self_organizing",
                    phase="round_4_mcda_scorecard",
                    agent=agent.name,
                    round_id=4,
                    output=scorecard,
                    confidence=scorecard.confidence,
                )
            )
            blackboard_items.append(_scorecard_blackboard_item(agent.name, scorecard, cycle_id, state["candidate_strategies"]))
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "scorecards": scorecards, "round_id": 4}

    def consensus_node(state: dict[str, Any]) -> dict[str, Any]:
        consensus = run_mcda_consensus(
            scorecards=state.get("scorecards", []),
            candidate_strategies=state["candidate_strategies"],
            blackboard=state.get("blackboard", []),
        )
        content = (
            f"Consensus selected {_strategy_label(state['candidate_strategies'], consensus.recommended_strategy)}."
            if consensus.recommended_strategy
            else f"Consensus unresolved among {', '.join(consensus.unresolved_strategies)}."
        )
        return {
            "final_recommendation": consensus.model_dump(mode="json"),
            "blackboard": [
                _make_blackboard_item(
                    round_id=4,
                    cycle_id=int(state.get("deliberation_cycle", 1)),
                    author="Consensus Engine",
                    item_type="consensus",
                    content=content,
                    index=1,
                    confidence=consensus.confidence,
                    topic_tags=["strategic_value"],
                    related_strategy=consensus.recommended_strategy,
                    criterion="strategic_value",
                    stance="supports",
                )
            ],
        }

    graph = StateGraph(BenchmarkState)
    graph.add_node("round_1_independent_analysis", round_1_independent_analysis)
    graph.add_node("update_blackboard_salience", update_blackboard_salience)
    graph.add_node("round_2_agent_selected_critique", round_2_agent_selected_critique)
    graph.add_node("optional_new_information_injection", optional_new_information_injection)
    graph.add_node("build_deterministic_conflict_map", build_deterministic_conflict_map)
    graph.add_node("round_3_belief_update", round_3_belief_update)
    graph.add_node("check_convergence", check_convergence)
    graph.add_node("increment_deliberation_cycle", increment_deliberation_cycle)
    graph.add_node("round_4_mcda_scorecard", round_4_mcda_scorecard)
    graph.add_node("consensus_node", consensus_node)
    graph.add_node("evaluator_node", make_evaluator_node(provider, settings))
    graph.add_edge(START, "round_1_independent_analysis")
    graph.add_edge("round_1_independent_analysis", "update_blackboard_salience")
    graph.add_edge("update_blackboard_salience", "round_2_agent_selected_critique")
    graph.add_edge("round_2_agent_selected_critique", "optional_new_information_injection")
    graph.add_edge("optional_new_information_injection", "build_deterministic_conflict_map")
    graph.add_edge("build_deterministic_conflict_map", "round_3_belief_update")
    graph.add_edge("round_3_belief_update", "check_convergence")
    graph.add_conditional_edges(
        "check_convergence",
        _graph_route_after_convergence,
        {
            "increment_deliberation_cycle": "increment_deliberation_cycle",
            "round_4_mcda_scorecard": "round_4_mcda_scorecard",
        },
    )
    graph.add_edge("increment_deliberation_cycle", "update_blackboard_salience")
    graph.add_edge("round_4_mcda_scorecard", "consensus_node")
    graph.add_edge("consensus_node", "evaluator_node")
    graph.add_edge("evaluator_node", END)
    return graph.compile()

