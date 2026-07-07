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
    Critique,
    Round1Analysis,
    Round2CritiqueOutput,
    Round3BeliefUpdate,
    Round4ScorecardOutput,
    Scorecard,
)
from agentic_decision_benchmark.settings import (
    BenchmarkSettings,
    format_role_private_brief,
    get_private_brief_for_role,
    load_private_role_briefs,
)
from agentic_decision_benchmark.state import BenchmarkState


def _round1_to_blackboard(agent_name: str, output: Round1Analysis) -> list[BlackboardItem]:
    items: list[BlackboardItem] = []
    for content in output.claims:
        items.append(BlackboardItem(round_id=1, author=agent_name, item_type="claim", content=content, confidence=output.confidence))
    for content in output.risks:
        items.append(BlackboardItem(round_id=1, author=agent_name, item_type="risk", content=content, confidence=output.confidence))
    for content in output.opportunities:
        items.append(BlackboardItem(round_id=1, author=agent_name, item_type="opportunity", content=content, confidence=output.confidence))
    for content in output.assumptions:
        items.append(BlackboardItem(round_id=1, author=agent_name, item_type="assumption", content=content, confidence=output.confidence))
    for content in output.questions_for_others:
        items.append(BlackboardItem(round_id=1, author=agent_name, item_type="question", content=content, confidence=output.confidence))
    items.append(
        BlackboardItem(
            round_id=1,
            author=agent_name,
            item_type="claim",
            content=f"Initial recommendation is Strategy {output.initial_recommendation}.",
            confidence=output.confidence,
            related_strategy=output.initial_recommendation,
        )
    )
    return items


def _critique_to_blackboard(agent_name: str, critique: Critique) -> BlackboardItem:
    return BlackboardItem(
        round_id=2,
        author=agent_name,
        item_type="critique",
        target_agent=critique.target_agent,
        target_claim=critique.target_claim,
        content=critique.critique,
        confidence=critique.confidence,
        related_strategy=critique.related_strategy,
        severity=critique.severity,
    )


def _belief_update_to_blackboard(agent_name: str, output: Round3BeliefUpdate) -> list[BlackboardItem]:
    items: list[BlackboardItem] = []
    for content in output.changed_beliefs + output.accepted_critiques + output.rejected_critiques:
        items.append(
            BlackboardItem(
                round_id=3,
                author=agent_name,
                item_type="belief_update",
                content=content,
                confidence=output.confidence,
                related_strategy=output.updated_recommendation,
            )
        )
    for content in output.remaining_concerns:
        items.append(
            BlackboardItem(
                round_id=3,
                author=agent_name,
                item_type="minority_concern",
                content=content,
                confidence=output.confidence,
                related_strategy=output.updated_recommendation,
            )
        )
    return items


def _scorecard_blackboard_item(agent_name: str, scorecard: Scorecard) -> BlackboardItem:
    top_strategy, top_score = sorted(scorecard.scores.items(), key=lambda item: (-item[1].overall, item[0]))[0]
    return BlackboardItem(
        round_id=4,
        author=agent_name,
        item_type="scorecard",
        content=f"{agent_name} top-scored Strategy {top_strategy} with overall score {top_score.overall}.",
        confidence=scorecard.confidence,
        related_strategy=top_strategy,
    )


def build_self_organizing_graph(provider: LLMProvider, settings: BenchmarkSettings):
    agents = load_agents()
    private_briefs = load_private_role_briefs()

    def private_brief_for_agent(agent_domain: str) -> str:
        role_brief = get_private_brief_for_role(private_briefs, agent_domain)
        return format_role_private_brief(agent_domain, role_brief)

    def round_1_independent_analysis(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
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
            if state.get("fault_injection") and agent.name == "Technology Agent" and settings.faulty_claim not in analysis.claims:
                analysis.claims.append(settings.faulty_claim)
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
            blackboard_items.extend(_round1_to_blackboard(agent.name, analysis))
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "round_id": 1}

    def round_2_cross_critique(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        for agent in agents:
            prompt = build_self_round2_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
                state.get("blackboard", []),
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
                    phase="round_2_cross_critique",
                    agent=agent.name,
                    round_id=2,
                    output=critique_output,
                    confidence=max([item.confidence for item in critique_output.critiques], default=0.6),
                )
            )
            blackboard_items.extend(_critique_to_blackboard(agent.name, item) for item in critique_output.critiques)
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "round_id": 2}

    def optional_new_information_injection(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("new_info_injection"):
            return {"round_id": 2}
        return {
            "blackboard": [
                BlackboardItem(
                    round_id=2,
                    author="System",
                    item_type="new_information",
                    content=settings.new_information,
                    confidence=1.0,
                )
            ],
            "round_id": 2,
        }

    def round_3_belief_update(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        for agent in agents:
            prompt = build_self_round3_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief_for_agent(agent.domain),
                state.get("blackboard", []),
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
                    phase="round_3_belief_update",
                    agent=agent.name,
                    round_id=3,
                    output=update,
                    confidence=update.confidence,
                )
            )
            blackboard_items.extend(_belief_update_to_blackboard(agent.name, update))
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "round_id": 3}

    def round_4_mcda_scorecard(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        blackboard_items: list[BlackboardItem] = []
        scorecards: list[Scorecard] = []
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
            blackboard_items.append(_scorecard_blackboard_item(agent.name, scorecard))
        return {"agent_outputs": outputs, "blackboard": blackboard_items, "scorecards": scorecards, "round_id": 4}

    def consensus_node(state: dict[str, Any]) -> dict[str, Any]:
        consensus = run_mcda_consensus(
            scorecards=state.get("scorecards", []),
            candidate_strategies=state["candidate_strategies"],
            blackboard=state.get("blackboard", []),
        )
        content = (
            f"Consensus selected Strategy {consensus.recommended_strategy}."
            if consensus.recommended_strategy
            else f"Consensus unresolved among {', '.join(consensus.unresolved_strategies)}."
        )
        return {
            "final_recommendation": consensus.model_dump(mode="json"),
            "blackboard": [
                BlackboardItem(
                    round_id=4,
                    author="Consensus Engine",
                    item_type="consensus",
                    content=content,
                    confidence=consensus.confidence,
                    related_strategy=consensus.recommended_strategy,
                )
            ],
        }

    graph = StateGraph(BenchmarkState)
    graph.add_node("round_1_independent_analysis", round_1_independent_analysis)
    graph.add_node("round_2_cross_critique", round_2_cross_critique)
    graph.add_node("optional_new_information_injection", optional_new_information_injection)
    graph.add_node("round_3_belief_update", round_3_belief_update)
    graph.add_node("round_4_mcda_scorecard", round_4_mcda_scorecard)
    graph.add_node("consensus_node", consensus_node)
    graph.add_node("evaluator_node", make_evaluator_node(provider, settings))
    graph.add_edge(START, "round_1_independent_analysis")
    graph.add_edge("round_1_independent_analysis", "round_2_cross_critique")
    graph.add_edge("round_2_cross_critique", "optional_new_information_injection")
    graph.add_edge("optional_new_information_injection", "round_3_belief_update")
    graph.add_edge("round_3_belief_update", "round_4_mcda_scorecard")
    graph.add_edge("round_4_mcda_scorecard", "consensus_node")
    graph.add_edge("consensus_node", "evaluator_node")
    graph.add_edge("evaluator_node", END)
    return graph.compile()

