from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agentic_decision_benchmark.graphs.common_nodes import make_agent_output, make_evaluator_node, provider_json
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.prompts import build_generalist_prompt
from agentic_decision_benchmark.schemas import SingleRecommendation
from agentic_decision_benchmark.settings import BenchmarkSettings
from agentic_decision_benchmark.state import BenchmarkState


def build_single_graph(provider: LLMProvider, settings: BenchmarkSettings):
    def generalist_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        faulty_claim = settings.faulty_claim if state.get("fault_injection") else None
        prompt = build_generalist_prompt(
            state["scenario"],
            state["candidate_strategies"],
            faulty_claim=faulty_claim,
        )
        recommendation = provider_json(
            provider,
            prompt,
            SingleRecommendation,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        if faulty_claim and not any(faulty_claim in item for item in recommendation.risks + recommendation.assumptions):
            recommendation.assumptions.append(f"Faulty information supplied for resilience test: {faulty_claim}")
        output = make_agent_output(
            mode="single",
            phase="generalist_recommendation",
            agent="Generalist Strategy Agent",
            round_id=1,
            output=recommendation,
            confidence=recommendation.confidence,
        )
        return {
            "round_id": 1,
            "agent_outputs": [output],
            "final_recommendation": recommendation.model_dump(mode="json"),
        }

    graph = StateGraph(BenchmarkState)
    graph.add_node("generalist_agent_node", generalist_agent_node)
    graph.add_node("evaluator_node", make_evaluator_node(provider, settings))
    graph.add_edge(START, "generalist_agent_node")
    graph.add_edge("generalist_agent_node", "evaluator_node")
    graph.add_edge("evaluator_node", END)
    return graph.compile()

