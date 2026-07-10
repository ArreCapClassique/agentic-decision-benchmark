from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agentic_decision_benchmark.agents.registry import load_agents
from agentic_decision_benchmark.graphs.common_nodes import make_agent_output, make_evaluator_node, provider_json
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.prompts import (
    build_domain_analysis_prompt,
    build_supervisor_plan_prompt,
    build_supervisor_synthesis_prompt,
)
from agentic_decision_benchmark.schemas import (
    DomainAnalysis,
    SupervisorPlan,
    SupervisorRecommendation,
    strategy_ids_from_candidates,
)
from agentic_decision_benchmark.settings import (
    BenchmarkSettings,
    format_role_private_brief,
    get_private_brief_for_role,
    load_private_role_briefs,
)
from agentic_decision_benchmark.state import BenchmarkState
from agentic_decision_benchmark.utils.json_utils import to_jsonable


def build_supervisor_graph(provider: LLMProvider, settings: BenchmarkSettings):
    agents = load_agents()
    private_briefs = load_private_role_briefs()

    def supervisor_plan_node(state: dict[str, Any]) -> dict[str, Any]:
        valid_strategy_ids = strategy_ids_from_candidates(state["candidate_strategies"])
        prompt = build_supervisor_plan_prompt(state["scenario"], state["candidate_strategies"])
        plan = provider_json(
            provider,
            prompt,
            SupervisorPlan,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            valid_strategy_ids=valid_strategy_ids,
        )
        output = make_agent_output(
            mode="supervisor",
            phase="supervisor_plan",
            agent="Supervisor Agent",
            round_id=0,
            output=plan,
            confidence=plan.confidence,
        )
        return {"agent_outputs": [output], "round_id": 0}

    def isolated_domain_analysis_node(state: dict[str, Any]) -> dict[str, Any]:
        outputs = []
        valid_strategy_ids = strategy_ids_from_candidates(state["candidate_strategies"])
        for agent in agents:
            inject_fault = state.get("fault_injection") and agent.name == "Technology Agent"
            role_brief = get_private_brief_for_role(private_briefs, agent.domain)
            prompt = build_domain_analysis_prompt(
                agent.definition,
                state["scenario"],
                state["candidate_strategies"],
                private_brief=format_role_private_brief(agent.domain, role_brief),
                faulty_claim=settings.faulty_claim if inject_fault else None,
            )
            analysis = provider_json(
                provider,
                prompt,
                DomainAnalysis,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                valid_strategy_ids=valid_strategy_ids,
            )
            if inject_fault and settings.faulty_claim not in analysis.claims:
                analysis.claims.append(settings.faulty_claim)
            outputs.append(
                make_agent_output(
                    mode="supervisor",
                    phase="isolated_domain_analysis",
                    agent=agent.name,
                    round_id=1,
                    output=analysis,
                    confidence=analysis.confidence,
                )
            )
        return {"agent_outputs": outputs, "round_id": 1}

    def supervisor_synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
        valid_strategy_ids = strategy_ids_from_candidates(state["candidate_strategies"])
        isolated_outputs = [
            to_jsonable(output)
            for output in state.get("agent_outputs", [])
            if getattr(output, "phase", None) == "isolated_domain_analysis"
            or (isinstance(output, dict) and output.get("phase") == "isolated_domain_analysis")
        ]
        new_information = settings.new_information if state.get("new_info_injection") else None
        prompt = build_supervisor_synthesis_prompt(
            state["scenario"],
            state["candidate_strategies"],
            isolated_outputs,
            new_information=new_information,
        )
        recommendation = provider_json(
            provider,
            prompt,
            SupervisorRecommendation,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            valid_strategy_ids=valid_strategy_ids,
        )
        output = make_agent_output(
            mode="supervisor",
            phase="supervisor_synthesis",
            agent="Supervisor Agent",
            round_id=2,
            output=recommendation,
            confidence=recommendation.confidence,
        )
        return {
            "agent_outputs": [output],
            "round_id": 2,
            "final_recommendation": recommendation.model_dump(mode="json"),
            "new_information_injected": bool(new_information),
        }

    graph = StateGraph(BenchmarkState)
    graph.add_node("supervisor_plan_node", supervisor_plan_node)
    graph.add_node("isolated_domain_analysis_node", isolated_domain_analysis_node)
    graph.add_node("supervisor_synthesis_node", supervisor_synthesis_node)
    graph.add_node("evaluator_node", make_evaluator_node(provider, settings))
    graph.add_edge(START, "supervisor_plan_node")
    graph.add_edge("supervisor_plan_node", "isolated_domain_analysis_node")
    graph.add_edge("isolated_domain_analysis_node", "supervisor_synthesis_node")
    graph.add_edge("supervisor_synthesis_node", "evaluator_node")
    graph.add_edge("evaluator_node", END)
    return graph.compile()

