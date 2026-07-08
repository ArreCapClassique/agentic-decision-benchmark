from __future__ import annotations

from agentic_decision_benchmark.graphs.self_organizing_graph import build_self_organizing_graph
from agentic_decision_benchmark.graphs.single_graph import build_single_graph
from agentic_decision_benchmark.graphs.supervisor_graph import build_supervisor_graph
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.settings import load_candidate_strategies, load_scenario, load_settings
from agentic_decision_benchmark.state import create_initial_state


def _settings(tmp_path):
    return load_settings(provider="mock", output_dir=str(tmp_path))


def _initial(mode: str, *, fault: bool = False, new_info: bool = False):
    return create_initial_state(
        mode=mode,
        scenario=load_scenario(),
        candidate_strategies=load_candidate_strategies(),
        fault_injection=fault,
        new_info_injection=new_info,
    )


def test_single_graph_compiles_and_runs_with_mock_provider(tmp_path) -> None:
    graph = build_single_graph(MockProvider(), _settings(tmp_path))
    state = graph.invoke(_initial("single"))
    assert state["final_recommendation"]["recommended_strategy"] == "C"
    assert state["evaluation"]["overall"] == 4
    assert state["metadata"]["information_setting"] == "equal_total"
    assert state["metadata"]["private_role_briefs_enabled"] is True


def test_supervisor_graph_compiles_and_runs_with_mock_provider(tmp_path) -> None:
    graph = build_supervisor_graph(MockProvider(), _settings(tmp_path))
    state = graph.invoke(_initial("supervisor"))
    assert state["final_recommendation"]["recommended_strategy"] == "C"
    assert state["evaluation"]["overall"] == 4
    assert len([item for item in state["agent_outputs"] if item.phase == "isolated_domain_analysis"]) == 6
    assert state["metadata"]["information_setting"] == "equal_total"
    assert state["metadata"]["private_role_briefs_enabled"] is True


def test_self_organizing_graph_runs_with_blackboard_scorecards_and_consensus(tmp_path) -> None:
    graph = build_self_organizing_graph(MockProvider(), _settings(tmp_path))
    state = graph.invoke(_initial("self_organizing", fault=True, new_info=True))
    rounds = {item.round_id for item in state["blackboard"]}
    assert {1, 2, 3, 4}.issubset(rounds)
    assert len(state["scorecards"]) == 6
    assert state["final_recommendation"]["recommended_strategy"] == "C"
    assert state["final_recommendation"]["agreement_ratio"] == 1.0
    assert state["salience_map"]
    assert state["conflict_map"]
    assert any(item.conflict_type == "direct_critique" for item in state["conflict_map"])
    assert state["convergence"]
    assert state["convergence_history"]
    assert state["deliberation_cycle"] == state["max_deliberation_cycles"]
    assert any(item.item_type == "critique" for item in state["blackboard"])
    assert any(item.item_type == "new_information" for item in state["blackboard"])
    assert any(item.item_type == "scorecard" for item in state["blackboard"])
    assert state["metadata"]["information_setting"] == "equal_total"
    assert state["metadata"]["private_role_briefs_enabled"] is True

