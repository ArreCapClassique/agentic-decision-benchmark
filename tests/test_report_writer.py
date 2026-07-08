from __future__ import annotations

from agentic_decision_benchmark.settings import load_candidate_strategies
from agentic_decision_benchmark.state import create_initial_state
from agentic_decision_benchmark.storage.report_writer import write_comparative_report


def test_report_includes_information_setting_and_synthetic_private_briefs(tmp_path) -> None:
    state = create_initial_state(
        mode="single",
        scenario="case",
        candidate_strategies=load_candidate_strategies(),
    )
    state["final_recommendation"] = {"recommended_strategy": "C"}
    state["metrics"] = {}
    state["evaluation"] = {}

    path = write_comparative_report(tmp_path, {"single": state})
    report = path.read_text(encoding="utf-8")

    assert "Information setting: equal_total" in report
    assert "synthetic private role briefs" in report
    assert "Self-Organizing Deliberation Loop" in report
    assert "deterministic conflict map" in report
