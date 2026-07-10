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
    state["evaluation"] = {
        "decision_quality": 4,
        "convergence": 3.6,
        "resilience": 3.1,
        "explainability": 3.4,
        "adaptability": 3.0,
        "cost_efficiency": 4.8,
        "runtime_efficiency": 4.7,
        "overall": 3.7,
    }

    path = write_comparative_report(tmp_path, {"single": state})
    report = path.read_text(encoding="utf-8")

    assert "Information setting: equal_total" in report
    assert "synthetic private role briefs" in report
    assert "Initial Domain Preference Diversity" in report
    assert "Initial Organization Preference Distribution" in report
    assert "Self-Organizing Deliberation Loop" in report
    assert "deterministic conflict map" in report
    assert "Cost Efficiency" in report
    assert "Runtime Efficiency" in report
    assert "| single | 4.0 | 3.6 | 3.1 | 3.4 | 3.0 | 4.8 | 4.7 | 3.7 |" in report
