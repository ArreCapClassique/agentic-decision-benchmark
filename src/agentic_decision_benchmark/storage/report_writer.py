from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from agentic_decision_benchmark.schemas import MODE_NAMES
from agentic_decision_benchmark.utils.json_utils import to_jsonable


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator] + body)


def _strategy_name(strategies: dict[str, Any], strategy_id: str | None) -> str:
    if not strategy_id:
        return "unresolved"
    details = strategies.get(strategy_id, {})
    return f"{strategy_id}: {details.get('name', strategy_id)}"


def _mode_rows(states: dict[str, dict[str, Any]], strategies: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for mode in MODE_NAMES:
        state = states.get(mode)
        if not state:
            rows.append([mode, "not run", "-", "-", "-"])
            continue
        final = state.get("final_recommendation", {})
        metrics = state.get("metrics", {})
        evaluation = state.get("evaluation", {})
        rows.append(
            [
                mode,
                _strategy_name(strategies, final.get("recommended_strategy")),
                metrics.get("model_calls", "-"),
                metrics.get("total_estimated_tokens", "-"),
                evaluation.get("overall", "-"),
            ]
        )
    return rows


def _self_blackboard_summary(state: dict[str, Any] | None) -> str:
    if not state:
        return "The `self_organizing` mode was not run."
    blackboard = [to_jsonable(item) for item in state.get("blackboard", [])]
    counts = Counter(item.get("item_type") for item in blackboard)
    rows = [[key, value] for key, value in sorted(counts.items())]
    return _table(["Item type", "Count"], rows)


def _items_by_type(state: dict[str, Any] | None, item_type: str, limit: int = 8) -> list[str]:
    if not state:
        return []
    items = []
    for item in [to_jsonable(entry) for entry in state.get("blackboard", [])]:
        if item.get("item_type") == item_type:
            author = item.get("author", "?")
            content = item.get("content", "")
            items.append(f"- {author}: {content}")
        if len(items) >= limit:
            break
    return items


def _scorecard_summary(state: dict[str, Any] | None) -> str:
    if not state:
        return "No scorecards were produced."
    final = state.get("final_recommendation", {})
    aggregate = final.get("aggregate_scores", {})
    if not aggregate:
        return "No aggregate MCDA scores were produced."
    rows = []
    for strategy, score in aggregate.items():
        rows.append([strategy, score.get("mean"), score.get("stdev"), score.get("top_votes")])
    return _table(["Strategy", "Mean", "Stdev", "Top votes"], rows)


def write_comparative_report(run_dir: Path, states: dict[str, dict[str, Any]]) -> Path:
    first_state = next(iter(states.values()))
    strategies = first_state.get("candidate_strategies", {})
    self_state = states.get("self_organizing")

    candidate_lines = [
        f"- {key}: {value.get('name')} - {value.get('core_tradeoff')}"
        for key, value in strategies.items()
    ]

    report = [
        "# Agentic Decision Benchmark Comparative Report",
        "",
        "## Scenario Summary",
        "EuroTech must respond to an OEM predictive quality requirement that puts 30% of revenue at risk while evaluating a wind turbine partnership requiring retooling and hiring.",
        "",
        "## Candidate Strategies",
        *candidate_lines,
        "",
        "## Architecture Comparison",
        _table(
            ["Mode", "Decision structure"],
            [
                ["single", "One generalist agent produces the final answer."],
                ["supervisor", "A supervisor centrally synthesizes isolated domain-agent outputs."],
                ["self_organizing", "Agents deliberate through blackboard rounds and deterministic MCDA consensus."],
            ],
        ),
        "",
        "## Why Delphi-Blackboard-MCDA",
        "The `self_organizing` mode uses a Delphi-Blackboard Swarm with MCDA Consensus rather than free-form chat so that independence, critique, belief updates, scorecards, and consensus are all inspectable artifacts.",
        "",
        "## Results Table",
        _table(["Mode", "Final recommendation", "Model calls", "Estimated tokens", "Overall evaluation"], _mode_rows(states, strategies)),
        "",
        "## Evaluation Table",
        _table(
            ["Mode", "Decision", "Convergence", "Resilience", "Explainability", "Adaptability", "Overall"],
            [
                [
                    mode,
                    states.get(mode, {}).get("evaluation", {}).get("decision_quality", "-"),
                    states.get(mode, {}).get("evaluation", {}).get("convergence", "-"),
                    states.get(mode, {}).get("evaluation", {}).get("resilience", "-"),
                    states.get(mode, {}).get("evaluation", {}).get("explainability", "-"),
                    states.get(mode, {}).get("evaluation", {}).get("adaptability", "-"),
                    states.get(mode, {}).get("evaluation", {}).get("overall", "-"),
                ]
                for mode in MODE_NAMES
            ],
        ),
        "",
        "## Final Recommendations By Mode",
        *[
            f"- {mode}: {_strategy_name(strategies, states.get(mode, {}).get('final_recommendation', {}).get('recommended_strategy'))}"
            for mode in MODE_NAMES
        ],
        "",
        "## self_organizing Blackboard Summary",
        _self_blackboard_summary(self_state),
        "",
        "## Cross-Critique Summary",
        *(_items_by_type(self_state, "critique") or ["No critique items were recorded."]),
        "",
        "## Belief Update Summary",
        *(_items_by_type(self_state, "belief_update") or ["No belief update items were recorded."]),
        "",
        "## MCDA Scorecard Summary",
        _scorecard_summary(self_state),
        "",
        "## Convergence Analysis",
        f"Agreement ratio: {self_state.get('final_recommendation', {}).get('agreement_ratio', '-') if self_state else '-'}. The report preserves score variance and minority concerns instead of hiding disagreement.",
        "",
        "## Resilience Analysis",
        "Fault correction is measured by whether unsupported or overconfident injected claims are challenged in critiques or belief updates.",
        "",
        "## Adaptability Analysis",
        "When new information is enabled, the `self_organizing` mode injects it after cross-critique and incorporates it during belief update without restarting the run.",
        "",
        "## Strengths And Weaknesses",
        "- `single` is efficient but has limited peer challenge.",
        "- `supervisor` adds domain breadth but centralizes synthesis authority.",
        "- `self_organizing` improves auditability and critique coverage at the cost of more calls and protocol complexity.",
        "",
        "## Conclusion",
        "Structured self-organization appears most useful when the problem benefits from independent expertise, explicit challenge, and auditable consensus. It may be worse than supervisor orchestration when time, model-call cost, or simple accountability matter more than deliberative traceability.",
        "",
        "## Runtime Orchestration Versus Decision-Making",
        "LangGraph controls execution order in all modes. In `supervisor`, a supervisor agent is the central decision-maker. In `self_organizing`, the graph only enforces protocol timing; the final answer is selected from scorecards by deterministic MCDA aggregation.",
    ]

    path = run_dir / "comparative_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path

