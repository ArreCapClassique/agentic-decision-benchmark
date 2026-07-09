from __future__ import annotations

import argparse
import time
from typing import Any

from agentic_decision_benchmark.evaluation.metrics import compute_metrics
from agentic_decision_benchmark.graphs.self_organizing_graph import build_self_organizing_graph
from agentic_decision_benchmark.graphs.single_graph import build_single_graph
from agentic_decision_benchmark.graphs.supervisor_graph import build_supervisor_graph
from agentic_decision_benchmark.llm.factory import create_provider
from agentic_decision_benchmark.schemas import MODE_NAMES
from agentic_decision_benchmark.settings import (
    load_candidate_strategies,
    load_private_role_briefs,
    load_scenario,
    load_settings,
)
from agentic_decision_benchmark.state import create_initial_state
from agentic_decision_benchmark.storage.report_writer import write_comparative_report
from agentic_decision_benchmark.storage.run_store import RunStore


def _build_graph(mode: str, provider, settings):
    if mode == "single":
        return build_single_graph(provider, settings)
    if mode == "supervisor":
        return build_supervisor_graph(provider, settings)
    if mode == "self_organizing":
        return build_self_organizing_graph(provider, settings)
    raise ValueError(f"Unsupported mode: {mode}")


def run_mode(
    *,
    mode: str,
    settings,
    scenario: str,
    candidate_strategies: dict[str, Any],
    fault_injection: bool,
    new_info: bool,
) -> dict[str, Any]:
    provider = create_provider(settings)
    graph = _build_graph(mode, provider, settings)
    initial_state = create_initial_state(
        mode=mode,
        scenario=scenario,
        candidate_strategies=candidate_strategies,
        fault_injection=fault_injection,
        new_info_injection=new_info,
    )
    started = time.perf_counter()
    state = graph.invoke(initial_state)
    runtime = time.perf_counter() - started
    metrics = compute_metrics(state, provider, runtime_seconds=runtime)
    state["metrics"] = metrics.model_dump(mode="json")
    return state


def run_command(args: argparse.Namespace) -> int:
    settings = load_settings(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )
    scenario = load_scenario()
    candidate_strategies = load_candidate_strategies()
    modes = list(MODE_NAMES) if args.command == "run-all" else [args.mode]
    fault_injection = settings.fault_injection_enabled if args.fault_injection is None else args.fault_injection
    new_info = settings.new_info_injection_enabled if args.new_info is None else args.new_info
    store = RunStore(settings.run_output_dir)
    store.save_private_role_briefs(load_private_role_briefs())
    states: dict[str, dict[str, Any]] = {}

    for mode in modes:
        state = run_mode(
            mode=mode,
            settings=settings,
            scenario=scenario,
            candidate_strategies=candidate_strategies,
            fault_injection=fault_injection,
            new_info=new_info,
        )
        states[mode] = state
        store.save_mode(mode, state)

    write_comparative_report(store.run_dir, states)
    print(f"Saved run artifacts to {store.run_dir}")
    return 0


def _add_common_options(parser: argparse.ArgumentParser, mode: str | None = None) -> None:
    parser.set_defaults(func=run_command)
    if mode is not None:
        parser.set_defaults(mode=mode)
    parser.add_argument("--provider", choices=["mock", "ollama", "openai"], default=None)
    parser.add_argument("--model", default=None, help="Override the configured model name.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fault-injection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configured faulty-claim resilience stress test switch.",
    )
    parser.add_argument(
        "--new-info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configured new-information adaptability stress test switch.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the agentic decision benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_single = subparsers.add_parser("run-single")
    _add_common_options(run_single, "single")

    run_supervisor = subparsers.add_parser("run-supervisor")
    _add_common_options(run_supervisor, "supervisor")

    run_self = subparsers.add_parser("run-self-organizing")
    _add_common_options(run_self, "self_organizing")

    run_all = subparsers.add_parser("run-all")
    _add_common_options(run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

