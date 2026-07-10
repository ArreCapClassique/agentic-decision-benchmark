from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from agentic_decision_benchmark.evaluation.evaluator import Evaluator
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.schemas import (
    AgentOutput,
    ModeName,
    StrategyId,
    normalize_strategy_id,
    normalize_strategy_id_list,
    strategy_id_label,
)
from agentic_decision_benchmark.settings import BenchmarkSettings
from agentic_decision_benchmark.utils.json_utils import parse_model_json, to_jsonable


STRATEGY_FIELD_NAMES = {
    "recommended_strategy",
    "domain_preferred_strategy",
    "organization_preferred_strategy",
    "updated_recommendation",
    "related_strategy",
    "leading_strategy",
    "final_selected_strategy",
}
STRATEGY_LIST_FIELD_NAMES = {
    "domain_ranking",
    "organization_ranking",
    "strategies_supported",
    "strategies_challenged",
    "strategies_conditioned",
}
STRATEGY_KEYED_MAP_FIELD_NAMES = {
    "conditions_for_accepting_other_strategy",
}


def _schema_repair_prompt(
    *,
    failed_response: str,
    schema: type[BaseModel],
    error: ValueError,
    valid_strategy_ids: tuple[StrategyId, ...] | None = None,
) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)
    lines = [
        "PREVIOUS_MODEL_OUTPUT_FAILED_SCHEMA_VALIDATION",
        "Your previous output was valid JSON but did not match the required schema.",
        "Preserve the same substantive content, but correct only the JSON shape and enum values.",
        "Return a corrected JSON object only. Do not include Markdown or explanations.",
        "If a schema field is an array, return a JSON array even when it has only one item.",
    ]
    if valid_strategy_ids:
        label = strategy_id_label(valid_strategy_ids)
        lines.extend(
            [
                f"Allowed strategy IDs: {label}.",
                f"If a strategy field expects a strategy ID, return exactly one of: {label}.",
                f"For scorecard schemas, scores must contain exactly these keys: {label}.",
            ]
        )
    else:
        lines.append("If a strategy field expects a strategy ID, return the exact strategy key from candidate_strategies.")
    lines.extend(
        [
            "For scorecard schemas, each strategy score must contain criteria, overall, and rationale.",
            "For scorecard schemas, criteria must contain the five integer keys financial_viability, operational_feasibility, strategic_value, workforce_feasibility, and legal_compliance_safety.",
            f"SCHEMA_NAME: {schema.__name__}",
            f"VALIDATION_ERROR: {error}",
            f"PREVIOUS_OUTPUT: {failed_response}",
            f"JSON_SCHEMA: {schema_json}",
        ]
    )
    return "\n".join(lines)


def _validate_strategy_id(value: Any, valid_strategy_ids: tuple[StrategyId, ...], path: str) -> None:
    if value is None:
        return
    strategy_id = normalize_strategy_id(value)
    if strategy_id not in set(valid_strategy_ids):
        raise ValueError(
            f"{path} must be one of {strategy_id_label(valid_strategy_ids)}, got {value!r}."
        )


def _validate_strategy_id_list(value: Any, valid_strategy_ids: tuple[StrategyId, ...], path: str) -> None:
    expected = set(valid_strategy_ids)
    for index, strategy_id in enumerate(normalize_strategy_id_list(value)):
        if strategy_id not in expected:
            raise ValueError(
                f"{path}[{index}] must be one of {strategy_id_label(valid_strategy_ids)}, got {strategy_id!r}."
            )


def _validate_strategy_keyed_map(value: Any, valid_strategy_ids: tuple[StrategyId, ...], path: str) -> None:
    if not isinstance(value, dict):
        return
    for key in value:
        _validate_strategy_id(key, valid_strategy_ids, f"{path}.{key}")


def _walk_strategy_fields(value: Any, valid_strategy_ids: tuple[StrategyId, ...], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in STRATEGY_FIELD_NAMES:
                _validate_strategy_id(item, valid_strategy_ids, child_path)
            if key in STRATEGY_LIST_FIELD_NAMES:
                _validate_strategy_id_list(item, valid_strategy_ids, child_path)
            if key in STRATEGY_KEYED_MAP_FIELD_NAMES:
                _validate_strategy_keyed_map(item, valid_strategy_ids, child_path)
            _walk_strategy_fields(item, valid_strategy_ids, child_path)
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_strategy_fields(item, valid_strategy_ids, f"{path}[{index}]")


def _validate_strategy_output(
    output: BaseModel,
    schema: type[BaseModel],
    valid_strategy_ids: tuple[StrategyId, ...] | None,
) -> None:
    if not valid_strategy_ids:
        return
    payload = to_jsonable(output)
    _walk_strategy_fields(payload, valid_strategy_ids)
    if schema.__name__ == "DomainAnalysis" and isinstance(payload, dict):
        _validate_strategy_id(payload.get("recommendation"), valid_strategy_ids, "$.recommendation")
    if schema.__name__ == "Round4ScorecardOutput" and isinstance(payload, dict):
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("$.scores must be an object keyed by strategy ID.")
        expected = set(valid_strategy_ids)
        actual = {strategy_id for key in scores if (strategy_id := normalize_strategy_id(key)) is not None}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "Scorecard scores must contain exactly "
                f"{strategy_id_label(valid_strategy_ids)}; missing={missing}, extra={extra}."
            )


def provider_json(
    provider: LLMProvider,
    prompt: str,
    schema: type[BaseModel],
    *,
    temperature: float,
    max_tokens: int,
    valid_strategy_ids: tuple[StrategyId, ...] | None = None,
) -> BaseModel:
    response = provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    try:
        parsed = parse_model_json(response, schema)  # type: ignore[assignment]
        _validate_strategy_output(parsed, schema, valid_strategy_ids)
        return parsed
    except ValueError as first_error:
        repair_prompt = _schema_repair_prompt(
            failed_response=response,
            schema=schema,
            error=first_error,
            valid_strategy_ids=valid_strategy_ids,
        )
        repaired_response = provider.generate(repair_prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            repaired = parse_model_json(repaired_response, schema)  # type: ignore[assignment]
            _validate_strategy_output(repaired, schema, valid_strategy_ids)
            return repaired
        except ValueError as second_error:
            raise ValueError(
                f"Model output failed {schema.__name__} validation after repair retry. "
                f"Initial error: {first_error}. Repair error: {second_error}"
            ) from second_error


def make_agent_output(
    *,
    mode: ModeName,
    phase: str,
    agent: str,
    round_id: int,
    output: BaseModel | dict[str, Any],
    confidence: float,
) -> AgentOutput:
    return AgentOutput(
        mode=mode,
        phase=phase,
        agent=agent,
        round_id=round_id,
        output=to_jsonable(output),
        confidence=confidence,
    )


def make_evaluator_node(provider: LLMProvider, settings: BenchmarkSettings):
    def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
        evaluator = Evaluator(provider, temperature=settings.temperature, max_tokens=settings.max_tokens)
        evaluation = evaluator.evaluate(state)
        return {"evaluation": evaluation.model_dump(mode="json")}

    return evaluator_node

