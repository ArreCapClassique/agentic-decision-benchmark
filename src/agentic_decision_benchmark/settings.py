from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.2
    max_tokens: int = 1200
    run_output_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "runs")
    faulty_claim: str
    new_information: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    output_dir: str | None = None,
) -> BenchmarkSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    benchmark = load_yaml(PROJECT_ROOT / "config" / "benchmark.yaml")
    configured_provider = provider or os.getenv("MODEL_PROVIDER") or "ollama"
    configured_model = model or os.getenv("OLLAMA_MODEL") or "llama3.1:8b"
    configured_temperature = temperature
    if configured_temperature is None:
        configured_temperature = float(os.getenv("TEMPERATURE", "0.2"))
    configured_max_tokens = max_tokens
    if configured_max_tokens is None:
        configured_max_tokens = int(os.getenv("MAX_TOKENS", "1200"))
    configured_output_dir = output_dir or os.getenv("RUN_OUTPUT_DIR") or "runs"

    return BenchmarkSettings(
        provider=configured_provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=configured_model,
        temperature=configured_temperature,
        max_tokens=configured_max_tokens,
        run_output_dir=(PROJECT_ROOT / configured_output_dir).resolve()
        if not Path(configured_output_dir).is_absolute()
        else Path(configured_output_dir),
        faulty_claim=str(benchmark["faulty_claim"]),
        new_information=str(benchmark["new_information"]),
    )


def load_candidate_strategies() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "data" / "candidate_strategies.yaml")


def load_scenario() -> str:
    return load_text(PROJECT_ROOT / "data" / "eurotech_case.md")

