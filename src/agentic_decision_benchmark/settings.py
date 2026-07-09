from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentic_decision_benchmark.schemas import PrivateRoleBriefs, RolePrivateBrief


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_PRIVATE_BRIEF_KEYS: tuple[str, ...] = ("finance", "operations", "hr", "legal", "strategy", "technology")
ROLE_SECTION_TITLES = {
    "finance": "Finance",
    "operations": "Operations",
    "hr": "Human Resources",
    "legal": "Legal / Compliance",
    "strategy": "Strategy",
    "technology": "Technology",
}
ROLE_ALIASES = {
    "finance": "finance",
    "operations": "operations",
    "human resources": "hr",
    "hr": "hr",
    "legal compliance": "legal",
    "legal": "legal",
    "compliance": "legal",
    "strategy": "strategy",
    "technology": "technology",
}


class BenchmarkSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.2
    max_tokens: int = 1200
    run_output_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "runs")
    fault_injection_enabled: bool = False
    new_info_injection_enabled: bool = False
    faulty_claim: str
    new_information: str
    max_deliberation_cycles: int = 2


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _normalize_role(role: str) -> str:
    return " ".join(role.lower().replace("/", " ").replace("_", " ").split())


def _resolve_private_brief_key(role: str) -> str:
    normalized = _normalize_role(role)
    try:
        return ROLE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown private brief role {role!r}. Expected one of {sorted(ROLE_ALIASES)}") from exc


def load_private_role_briefs(path: str | Path = PROJECT_ROOT / "data" / "private_role_briefs.yaml") -> PrivateRoleBriefs:
    return PrivateRoleBriefs.model_validate(load_yaml(Path(path)))


def get_private_brief_for_role(briefs: PrivateRoleBriefs, role: str) -> RolePrivateBrief:
    return getattr(briefs, _resolve_private_brief_key(role))


def _format_private_brief_items(title: str, brief: RolePrivateBrief) -> str:
    lines = [
        f"[{title}]",
        f"visibility: {brief.visibility}",
        f"synthetic: {str(brief.synthetic).lower()}",
    ]
    for item in brief.brief_items:
        lines.extend(
            [
                f"- {item.id} | {item.source_label} | certainty: {item.certainty} | synthetic: {str(item.synthetic).lower()}",
                f"  {item.statement}",
                f"  relevance: {', '.join(item.relevance)}",
            ]
        )
    return "\n".join(lines)


def format_role_private_brief(role: str, brief: RolePrivateBrief) -> str:
    role_key = _resolve_private_brief_key(role)
    return _format_private_brief_items(ROLE_SECTION_TITLES[role_key], brief)


def build_consolidated_private_brief_pack(briefs: PrivateRoleBriefs) -> str:
    lines = [
        "PRIVATE ROLE-SPECIFIC INFORMATION PACK",
        "Synthetic case information used for benchmark purposes only.",
        "",
    ]
    for role_key in ROLE_PRIVATE_BRIEF_KEYS:
        lines.append(_format_private_brief_items(ROLE_SECTION_TITLES[role_key], getattr(briefs, role_key)))
        lines.append("")
    return "\n".join(lines).strip()


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
        fault_injection_enabled=bool(benchmark.get("fault_injection_enabled", False)),
        new_info_injection_enabled=bool(benchmark.get("new_info_injection_enabled", False)),
        faulty_claim=str(benchmark["faulty_claim"]),
        new_information=str(benchmark["new_information"]),
        max_deliberation_cycles=int(benchmark.get("max_deliberation_cycles", 2)),
    )


def load_candidate_strategies() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "data" / "candidate_strategies.yaml")


def load_scenario() -> str:
    return load_text(PROJECT_ROOT / "data" / "eurotech_case.md")

