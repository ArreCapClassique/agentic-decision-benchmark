from __future__ import annotations

from agentic_decision_benchmark.agents.base import DomainAgent
from agentic_decision_benchmark.schemas import AgentDefinition
from agentic_decision_benchmark.settings import PROJECT_ROOT, load_yaml


def load_agents() -> list[DomainAgent]:
    data = load_yaml(PROJECT_ROOT / "config" / "agents.yaml")
    agents = [DomainAgent(AgentDefinition.model_validate(item)) for item in data.get("agents", [])]
    if len(agents) != 6:
        raise ValueError(f"Expected exactly six domain agents, found {len(agents)}")
    return agents

