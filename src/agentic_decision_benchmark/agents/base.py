from __future__ import annotations

from dataclasses import dataclass

from agentic_decision_benchmark.schemas import AgentDefinition


@dataclass(frozen=True)
class DomainAgent:
    definition: AgentDefinition

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def domain(self) -> str:
        return self.definition.domain

