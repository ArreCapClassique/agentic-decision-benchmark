from __future__ import annotations

import json

from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.prompts import build_generalist_prompt


def test_mock_provider_uses_candidate_strategy_context() -> None:
    strategies = {
        "A": {"name": "A option", "description": "A description", "core_tradeoff": "A tradeoff"},
        "B": {"name": "B option", "description": "B description", "core_tradeoff": "B tradeoff"},
        "C": {
            "name": "Sequential real-options strategy",
            "description": "Pursue an OEM pilot first, then commit after explicit gates.",
            "core_tradeoff": "Preserves flexibility but may satisfy neither counterparty quickly.",
        },
        "D": {"name": "D option", "description": "D description", "core_tradeoff": "D tradeoff"},
    }
    prompt = build_generalist_prompt("Scenario", strategies)

    output = json.loads(MockProvider().generate(prompt, temperature=0.2, max_tokens=100))
    output_text = json.dumps(output)

    assert "Sequential real-options strategy" in output_text
    assert "Pursue an OEM pilot first" in output_text
    assert "dual-track" not in output_text.lower()
