from __future__ import annotations

from agentic_decision_benchmark.main import build_parser
from agentic_decision_benchmark.settings import load_settings


def test_settings_load_stress_test_switch_defaults(tmp_path) -> None:
    settings = load_settings(provider="mock", output_dir=str(tmp_path))

    assert settings.fault_injection_enabled is False
    assert settings.new_info_injection_enabled is False


def test_stress_test_flags_can_be_enabled_and_disabled() -> None:
    parser = build_parser()

    defaults = parser.parse_args(["run-all"])
    assert defaults.fault_injection is None
    assert defaults.new_info is None

    enabled = parser.parse_args(["run-all", "--fault-injection", "--new-info"])
    assert enabled.fault_injection is True
    assert enabled.new_info is True

    disabled = parser.parse_args(
        [
            "run-all",
            "--fault-injection",
            "--new-info",
            "--no-fault-injection",
            "--no-new-info",
        ]
    )
    assert disabled.fault_injection is False
    assert disabled.new_info is False
