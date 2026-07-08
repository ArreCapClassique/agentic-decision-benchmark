from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_decision_benchmark.schemas import MODE_NAMES
from agentic_decision_benchmark.utils.json_utils import write_json


class RunStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def save_private_role_briefs(self, private_role_briefs: Any) -> Path:
        path = self.run_dir / "private_role_briefs_used.json"
        write_json(path, private_role_briefs)
        return path

    def save_mode(self, mode: str, state: dict[str, Any]) -> Path:
        if mode not in MODE_NAMES:
            raise ValueError(f"Invalid mode folder {mode!r}")
        mode_dir = self.run_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_json(mode_dir / "state.json", state)
        write_json(mode_dir / "final_recommendation.json", state.get("final_recommendation", {}))
        write_json(mode_dir / "evaluation.json", state.get("evaluation", {}))
        write_json(mode_dir / "metrics.json", state.get("metrics", {}))
        if mode in {"supervisor", "self_organizing"}:
            write_json(mode_dir / "blackboard.json", state.get("blackboard", []))
        if mode == "self_organizing":
            write_json(mode_dir / "scorecards.json", state.get("scorecards", []))
            write_json(mode_dir / "salience_map.json", state.get("salience_map", {}))
            write_json(mode_dir / "conflict_map.json", state.get("conflict_map", []))
            write_json(mode_dir / "convergence.json", state.get("convergence", {}))
        return mode_dir

