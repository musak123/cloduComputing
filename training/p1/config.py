from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    experiment_id: str
    output_dir: str
    hardware: List[str]
    models: List[str]
    datasets: List[str]
    seeds: List[int]
    kappa: List[float]
    precision: List[str]
    routing: List[str]
    objective: Dict[str, Any]



def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = load_json(path)
    return ExperimentConfig(**payload)
