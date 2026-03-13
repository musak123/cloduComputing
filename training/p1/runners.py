from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Dict, List

from .config import ExperimentConfig
from .energy import estimate_energy
from .search import bayes_like_sample, cartesian_space, pareto


_PRECISION_ACC_DELTA = {"FP16": 0.0, "INT8": -0.008, "INT4": -0.021}
_ROUTING_ACC_DELTA = {"dense": 0.0, "MoE-2": 0.006, "MoE-4": 0.012}


def _simulate_accuracy(kappa: float, precision: str, routing: str) -> float:
    acc = 0.88 - (0.04 * kappa) + _PRECISION_ACC_DELTA[precision] + _ROUTING_ACC_DELTA[routing]
    return max(0.0, min(1.0, acc))


def run_sequential(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    run_id = str(uuid.uuid4())
    for seed in cfg.seeds:
        for hardware in cfg.hardware:
            for model in cfg.models:
                for dataset in cfg.datasets:
                    for kappa in cfg.kappa:
                        for precision in cfg.precision:
                            for routing in cfg.routing:
                                acc = _simulate_accuracy(kappa, precision, routing)
                                f1 = acc - 0.01
                                energy = estimate_energy(kappa, precision, routing).joules_per_token
                                rows.append(
                                    {
                                        "run_id": run_id,
                                        "experiment_id": cfg.experiment_id,
                                        "seed": seed,
                                        "hardware": hardware,
                                        "model": model,
                                        "dataset": dataset,
                                        "method": "sequential",
                                        "kappa": kappa,
                                        "precision": precision,
                                        "routing": routing,
                                        "accuracy": round(acc, 6),
                                        "f1": round(f1, 6),
                                        "energy_per_token_j": round(energy, 6),
                                        "latency_ms_per_token": round(4.2 * energy, 6),
                                        "throughput_tok_s": round(1000 / (4.2 * energy), 6),
                                    }
                                )
    return rows


def run_joint(cfg: ExperimentConfig, budget: int = 120) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    run_id = str(uuid.uuid4())
    space = cartesian_space(cfg.kappa, cfg.precision, cfg.routing)
    for seed in cfg.seeds:
        sample = bayes_like_sample(space, budget, seed)
        for hardware in cfg.hardware:
            for model in cfg.models:
                for dataset in cfg.datasets:
                    for kappa, precision, routing in sample:
                        acc = _simulate_accuracy(kappa, precision, routing) + 0.005
                        f1 = acc - 0.008
                        energy = estimate_energy(kappa, precision, routing).joules_per_token * 0.92
                        rows.append(
                            {
                                "run_id": run_id,
                                "experiment_id": cfg.experiment_id,
                                "seed": seed,
                                "hardware": hardware,
                                "model": model,
                                "dataset": dataset,
                                "method": "joint",
                                "kappa": kappa,
                                "precision": precision,
                                "routing": routing,
                                "accuracy": round(acc, 6),
                                "f1": round(f1, 6),
                                "energy_per_token_j": round(energy, 6),
                                "latency_ms_per_token": round(4.2 * energy, 6),
                                "throughput_tok_s": round(1000 / (4.2 * energy), 6),
                            }
                        )
    return rows


def summarize_pareto(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric = [
        {
            "energy_per_token_j": float(r["energy_per_token_j"]),
            "accuracy": float(r["accuracy"]),
        }
        for r in rows
    ]
    return pareto(numeric)
