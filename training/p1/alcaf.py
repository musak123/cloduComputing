from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Sequence

from .baselines import EvalPoint, evaluate_config
from .search import cartesian_space


def optimize_joint(
    kappa: Sequence[float],
    precision: Sequence[str],
    routing: Sequence[str],
    model: str,
    dataset: str,
    lambda_energy: float,
    lambda_accuracy: float,
    seed: int,
    warmup_random_points: int = 40,
    iterations: int = 120,
) -> List[EvalPoint]:
    rng = random.Random(seed)
    space = cartesian_space(kappa, precision, routing)
    sampled = []

    # Warmup random exploration.
    random.shuffle(space)
    for kk, pp, rr in space[: min(warmup_random_points, len(space))]:
        sampled.append(evaluate_config(kk, pp, rr, model, dataset, method="joint"))

    # BO-like exploitation with stochastic exploration.
    for _ in range(iterations):
        candidate_pool = rng.sample(space, k=min(12, len(space)))
        best = None
        best_obj = float("inf")
        for kk, pp, rr in candidate_pool:
            p = evaluate_config(kk, pp, rr, model, dataset, method="joint")
            obj = lambda_energy * p.energy_per_token_j + lambda_accuracy * (1.0 - p.accuracy)
            if obj < best_obj:
                best_obj = obj
                best = p
        if best is not None:
            sampled.append(best)

    return sampled


def points_to_rows(points: Sequence[EvalPoint], **extra: object) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in points:
        row = asdict(p)
        row.update(extra)
        rows.append(row)
    return rows
