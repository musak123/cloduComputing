from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple


def cartesian_space(
    kappa: Sequence[float], precision: Sequence[str], routing: Sequence[str]
) -> List[Tuple[float, str, str]]:
    return [(k, p, r) for k in kappa for p in precision for r in routing]


def bayes_like_sample(space: Sequence[Tuple[float, str, str]], n: int, seed: int) -> List[Tuple[float, str, str]]:
    rng = random.Random(seed)
    n = min(n, len(space))
    return rng.sample(list(space), n)


def pareto(points: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    # minimize energy_per_token_j, maximize accuracy
    ordered = sorted(points, key=lambda x: (x["energy_per_token_j"], -x["accuracy"]))
    frontier: List[Dict[str, float]] = []
    best_accuracy = -1.0
    for p in ordered:
        if p["accuracy"] > best_accuracy:
            frontier.append(p)
            best_accuracy = p["accuracy"]
    return frontier
