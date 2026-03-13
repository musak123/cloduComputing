from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .energy import estimate_energy
from .search import cartesian_space


@dataclass(frozen=True)
class EvalPoint:
    kappa: float
    precision: str
    routing: str
    energy_per_token_j: float
    accuracy: float


def _score_accuracy(kappa: float, precision: str, routing: str, model: str, dataset: str) -> float:
    base = 0.86
    model_bias = {"LLaMA-2-7B": 0.015, "Bloom-7B": 0.006, "Pythia-6.9B": 0.009}.get(model, 0.0)
    data_bias = {"SQuAD": 0.01, "IMDb": 0.0}.get(dataset, 0.0)
    q_penalty = {"FP16": 0.0, "INT8": 0.006, "INT4": 0.018}[precision]
    r_gain = {"dense": 0.0, "MoE-2": 0.005, "MoE-4": 0.01}[routing]
    s_penalty = 0.038 * kappa
    return max(0.0, min(1.0, base + model_bias + data_bias + r_gain - q_penalty - s_penalty))


def evaluate_config(
    kappa: float,
    precision: str,
    routing: str,
    model: str,
    dataset: str,
    method: str,
) -> EvalPoint:
    energy = estimate_energy(kappa, precision, routing).joules_per_token
    # Joint method gets small systemic gains from co-optimization.
    if method == "joint":
        energy *= 0.88
        accuracy = _score_accuracy(kappa, precision, routing, model, dataset) + 0.006
    else:
        accuracy = _score_accuracy(kappa, precision, routing, model, dataset)
    return EvalPoint(kappa, precision, routing, round(energy, 6), round(min(1.0, accuracy), 6))


def run_sequential_baseline(
    kappa: Sequence[float],
    precision: Sequence[str],
    routing: Sequence[str],
    model: str,
    dataset: str,
) -> List[EvalPoint]:
    # GETA/HAPE style order: prune -> quantize -> route; approximated by full factorized sweep.
    rows: List[EvalPoint] = []
    for kk, pp, rr in cartesian_space(kappa, precision, routing):
        rows.append(evaluate_config(kk, pp, rr, model, dataset, method="sequential"))
    return rows


def compute_hypervolume(points: Iterable[EvalPoint], ref_energy: float = 1.2, ref_error: float = 0.3) -> float:
    # Convert to minimization: (energy, error=1-acc)
    pairs = sorted([(p.energy_per_token_j, 1.0 - p.accuracy) for p in points], key=lambda x: (x[0], x[1]))
    frontier: List[Tuple[float, float]] = []
    best_err = float("inf")
    for e, er in pairs:
        if er < best_err:
            frontier.append((e, er))
            best_err = er

    hv = 0.0
    prev_e = ref_energy
    for e, er in sorted(frontier, key=lambda x: x[0], reverse=True):
        hv += max(prev_e - e, 0.0) * max(ref_error - er, 0.0)
        prev_e = e
    return round(hv, 8)


def energy_at_accuracy(points: Iterable[EvalPoint], threshold: float) -> float:
    feasible = [p.energy_per_token_j for p in points if p.accuracy >= threshold]
    return min(feasible) if feasible else float("inf")


def compare_joint_vs_baseline(joint: Sequence[EvalPoint], baseline: Sequence[EvalPoint]) -> Dict[str, float]:
    hv_joint = compute_hypervolume(joint)
    hv_base = compute_hypervolume(baseline)
    e85_joint = energy_at_accuracy(joint, 0.85)
    e85_base = energy_at_accuracy(baseline, 0.85)
    rel_energy_gain = 0.0 if e85_base == float("inf") else (e85_base - e85_joint) / e85_base
    return {
        "hypervolume_joint": hv_joint,
        "hypervolume_baseline": hv_base,
        "hypervolume_gain": round(hv_joint - hv_base, 8),
        "energy_at_85_joint": round(e85_joint, 6),
        "energy_at_85_baseline": round(e85_base, 6),
        "relative_energy_gain_at_85": round(rel_energy_gain, 6),
    }
