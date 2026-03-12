from __future__ import annotations

from typing import Dict


def estimate_bops(kappa: float, precision: str, routing: str, model: str) -> float:
    """Coarse BOPs estimate for comparative tracking.

    We use model-scale constants and transform factors for sparsity/precision/routing.
    """
    base_bops = {
        "LLaMA-2-7B": 2.8e12,
        "Bloom-7B": 2.9e12,
        "Pythia-6.9B": 2.6e12,
    }.get(model, 2.5e12)
    precision_factor = {"FP16": 1.0, "INT8": 0.55, "INT4": 0.32}[precision]
    routing_factor = {"dense": 1.0, "MoE-2": 0.76, "MoE-4": 0.62}[routing]
    sparsity_factor = 1.0 - (0.68 * kappa)
    return base_bops * precision_factor * routing_factor * sparsity_factor


def augment_with_bops(row: Dict[str, object], model: str) -> Dict[str, object]:
    kappa = float(row["kappa"])
    precision = str(row["precision"])
    routing = str(row["routing"])
    row["bops"] = round(estimate_bops(kappa, precision, routing, model), 2)
    return row
