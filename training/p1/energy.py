from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyEstimate:
    joules_per_token: float


_PRECISION_FACTOR = {"FP16": 1.0, "INT8": 0.72, "INT4": 0.58}
_ROUTING_FACTOR = {"dense": 1.0, "MoE-2": 0.85, "MoE-4": 0.78}


def estimate_energy(kappa: float, precision: str, routing: str, base: float = 0.95) -> EnergyEstimate:
    sparsity_factor = 1.0 - (0.55 * kappa)
    joules = base * sparsity_factor * _PRECISION_FACTOR[precision] * _ROUTING_FACTOR[routing]
    return EnergyEstimate(joules_per_token=joules)
