#!/usr/bin/env python3
"""Utilities for P1 Experiment 2.1.

This helper script provides:
1) generation of the Experiment 1A Cartesian configuration grid
2) random sampling for Experiment 1B transfer-evaluation measurement sets
3) Pareto frontier extraction and 2D hypervolume computation

It is lightweight by design and does not execute training/inference itself.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

KAPPA_VALUES = [0.0, 0.3, 0.5, 0.7]
PRECISION_VALUES = ["FP16", "INT8", "INT4"]
ROUTING_VALUES = ["dense", "MoE-2", "MoE-4"]


@dataclass(frozen=True)
class Config:
    hardware: str
    model: str
    dataset: str
    kappa: float
    precision: str
    routing: str


def build_grid(
    hardware: Sequence[str], models: Sequence[str], datasets: Sequence[str]
) -> List[Config]:
    rows: List[Config] = []
    for hw in hardware:
        for model in models:
            for dataset in datasets:
                for kappa in KAPPA_VALUES:
                    for precision in PRECISION_VALUES:
                        for routing in ROUTING_VALUES:
                            rows.append(
                                Config(
                                    hardware=hw,
                                    model=model,
                                    dataset=dataset,
                                    kappa=kappa,
                                    precision=precision,
                                    routing=routing,
                                )
                            )
    return rows


def sample_rows(rows: Sequence[Config], n: int, seed: int) -> List[Config]:
    if n > len(rows):
        raise ValueError(f"cannot sample {n} rows from {len(rows)} candidates")
    rng = random.Random(seed)
    return rng.sample(list(rows), n)


def pareto_front(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return nondominated points for minimization on both axes.

    Input points are tuples: (energy_per_token_j, error = 1 - metric).
    """
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    front: List[Tuple[float, float]] = []
    best_error = float("inf")
    for energy, error in sorted_points:
        if error < best_error:
            front.append((energy, error))
            best_error = error
    return front


def hypervolume_2d(
    front: Sequence[Tuple[float, float]], reference: Tuple[float, float]
) -> float:
    """Compute 2D hypervolume for a minimization front.

    Assumes front is nondominated and ordered by increasing energy.
    """
    if not front:
        return 0.0

    ref_e, ref_err = reference
    hv = 0.0
    prev_e = ref_e

    for energy, error in sorted(front, key=lambda p: p[0], reverse=True):
        width = max(prev_e - energy, 0.0)
        height = max(ref_err - error, 0.0)
        hv += width * height
        prev_e = energy

    return hv


def write_csv(path: Path, rows: Sequence[Config]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hardware", "model", "dataset", "kappa", "precision", "routing"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 Experiment 2.1 helpers")
    parser.add_argument(
        "--mode",
        choices=["grid", "sample"],
        required=True,
        help="grid: full factorial grid, sample: random sample from full grid",
    )
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=["A800", "RTX4090"],
        help="hardware list",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LLaMA-2-7B", "Bloom-7B", "Pythia-6.9B"],
        help="model list",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SQuAD", "IMDb"],
        help="dataset list",
    )
    args = parser.parse_args()

    rows = build_grid(args.hardware, args.models, args.datasets)
    if args.mode == "sample":
        rows = sample_rows(rows, args.sample_size, args.seed)
    write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
