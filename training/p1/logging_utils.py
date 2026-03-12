from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


FIELDNAMES = [
    "run_id",
    "experiment_id",
    "seed",
    "hardware",
    "model",
    "dataset",
    "method",
    "kappa",
    "precision",
    "routing",
    "accuracy",
    "f1",
    "energy_per_token_j",
    "latency_ms_per_token",
    "throughput_tok_s",
    "bops",
]


def write_rows(path: str | Path, rows: Iterable[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
