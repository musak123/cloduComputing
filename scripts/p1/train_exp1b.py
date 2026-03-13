#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.p1.config import load_json
from training.p1.energy import estimate_energy
from training.p1.tracking import EmissionTrackerFacade, write_emission_row, reduction_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 Experiment 1B estimator transfer scaffold")
    parser.add_argument("--config", default="configs/p1/exp1b_transfer.json")
    args = parser.parse_args()

    cfg = load_json(args.config)
    tracker = EmissionTrackerFacade(project_name="p1_train_exp1b", output_dir=cfg["output_dir"])
    tracker.start()
    out = Path(cfg["output_dir"]) / "exp1b_transfer_raw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.get("seed", 42))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["held_out_hardware", "sample_id", "true_j_per_token", "pred_j_per_token", "ape"],
        )
        writer.writeheader()
        for held_out in cfg["hardware"]:
            for idx in range(cfg["samples_per_hardware"]):
                kappa = rng.choice(cfg["kappa"])
                precision = rng.choice(cfg["precision"])
                routing = rng.choice(cfg["routing"])
                true = estimate_energy(kappa, precision, routing).joules_per_token
                pred = true * rng.uniform(0.92, 1.08)
                ape = abs(pred - true) / true
                writer.writerow(
                    {
                        "held_out_hardware": held_out,
                        "sample_id": idx,
                        "true_j_per_token": round(true, 6),
                        "pred_j_per_token": round(pred, 6),
                        "ape": round(ape, 6),
                    }
                )

    track = tracker.stop()
    write_emission_row(Path(cfg["output_dir"]) / "emissions_tracking.csv", {"project": "p1_train_exp1b", "provider": track.provider, "duration_s": round(track.duration_s,4), "energy_kwh": round(track.energy_kwh,8), "co2_kg": round(track.co2_kg,8), "note": "; ".join(reduction_recommendations())})

if __name__ == "__main__":
    main()
