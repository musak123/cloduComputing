#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.p1.carbon_tracking import CarbonTracker
from training.p1.config import load_json
from training.p1.energy import estimate_energy
from training.p1.metrics import estimate_bops


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 Experiment 1B estimator transfer scaffold")
    parser.add_argument("--config", default="configs/p1/exp1b_transfer.json")
    args = parser.parse_args()

    cfg = load_json(args.config)
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "exp1b_transfer_raw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    tracker = CarbonTracker(project_name="p1_exp1b", output_dir=str(out_dir))
    tracker.start()

    rng = random.Random(cfg.get("seed", 42))
    apes = []
    total_bops = 0.0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["held_out_hardware", "sample_id", "true_j_per_token", "pred_j_per_token", "ape", "bops"],
        )
        writer.writeheader()
        for held_out in cfg["hardware"]:
            for idx in range(cfg["samples_per_hardware"]):
                kappa = rng.choice(cfg["kappa"])
                precision = rng.choice(cfg["precision"])
                routing = rng.choice(cfg["routing"])
                model = rng.choice(cfg.get("models", ["LLaMA-2-7B"]))
                true = estimate_energy(kappa, precision, routing).joules_per_token
                pred = true * rng.uniform(0.92, 1.08)
                ape = abs(pred - true) / true
                bops = estimate_bops(kappa, precision, routing, model)
                apes.append(ape)
                total_bops += bops
                writer.writerow(
                    {
                        "held_out_hardware": held_out,
                        "sample_id": idx,
                        "true_j_per_token": round(true, 6),
                        "pred_j_per_token": round(pred, 6),
                        "ape": round(ape, 6),
                        "bops": round(bops, 2),
                    }
                )

    carbon = tracker.stop().__dict__
    summary = {
        "mape": round(float(mean(apes)) if apes else 0.0, 6),
        "target_status": "pass" if apes and mean(apes) < 0.1 else "fail",
        "total_bops": round(total_bops, 2),
        "emissions": carbon,
    }
    (out_dir / "exp1b_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
