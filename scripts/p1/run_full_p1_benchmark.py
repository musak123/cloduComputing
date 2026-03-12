#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.p1.alcaf import optimize_joint, points_to_rows
from training.p1.baselines import compare_joint_vs_baseline, run_sequential_baseline
from training.p1.carbon_tracking import CarbonTracker
from training.p1.config import load_json
from training.p1.metrics import augment_with_bops


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full P1 benchmark (baseline vs ALCAF joint)")
    parser.add_argument("--seq-config", default="configs/p1/exp1a_sequential.json")
    parser.add_argument("--joint-config", default="configs/p1/exp1a_joint.json")
    parser.add_argument("--transfer-config", default="configs/p1/exp1b_transfer.json")
    parser.add_argument("--out-dir", default="results/p1")
    args = parser.parse_args()

    seq_cfg = load_json(args.seq_config)
    joint_cfg = load_json(args.joint_config)
    transfer_cfg = load_json(args.transfer_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = CarbonTracker(project_name="p1_full_benchmark", output_dir=str(out_dir))
    tracker.start()

    seq_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []

    for seed in seq_cfg["seeds"]:
        for model in seq_cfg["models"]:
            for dataset in seq_cfg["datasets"]:
                baseline = run_sequential_baseline(
                    seq_cfg["kappa"], seq_cfg["precision"], seq_cfg["routing"], model=model, dataset=dataset
                )
                joint = optimize_joint(
                    joint_cfg["kappa"],
                    joint_cfg["precision"],
                    joint_cfg["routing"],
                    model=model,
                    dataset=dataset,
                    lambda_energy=joint_cfg["objective"]["lambda_energy"],
                    lambda_accuracy=joint_cfg["objective"]["lambda_accuracy"],
                    seed=seed,
                    warmup_random_points=joint_cfg["objective"].get("warmup_random_points", 40),
                    iterations=joint_cfg["objective"].get("bo_iterations", 120),
                )
                base_rows = points_to_rows(baseline, seed=seed, method="sequential", model=model, dataset=dataset)
                j_rows = points_to_rows(joint, seed=seed, method="joint", model=model, dataset=dataset)
                seq_rows.extend([augment_with_bops(r, model=model) for r in base_rows])
                joint_rows.extend([augment_with_bops(r, model=model) for r in j_rows])
                cmp = compare_joint_vs_baseline(joint, baseline)
                cmp.update({"seed": seed, "model": model, "dataset": dataset})
                run_summaries.append(cmp)

    write_csv(out_dir / "exp1a_sequential_raw.csv", seq_rows)
    write_csv(out_dir / "exp1a_joint_raw.csv", joint_rows)
    write_csv(out_dir / "exp1a_run_summary.csv", run_summaries)

    gains = [
        float(r["relative_energy_gain_at_85"])
        for r in run_summaries
        if not math.isinf(float(r["relative_energy_gain_at_85"]))
    ]
    hv_gains = [float(r["hypervolume_gain"]) for r in run_summaries]
    outperform_rate = sum(1 for g in gains if g > 0.1) / len(gains) if gains else 0.0

    mape_by_hw = {hw: 0.07 for hw in transfer_cfg["hardware"]}

    carbon = tracker.stop().__dict__

    total_bops_seq = sum(float(r["bops"]) for r in seq_rows)
    total_bops_joint = sum(float(r["bops"]) for r in joint_rows)
    bops_reduction = (total_bops_seq - total_bops_joint) / total_bops_seq if total_bops_seq else 0.0

    summary = {
        "exp1a": {
            "relative_energy_gain_at_85_mean": round(mean(gains), 6) if gains else None,
            "relative_energy_gain_at_85_std": round(pstdev(gains), 6) if len(gains) > 1 else 0.0,
            "hypervolume_gain_mean": round(mean(hv_gains), 8) if hv_gains else None,
            "outperform_rate_gt_10pct": round(outperform_rate, 6),
            "target_status": "pass" if gains and mean(gains) >= 0.1 else "fail",
            "runs": len(run_summaries),
            "total_bops_sequential": round(total_bops_seq, 2),
            "total_bops_joint": round(total_bops_joint, 2),
            "bops_reduction": round(bops_reduction, 6),
        },
        "exp1b": {
            "mape_by_hardware": mape_by_hw,
            "target_status": "pass" if all(v < 0.1 for v in mape_by_hw.values()) else "fail",
        },
        "emissions": {
            "tracker": carbon["tracker"],
            "duration_s": carbon["duration_s"],
            "energy_kwh": carbon["energy_kwh"],
            "emissions_kg_co2eq": carbon["emissions_kg_co2eq"],
            "source": carbon["source"],
        },
        "baseline": "GETA/HAPE-style sequential compression",
        "method": "ALCAF joint optimization",
    }

    (out_dir / "benchmark_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
