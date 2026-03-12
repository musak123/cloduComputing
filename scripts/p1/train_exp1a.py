#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.p1.carbon_tracking import CarbonTracker
from training.p1.config import load_experiment_config
from training.p1.logging_utils import write_rows
from training.p1.metrics import augment_with_bops
from training.p1.runners import run_joint, run_sequential, summarize_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 Experiment 1A training scaffold")
    parser.add_argument("--sequential-config", default="configs/p1/exp1a_sequential.json")
    parser.add_argument("--joint-config", default="configs/p1/exp1a_joint.json")
    args = parser.parse_args()

    seq_cfg = load_experiment_config(args.sequential_config)
    joint_cfg = load_experiment_config(args.joint_config)

    out_dir = Path(joint_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = CarbonTracker(project_name="p1_exp1a", output_dir=str(out_dir))
    tracker.start()

    seq_rows = [augment_with_bops(r, model=str(r["model"])) for r in run_sequential(seq_cfg)]
    joint_rows = [augment_with_bops(r, model=str(r["model"])) for r in run_joint(joint_cfg, budget=joint_cfg.objective.get("bo_iterations", 120))]

    write_rows(Path(seq_cfg.output_dir) / "exp1a_sequential_raw.csv", seq_rows)
    write_rows(Path(joint_cfg.output_dir) / "exp1a_joint_raw.csv", joint_rows)

    carbon = tracker.stop().__dict__
    summary = {
        "sequential_pareto": summarize_pareto(seq_rows),
        "joint_pareto": summarize_pareto(joint_rows),
        "sequential_rows": len(seq_rows),
        "joint_rows": len(joint_rows),
        "total_bops_sequential": round(sum(float(r["bops"]) for r in seq_rows), 2),
        "total_bops_joint": round(sum(float(r["bops"]) for r in joint_rows), 2),
        "emissions": carbon,
    }
    out_path = Path(joint_cfg.output_dir) / "exp1a_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
