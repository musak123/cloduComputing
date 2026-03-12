#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.p1.geta_adapter import run_geta_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GETA baseline harness for P1 comparisons")
    parser.add_argument("--geta-repo", default="/tmp/geta", help="Path to cloned GETA repository")
    parser.add_argument("--output-dir", default="results/p1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-group-sparsity", type=float, default=0.5)
    parser.add_argument("--min-bit-wt", type=int, default=4)
    parser.add_argument("--max-bit-wt", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = run_geta_baseline(
        geta_repo=args.geta_repo,
        output_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed,
        target_group_sparsity=args.target_group_sparsity,
        min_bit_wt=args.min_bit_wt,
        max_bit_wt=args.max_bit_wt,
        dry_run=args.dry_run,
    )
    print(json.dumps({"best_accuracy": report["best_accuracy"], "relative_bops": report["relative_bops"]}, indent=2))


if __name__ == "__main__":
    main()
