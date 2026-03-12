#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


INCLUDE = [
    "configs/p1",
    "scripts/p1",
    "training/p1",
    "deployment/p1",
    "experiments/p1/experiment_2_1_protocol.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package P1 experiment training/deployment bundle")
    parser.add_argument("--output", default="results/p1/p1_training_deployment_bundle.tar.gz")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        for path in INCLUDE:
            p = Path(path)
            if p.exists():
                tf.add(p)


if __name__ == "__main__":
    main()
