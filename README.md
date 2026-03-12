# cloduComputing

Energy-aware experiment scaffolding for P1 (Task 2.1), including training runners, configs, deployment packaging, and carbon tracking.

## P1 full training experiment structure

- Protocol: `experiments/p1/experiment_2_1_protocol.md`
- Training structure doc: `experiments/p1/TRAINING_STRUCTURE.md`
- Training package: `training/p1/`
- CLI runners: `scripts/p1/`
- Deployment assets: `deployment/p1/`

## Carbon and energy tracking

This repository integrates **CodeCarbon** and **Eco2AI** via a unified wrapper (`training/p1/carbon_tracking.py`).

- Tracker preference: `codecarbon` → `eco2ai` → deterministic fallback estimator.
- Metrics recorded in reports:
  - `energy_kwh`
  - `emissions_kg_co2eq`
  - `duration_s`
  - source label (`measured` / `estimated_from_runtime`)
- Additional efficiency metric:
  - `bops` (per-row estimate) and total BOPs reduction in benchmark report.

If dependencies are available, install:

```bash
pip install codecarbon eco2ai
```

## Run experiments

### Experiment 1A (Sequential vs Joint)

```bash
python scripts/p1/train_exp1a.py \
  --sequential-config configs/p1/exp1a_sequential.json \
  --joint-config configs/p1/exp1a_joint.json
```

Produces `exp1a_sequential_raw.csv`, `exp1a_joint_raw.csv`, and `exp1a_summary.json` (includes BOPs + kWh + kgCO₂eq).

### Experiment 1B (Hardware-Agnostic Transfer)

```bash
python scripts/p1/train_exp1b.py \
  --config configs/p1/exp1b_transfer.json
```

Produces `exp1b_transfer_raw.csv` and `exp1b_summary.json` (includes MAPE + BOPs + kWh + kgCO₂eq).

### Full benchmark (ALCAF vs GETA/HAPE baseline + transfer report)

```bash
python scripts/p1/run_full_p1_benchmark.py \
  --seq-config configs/p1/exp1a_sequential.json \
  --joint-config configs/p1/exp1a_joint.json \
  --transfer-config configs/p1/exp1b_transfer.json \
  --out-dir results/p1
```

Produces `benchmark_report.json` with:
- relative energy gain at matched accuracy,
- hypervolume gains,
- total BOPs and BOPs reduction,
- tracked kWh and kgCO₂eq.

### Build deployment bundle

```bash
python scripts/p1/build_deployment_bundle.py \
  --output results/p1/p1_training_deployment_bundle.tar.gz
```

## Deployment

- Docker build context: `deployment/p1/docker/Dockerfile`
- Kubernetes jobs:
  - `deployment/p1/k8s/job-exp1a.yaml`
  - `deployment/p1/k8s/job-exp1b.yaml`
