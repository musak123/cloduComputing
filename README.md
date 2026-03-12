# cloduComputing

Energy-aware experiment scaffolding for P1 (Task 2.1), including training runners, configs, and deployment packaging.

## P1 full training experiment structure

- Protocol: `experiments/p1/experiment_2_1_protocol.md`
- Training structure doc: `experiments/p1/TRAINING_STRUCTURE.md`
- Training package: `training/p1/`
- CLI runners: `scripts/p1/`
- Deployment assets: `deployment/p1/`

## Run experiments

### Experiment 1A (Sequential vs Joint)

```bash
python scripts/p1/train_exp1a.py \
  --sequential-config configs/p1/exp1a_sequential.json \
  --joint-config configs/p1/exp1a_joint.json
```

### Experiment 1B (Hardware-Agnostic Transfer)

```bash
python scripts/p1/train_exp1b.py \
  --config configs/p1/exp1b_transfer.json
```


### Full benchmark (ALCAF vs GETA/HAPE baseline + transfer report)

```bash
python scripts/p1/run_full_p1_benchmark.py \
  --seq-config configs/p1/exp1a_sequential.json \
  --joint-config configs/p1/exp1a_joint.json \
  --transfer-config configs/p1/exp1b_transfer.json \
  --out-dir results/p1
```

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
