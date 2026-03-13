# P1 Experiment Training Structure (Deployment Ready)

## Directory layout

- `configs/p1/`
  - `exp1a_sequential.json`
  - `exp1a_joint.json`
  - `exp1b_transfer.json`
  - `deployment.json`
- `training/p1/`
  - `config.py` (typed config loading)
  - `energy.py` (energy estimator abstraction)
  - `search.py` (search space and Pareto utils)
  - `runners.py` (sequential/joint experiment runners)
  - `logging_utils.py` (CSV artifact writer)
  - `tracking.py` (CodeCarbon/Eco2AI tracking + kWh/kgCO2eq/BOP helpers)
- `scripts/p1/`
  - `run_full_p1_benchmark.py` (runs baseline vs ALCAF and writes report)
  - `run_geta_baseline.py` (GETA baseline harness)
  - `train_exp1a.py` (legacy split runner)
  - `train_exp1b.py` (legacy transfer runner)
  - `build_deployment_bundle.py` (tarball for deployment handoff)
- `deployment/p1/`
  - `docker/Dockerfile`
  - `k8s/job-exp1a.yaml`
  - `k8s/job-exp1b.yaml`

## Output artifacts

- `results/p1/exp1a_sequential_raw.csv`
- `results/p1/exp1a_joint_raw.csv`
- `results/p1/exp1a_summary.json`
- `results/p1/exp1b_transfer_raw.csv`
- `results/p1/p1_training_deployment_bundle.tar.gz`
- `results/p1/emissions_tracking.csv`
- `results/p1/benchmark_report.json`
- `results/p1/geta_baseline_report.json`

- `tests_p1_smoke.py` (sanity check for ALCAF outperforming baseline)
