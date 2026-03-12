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
  - `metrics.py` (BOPs estimation)
  - `carbon_tracking.py` (CodeCarbon/Eco2AI integration with fallback)
  - `search.py` (search space and Pareto utils)
  - `runners.py` (sequential/joint experiment runners)
  - `baselines.py` (GETA/HAPE-style baseline analysis)
  - `alcaf.py` (joint ALCAF optimization logic)
  - `logging_utils.py` (CSV artifact writer)
- `scripts/p1/`
  - `run_full_p1_benchmark.py` (runs baseline vs ALCAF, writes benchmark report + emissions)
  - `train_exp1a.py` (split runner with emissions + BOPs)
  - `train_exp1b.py` (transfer runner with emissions + BOPs)
  - `build_deployment_bundle.py` (tarball for deployment handoff)
- `deployment/p1/`
  - `docker/Dockerfile`
  - `k8s/job-exp1a.yaml`
  - `k8s/job-exp1b.yaml`

## Output artifacts

- `results/p1/exp1a_sequential_raw.csv`
- `results/p1/exp1a_joint_raw.csv`
- `results/p1/exp1a_run_summary.csv`
- `results/p1/exp1a_summary.json`
- `results/p1/exp1b_transfer_raw.csv`
- `results/p1/exp1b_summary.json`
- `results/p1/benchmark_report.json`
- `results/p1/p1_training_deployment_bundle.tar.gz`
- `tests_p1_smoke.py` (sanity checks)
