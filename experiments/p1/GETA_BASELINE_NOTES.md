# GETA Baseline Integration Notes

This repository now includes a dedicated GETA baseline harness to align P1 comparisons with the official GETA project.

## Source reviewed

- Repository: `https://github.com/microsoft/geta.git`
- Key usage pattern in GETA README:
  - Quantize model via `model_to_quantize_model(...)`
  - Initialize `OTO(...)`
  - Create optimizer via `oto.geta(...)`
  - Train and export subnet using `oto.construct_subnet(...)`

## Added integration points

- `training/p1/geta_adapter.py`
  - Exposes `run_geta_baseline(...)`
  - Captures GETA-aligned hyperparameter contract
  - Tracks kWh and kg CO2eq via shared tracking facade
  - Writes `results/p1/geta_baseline_report.json`
- `scripts/p1/run_geta_baseline.py`
  - CLI entrypoint to run baseline harness
- `configs/p1/geta_baseline.json`
  - Default baseline parameters

## Outputs

- `results/p1/geta_baseline_report.json`
- `results/p1/emissions_tracking.csv`

## Why this exists

The prior implementation compared against a synthetic sequential baseline only. This harness adds an explicit GETA baseline workflow so comparisons can be migrated from synthetic proxy baselines toward repository-grounded baselines.
