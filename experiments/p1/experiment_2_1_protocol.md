# 2.1 Experiments for P1: Proving Energy Can Be a Primary Objective

This document operationalizes the proposed P1 experiments into executable steps, logging schemas, and analysis procedures.

## 1) Hypothesis

- **H1 (Experiment 1A):** Jointly optimizing sparsity (`κ`), precision (`ρ`), and routing (`ξ`) yields a Pareto frontier that dominates sequential optimization in energy-accuracy space.
- **H2 (Experiment 1B):** The hardware-agnostic estimator can generalize to unseen hardware with **MAPE < 10%**.

---

## 2) Experiment 1A — Sequential vs. Joint Optimization

### 2.1 Objective

Demonstrate that co-optimizing (`κ`, `ρ`, `ξ`) simultaneously provides better energy-accuracy trade-offs than sequential optimization.

### 2.2 Setup

- **Hardware:** NVIDIA A800, RTX 4090
- **Models:** LLaMA-2-7B, Bloom-7B, Pythia-6.9B
- **Tasks/Datasets:**
  - SQuAD (classification framing as needed by your pipeline)
  - IMDb sentiment classification
- **Search space:**
  - `κ ∈ {0, 0.3, 0.5, 0.7}`
  - `ρ ∈ {FP16, INT8, INT4}`
  - `ξ ∈ {dense, MoE-2, MoE-4}`

### 2.3 Methods

#### A) Baseline: Sequential optimization (GETA/HAPE-style)

For each `(hardware, model, dataset)`:
1. Start from dense FP16 baseline.
2. Apply sparsification to each `κ` target.
3. Quantize each pruned variant to each `ρ`.
4. Apply routing option `ξ`.
5. Measure after each stage and final configuration:
   - Energy/token (J)
   - Accuracy/F1
   - Latency/token (optional but recommended)

#### B) ALCAF: Joint optimization

For each `(hardware, model, dataset)`:
1. Use Bayesian optimization over the full tuple (`κ`, `ρ`, `ξ`).
2. Objective:
   - Minimize `L_total = λ_E * E_token + λ_A * (1 - metric)`
   - or perform direct multi-objective optimization and extract Pareto set.
3. Budget recommendation:
   - 40 warm-up random points
   - 120 BO iterations
4. Save every evaluated point and final Pareto-optimal set.

### 2.4 Repetition and randomization

- Repeat each condition **10 runs** (different seeds).
- Fix and log seeds for model init, data order, and optimizer randomness.
- Randomize run order across hardware to reduce temporal bias (thermal/load drift).

### 2.5 Metrics

- Primary:
  - Energy per token (J/token)
  - Accuracy or macro-F1
  - Pareto frontier hypervolume
- Secondary:
  - Throughput (tokens/s)
  - p95 latency/token
  - Peak memory

### 2.6 Statistical analysis

- For each `(hardware, model, dataset)`, compare sequential vs joint on:
  - Hypervolume (mean ± std over 10 runs)
  - Energy at matched-accuracy slices (e.g., 80/85/90%)
- Use:
  - Paired t-test if normality holds; otherwise Wilcoxon signed-rank.
  - Report effect size (Cohen’s d or Cliff’s delta).
- Significance: `α = 0.05` with Holm correction across multiple comparisons.

### 2.7 Acceptance criteria

- Joint optimization should dominate the sequential Pareto frontier in most settings.
- Expected gain: **10–20% lower energy** at matched accuracy.

---

## 3) Experiment 1B — Hardware-Agnostic Estimator Accuracy

### 3.1 Objective

Validate estimator transfer to unseen hardware with MAPE < 10%.

### 3.2 Setup

- **Hardware:** A800, RTX 4090, Jetson Orin, Pixel phone GPU
- **Models:** subset from 1A (recommend at least 2 models covering size/architecture diversity)
- For each hardware, sample **50 random configurations** from (`κ`, `ρ`, `ξ`) × model × dataset.
- Measure true energy via:
  - EnergyMeter (server/workstation/edge)
  - Physical meter or platform power API for mobile

### 3.3 Procedure

For each leave-one-hardware-out split:
1. Train estimator on 3 hardware domains.
2. Evaluate on held-out hardware.
3. Compute:
   - MAPE
   - MAE (J/token)
   - R² (optional)

### 3.4 Acceptance criteria

- **MAPE < 10%** on each held-out hardware target.
- No hardware with catastrophic drift (>15%) without documented root cause.

---

## 4) Logging schema (minimum)

Each run record should include:

- Metadata:
  - `timestamp`, `run_id`, `seed`, `experiment_id`
  - `hardware`, `driver`, `cuda_version`
  - `model`, `dataset`
- Configuration:
  - `kappa`, `precision`, `routing`
  - method label: `sequential` or `joint`
  - optimizer state if joint (`iteration`, `acq_value`)
- Outputs:
  - `energy_per_token_j`
  - `accuracy`, `f1`
  - `latency_ms_per_token`
  - `throughput_tok_s`

Store as CSV/Parquet plus a versioned config snapshot (YAML/JSON).

---

## 5) Reproducibility checklist

- Fixed seeds and deterministic flags where possible.
- Pin software stack (CUDA/cuDNN/PyTorch versions).
- Record thermal/power mode for each device.
- Warm-up before each measurement window.
- Use a fixed evaluation subset when comparing config candidates.

---

## 6) Suggested artifacts

- `results/exp1a_sequential_raw.csv`
- `results/exp1a_joint_raw.csv`
- `results/exp1a_pareto_summary.csv`
- `results/exp1b_transfer_raw.csv`
- `figures/exp1a_pareto_{hardware}_{model}_{dataset}.png`
- `figures/exp1b_mape_bar.png`

