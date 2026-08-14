# PINN Crystal Growth

Clean public repository for a thesis project on adaptive Physics-Informed Neural Networks (PINNs) and corrected Czochralski crystal-growth CFD data.

The codebase is intentionally focused on three things:

1. PINN training for benchmark and crystal-growth-oriented PDE problems.
2. Corrected Czochralski CFD sweep data usage.
3. Reproducible result generation for validation and thesis review.

## Research Aim

The thesis investigates whether adaptive and agentic loss control can improve PINN training behavior for CFD-oriented silicon single-crystal growth simulation.

Core research question:

> Can adaptive loss control improve PDE residual minimization and training stability compared with baseline PINN and gradient-normalized PINN training?

## Current Scope

This repository contains a cleaned implementation focused on:

- 1D heat equation as a controlled PINN benchmark
- simplified 2D thermal-fluid PDE proxy for crystal-growth-oriented experiments
- MLP and SIREN neural architectures
- Baseline PINN, GNPINN, and AgenticPINN training strategies
- supervised CFD surrogate training on corrected Czochralski temperature, crucible-rotation, and crystal-rotation sweep datasets
- result tables, metrics, and plots for reproducible validation

The full local thesis workspace contains draft writing, larger outputs, and third-party exploration folders. Those are intentionally excluded from this clean public repository.

## Repository Structure

```text
.
├── Data/                  # Small synthetic demo dataset
├── equations/             # PDE formulations and residual definitions
├── experiments/           # CFD surrogate training script
├── examples/              # Small example runs
├── models/                # Neural architectures and PINN trainers
├── utils/                 # Training, seeding, and model IO helpers
├── visualization/         # Plotting utilities
├── main.py                # Main CLI experiment entry point
├── requirements.txt       # Python dependencies
├── DATASETS.md            # External CFD data placement and meaning
├── RESULTS_SUMMARY.md     # Final reported result summary
├── RESEARCH_OVERVIEW.md   # Short academic project overview
├── REPRODUCIBILITY.md     # Exact commands for validation
└── NOTICE.md              # Publication and attribution notice
```

## Installation

Python 3.8+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Reproducibility Checks

Run a short heat-equation baseline PINN:

```bash
python main.py --equation heat --epochs 20 --collocation_points 200 --output_dir results/heat_baseline_smoke
```

Run the same benchmark with Gradient Normalization:

```bash
python main.py --equation heat --use_gn --epochs 20 --collocation_points 200 --output_dir results/heat_gn_smoke
```

Run the AgenticPINN controller:

```bash
python main.py --equation heat --use_agentic --epochs 20 --collocation_points 200 --output_dir results/heat_agentic_smoke
```

Run a small crystal-growth-oriented thermal-fluid smoke test:

```bash
python main.py --equation crystal --epochs 5 --collocation_points 200 --plot_resolution 20 --output_dir results/crystal_smoke
```

Train supervised CFD surrogates on corrected external Czochralski CFD data:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --holdout 1780.0 --epochs 120
python experiments/train_cfd_surrogate.py --dataset crucible --holdout -4.0 --epochs 120
python experiments/train_cfd_surrogate.py --dataset crystal --holdout 7.0 --epochs 120
```

These commands expect the external CFD repositories to be available under:

```text
external_repos/CZ_Study_TempChange/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

The surrogate maps `(r, z, case_parameter)` to `(u_r, u_z, u_swirl, p, T)` and writes metrics and figures to `results/cfd_surrogate/`.

Generated outputs are written under `results/`, which is ignored by Git.

For more detail, see:

- `DATASETS.md` for corrected CFD data placement.
- `REPRODUCIBILITY.md` for exact run commands.
- `RESULTS_SUMMARY.md` for final thesis-reported outcomes.

## Main CLI Options

```bash
python main.py --help
```

Important options:

- `--equation`: `heat`, `kdv`, `crystal`, `thermal_coupling`
- `--network_type`: `mlp` or `siren`
- `--use_gn`: use Gradient-Normalized PINN
- `--use_agentic`: use rule-based AgenticPINN adaptive control
- `--epochs`: number of training epochs
- `--collocation_points`: number of PDE collocation points
- `--seed`: random seed
- `--device`: `cpu`, `cuda`, or specific CUDA device

## Method Summary

The training objective combines physics residual loss with boundary and initial condition constraints. The trainer variants differ in how they handle this multi-objective optimization:

- `PINN`: standard weighted sum of PDE and BC/IC losses
- `GNPINN`: gradient-normalized training to reduce imbalance between objectives
- `AgenticPINN`: rule-based adaptive controller that monitors loss and gradient signals, then adjusts weights and learning rate behavior

The crystal-growth case uses a simplified 2D thermal-fluid PDE setting intended as a thesis-scale proxy for more expensive CFD workflows.

## Data

The included synthetic dataset is:

```text
Data/cz_synthetic_data.csv
```

It is provided for reproducible local experiments and demonstration. For thesis claims, synthetic-data limitations should be stated clearly.

The CFD surrogate script uses corrected external datasets shared for thesis development. These are not bundled in this clean repository because they are large external research data. Clone or place the datasets under `external_repos/` before running `experiments/train_cfd_surrogate.py`.

Required external data folders:

```text
external_repos/CZ_Study_TempChange/Steady/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

## Publication Note

Before public release or formal submission, review `NOTICE.md` and confirm that all included files, license metadata, and attribution are appropriate for your institution and supervisor requirements.

## License

This repository currently includes the MIT License file present in the cleaned project. See `LICENSE` and `NOTICE.md`.
