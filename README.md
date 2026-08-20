# PINN Crystal Growth

Clean public repository for a thesis project on adaptive Physics-Informed Neural Networks (PINNs) and corrected Czochralski crystal-growth CFD data.

The codebase is intentionally focused on three things:

1. PINN training for benchmark and crystal-growth-oriented PDE problems.
2. Corrected Czochralski CFD sweep data usage.
3. Reproducible result generation for validation and thesis review.

## Short Abstract

This project studies adaptive physics-informed neural-network training for CFD-oriented Czochralski silicon single-crystal growth simulation. Baseline PINN, gradient-normalized PINN, and AgenticPINN variants are compared on benchmark and simplified crystal-growth-oriented PDE problems. The central finding is that adaptive control can reduce PDE residuals in selected settings, but it can also shift error toward boundary or initial-condition constraints; therefore, component-level physics metrics are more informative than total loss alone. Corrected Czochralski CFD sweep data are additionally used with Gaussian Process Regression to provide fast surrogate validation and uncertainty maps for temperature, crucible-rotation, and crystal-rotation cases.

## Research Aim

The thesis investigates whether adaptive and agentic loss control can improve PINN training behavior for CFD-oriented silicon single-crystal growth simulation.

Core research question:

> Can adaptive loss control improve PDE residual minimization and training stability compared with baseline PINN and gradient-normalized PINN training?

## Current Scope

This repository contains a cleaned implementation focused on:

- 1D heat equation as a controlled PINN benchmark
- simplified 2D thermal-fluid PDE proxy for crystal-growth-oriented experiments
- MLP and SIREN neural architectures for PINN experiments
- Baseline PINN, GNPINN, and AgenticPINN training strategies
- fast Gaussian Process Regression (GPR) surrogate checks on corrected Czochralski temperature, crucible-rotation, and crystal-rotation sweep datasets
- result tables, metrics, and plots for reproducible validation

The full local thesis workspace contains draft writing, larger outputs, and third-party exploration folders. Those are intentionally excluded from this clean public repository.

## Repository Structure

```text
.
├── Data/                  # Small synthetic demo dataset
├── core/                  # Reusable CFD data loading and GPR surrogate utilities
├── equations/             # PDE formulations and residual definitions
├── experiments/           # Reproducible experiment scripts and local tests
├── examples/              # Small example runs
├── models/                # Neural architectures and PINN trainers
├── utils/                 # Training, seeding, and model IO helpers
├── visualization/         # Plotting utilities
├── results/               # Selected lightweight review results
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

Fit supervised GPR CFD surrogates on corrected external Czochralski CFD data:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --validate_only
python experiments/train_cfd_surrogate.py --dataset temperature --holdout 1780.0
python experiments/train_cfd_surrogate.py --dataset crucible --holdout -4.0
python experiments/train_cfd_surrogate.py --dataset crystal --holdout 7.0
```

These commands expect the external CFD repositories to be available under:

```text
external_repos/CZ_Study_TempChange/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

It also supports the raw `data.zip` layout shared during thesis development after extraction:

```text
data/temperature/
data/crucible/
data/crystal/
```

Use `--data_root /path/to/extracted/data_parent` if the corrected CSV files are stored outside `external_repos/`.

The GPR surrogate maps `(r, z, case_parameter)` to `(u_r, u_z, u_swirl, p, T)` and writes metrics, parity plots, and uncertainty maps to `results/cfd_surrogate_gpr/`.

Selected GPR result summaries and plots are kept under `results/cfd_surrogate_gpr/` so the repository can be inspected quickly. Larger exploratory outputs remain ignored.

For more detail, see:

- `DATASETS.md` for corrected CFD data placement.
- `REPRODUCIBILITY.md` for exact run commands.
- `RESULTS_SUMMARY.md` for final thesis-reported outcomes.
- `ABSTRACT.md` for a polished project abstract.

The corrected CFD datasets are archived on Zenodo for citation:

- Temperature sweep: https://doi.org/10.5281/zenodo.21955315
- Crucible rotation sweep: https://doi.org/10.5281/zenodo.21955323
- Crystal rotation sweep: https://doi.org/10.5281/zenodo.21955299

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

Alternative extracted raw zip layout:

```text
data/temperature/
data/crucible/
data/crystal/
```

Example with an extracted zip folder:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /path/to/extracted --validate_only
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /path/to/extracted --holdout 1780.0
```

## Publication Note

Before public release or formal submission, review `NOTICE.md` and confirm that all included files, license metadata, and attribution are appropriate for your institution and supervisor requirements.

## License

This repository currently includes the MIT License file present in the cleaned project. See `LICENSE` and `NOTICE.md`.
