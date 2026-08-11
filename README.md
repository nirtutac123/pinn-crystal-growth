# Physics-Informed Neural Networks for Crystal Growth CFD

Clean public repository for a thesis project on physics-based neural solvers for PDE-driven thermal-fluid and crystal-growth simulations.

The project compares standard Physics-Informed Neural Networks (PINNs), Gradient-Normalized PINNs (GNPINNs), and a rule-based AgenticPINN controller for balancing PDE residual and boundary/initial-condition losses.

## Research Aim

The thesis investigates whether adaptive physics-informed training can improve the efficiency and stability of neural PDE solvers for silicon single-crystal growth simulations.

Core research question:

> Can adaptive loss control improve PINN training behavior for crystal-growth thermal-fluid PDEs compared with baseline PINN and gradient-normalized PINN training?

## Current Scope

This repository contains a cleaned implementation focused on:

- 1D heat equation as a controlled benchmark
- KdV equation as an additional nonlinear PDE benchmark
- 2D Navier-Stokes-style thermal-fluid coupling for crystal-growth simulation
- MLP and SIREN neural architectures
- Baseline PINN, GNPINN, and AgenticPINN training strategies
- Supervised CFD surrogate training on externally shared Czochralski temperature, crucible-rotation, and crystal-rotation sweep datasets
- Plotting and artifact export for reproducible experiments

The full thesis workspace may contain additional notes, draft documents, large generated outputs, and third-party model explorations. Those are intentionally excluded from this public repository.

## Repository Structure

```text
.
├── Data/                  # Synthetic Czochralski-style dataset and generator
├── equations/             # PDE formulations and residual definitions
├── experiments/           # Reproducible experiment scripts
├── examples/              # Example scripts and visualization workflows
├── models/                # Neural architectures and PINN trainers
├── utils/                 # Training, seeding, and model IO helpers
├── visualization/         # Plotting utilities
├── main.py                # Main CLI experiment entry point
├── requirements.txt       # Python dependencies
├── RESEARCH_OVERVIEW.md   # Short academic project overview
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

Run a short heat-equation baseline:

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

Run a small crystal-growth thermal-fluid smoke test:

```bash
python main.py --equation crystal --epochs 5 --collocation_points 200 --plot_resolution 20 --output_dir results/crystal_smoke
```

Train supervised CFD surrogates on external Czochralski CFD data:

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

The CFD surrogate script uses external datasets shared for thesis development. These are not bundled in this clean repository. Clone or place the datasets under `external_repos/` before running `experiments/train_cfd_surrogate.py`.

## Publication Note

Before public release or formal submission, review `NOTICE.md` and confirm that all included files, license metadata, and attribution are appropriate for your institution and supervisor requirements.

## License

This repository currently includes the MIT License file present in the cleaned project. See `LICENSE` and `NOTICE.md`.
