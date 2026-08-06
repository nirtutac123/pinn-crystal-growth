# Reproducibility Guide

This guide lists minimal commands for checking that the cleaned repository runs.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Tests

Baseline heat PINN:

```bash
python main.py --equation heat --epochs 20 --collocation_points 200 --output_dir results/heat_baseline_smoke
```

Gradient-normalized heat PINN:

```bash
python main.py --equation heat --use_gn --epochs 20 --collocation_points 200 --output_dir results/heat_gn_smoke
```

Agentic heat PINN:

```bash
python main.py --equation heat --use_agentic --epochs 20 --collocation_points 200 --output_dir results/heat_agentic_smoke
```

Crystal-growth proxy:

```bash
python main.py --equation crystal --epochs 5 --collocation_points 200 --plot_resolution 20 --output_dir results/crystal_smoke
```

Supervised CFD surrogate using the external temperature-change dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --holdout 1780.0 --epochs 120 --max_rows_per_case 1800 --hidden_dim 96 --hidden_layers 3
```

Supervised CFD surrogate using the external swirl-change dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset swirl --holdout 1.676 --epochs 120 --max_rows_per_case 1800 --hidden_dim 96 --hidden_layers 3
```

The CFD surrogate commands require the external datasets to be available at:

```text
external_repos/CZ_Study_TempChange/
external_repos/CZ_study_Swirl-Change/
```

## Expected Outputs

Each run should create:

- `*_history.json`
- `*_summary.json`
- loss plots
- solution or field plots when plotting succeeds
- CFD surrogate metrics, parity plots, and field-error plots when external CFD data is available

The `results/` directory is intentionally ignored by Git.
