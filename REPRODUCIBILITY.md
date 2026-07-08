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

## Expected Outputs

Each run should create:

- `*_history.json`
- `*_summary.json`
- loss plots
- solution or field plots when plotting succeeds

The `results/` directory is intentionally ignored by Git.
