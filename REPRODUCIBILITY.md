# Reproducibility Guide

This guide lists the main commands for checking and reproducing the thesis-relevant code path.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. PINN Smoke Tests

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

These smoke tests are intentionally short. They are useful for checking that the code runs, not for reproducing final thesis-quality metrics.

## 2. Corrected CFD Surrogate Runs

Place or clone the corrected external CFD repositories at:

```text
external_repos/CZ_Study_TempChange/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

Then run the three case-wise holdout surrogate experiments.

Temperature-change dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --holdout 1780.0 --epochs 120 --max_rows_per_case 1800 --hidden_dim 96 --hidden_layers 3
```

Crucible-rotation dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset crucible --holdout -4.0 --epochs 120 --max_rows_per_case 1800 --hidden_dim 96 --hidden_layers 3
```

Crystal-rotation dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset crystal --holdout 7.0 --epochs 120 --max_rows_per_case 1800 --hidden_dim 96 --hidden_layers 3
```

## 3. Expected CFD Surrogate Outputs

Each surrogate run writes outputs to:

```text
results/cfd_surrogate/<dataset>/
```

Expected files include:

- `<dataset>_history.csv`
- `<dataset>_summary.json`
- `<dataset>_metrics.csv`
- `<dataset>_training_loss.png`
- `<dataset>_holdout_parity.png`
- `<dataset>_holdout_T_field.png`
- `<dataset>_holdout_u_swirl_field.png`

## 4. Interpreting the Runs

The CFD surrogate experiments use one full unseen case for testing:

| Dataset | Holdout |
| --- | --- |
| Temperature-change | `1780 K` |
| Crucible-rotation | `-4 rpm` |
| Crystal-rotation | `7 rpm` |

Temperature and crucible sweeps are expected to be easier because their field changes are smoother. The crystal-rotation holdout is expected to be harder, especially for radial and axial velocity, because the corrected data show stronger flow-structure change around the 6 to 7 rpm region.

The `results/` directory is intentionally ignored by Git so generated artifacts do not make the public repository heavy.
