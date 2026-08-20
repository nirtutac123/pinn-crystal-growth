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

## 2. Corrected CFD GPR Surrogate Runs

Place or clone the corrected external CFD repositories at:

```text
external_repos/CZ_Study_TempChange/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

Then run the three case-wise holdout surrogate experiments. These use Gaussian Process Regression rather than an MLP so the runs are faster on a laptop and provide predictive uncertainty.

Quick data validation without training:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --validate_only
python experiments/train_cfd_surrogate.py --dataset crucible --validate_only
python experiments/train_cfd_surrogate.py --dataset crystal --validate_only
```

Temperature-change dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --holdout 1780.0 --max_rows_per_case 80
```

Crucible-rotation dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset crucible --holdout -4.0 --max_rows_per_case 80
```

Crystal-rotation dataset:

```bash
python experiments/train_cfd_surrogate.py --dataset crystal --holdout 7.0 --max_rows_per_case 80
```

If using Bertwin's raw `data.zip`, extract it first and pass the parent folder with `--data_root`:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /path/to/extracted --validate_only
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /path/to/extracted --holdout 1780.0 --max_rows_per_case 80
python experiments/train_cfd_surrogate.py --dataset crucible --data_root /path/to/extracted --holdout -4.0 --max_rows_per_case 80
python experiments/train_cfd_surrogate.py --dataset crystal --data_root /path/to/extracted --holdout 7.0 --max_rows_per_case 80
```

## 3. Expected CFD Surrogate Outputs

Each surrogate run writes outputs to:

```text
results/cfd_surrogate_gpr/<dataset>/
```

Expected files include:

- `<dataset>_gpr_summary.json`
- `<dataset>_gpr_metrics.csv`
- `<dataset>_gpr_metrics.md`
- `<dataset>_holdout_parity.png`
- `<dataset>_holdout_T_field_uncertainty.png`
- `<dataset>_holdout_u_swirl_field_uncertainty.png`

## 4. Interpreting the Runs

The CFD surrogate experiments use one full unseen case for testing:

| Dataset | Holdout |
| --- | --- |
| Temperature-change | `1780 K` |
| Crucible-rotation | `-4 rpm` |
| Crystal-rotation | `7 rpm` |

Temperature and crucible sweeps are expected to be easier because their field changes are smoother. The crystal-rotation holdout is expected to be harder, especially for radial and axial velocity, because the corrected data show stronger flow-structure change around the 6 to 7 rpm region.

Selected `results/cfd_surrogate_gpr/` artifacts are kept in Git so collaborators can inspect the outcome without rerunning everything. Larger generated artifacts remain ignored.

## 5. Local Test Command

The CFD surrogate data-loading tests live beside the experiment script:

```bash
python -m unittest discover -s experiments/tests -v
```
