# Results Index

This folder contains selected lightweight result artifacts that are useful for review. Larger exploratory outputs and model checkpoints are still ignored by Git.

## Included Result Set

`results/cfd_surrogate_gpr/` contains Gaussian Process Regression surrogate outputs for corrected Czochralski CFD sweep data.

Each dataset folder includes:

- `*_gpr_metrics.csv`
- `*_gpr_metrics.md`
- `*_gpr_summary.json`
- `*_holdout_parity.png`
- `*_holdout_T_field_uncertainty.png`
- `*_holdout_u_swirl_field_uncertainty.png`

These results are intentionally compact so physics collaborators can inspect the numerical errors and uncertainty plots without rerunning the full workflow.
