# Results Index

This folder contains selected lightweight result artifacts that are useful for review. Larger exploratory outputs and model checkpoints are still ignored by Git.

## Publication-Ready SVG Figures

`results/svg_figures/` is a dedicated vector-figure folder. It contains only
SVG files generated from the final saved PINN and corrected CFD surrogate
metrics:

- `pinn_final_total_loss.svg`
- `pinn_component_losses.svg`
- `pinn_pde_bc_tradeoff.svg`
- `crystal_mlp_convergence.svg`
- `heat_mlp_convergence.svg`
- `agentic_controller_dynamics.svg`
- `gpr_relative_rmse.svg`
- `gpr_uncertainty_error_correlation.svg`
- `paper_results_overview.svg`

The compact `paper_results_overview.svg` is the best starting point for a short
paper. The remaining figures provide supporting material for a longer paper,
presentation, repository documentation, or supplementary material.

The SVG collection is reproducible with:

```bash
python tools/generate_svg_figures.py
```

The PINN inputs used by the generator are stored in `results/pinn_final/`. The
GPR inputs are the committed metric files under `results/cfd_surrogate_gpr/`.

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
