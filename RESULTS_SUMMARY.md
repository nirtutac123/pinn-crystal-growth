# Results Summary

This file summarizes the final thesis-relevant results at a high level. Generated figures and detailed metrics are written to `results/` when the experiments are run locally.

## PINN Comparison

The final PINN experiments compare:

- Baseline PINN
- Gradient-Normalized PINN
- AgenticPINN

The comparison uses:

- final total loss
- PDE residual loss
- boundary or initial-condition loss
- convergence behavior
- component-level trade-offs

## Main PINN Finding

AgenticPINN can strongly reduce PDE residual loss in selected settings, especially in heat-equation MLP and crystal-growth-oriented MLP experiments. However, this can introduce a trade-off with boundary or initial-condition accuracy.

The baseline PINN remains competitive and can achieve the lowest final total loss in the crystal-growth-oriented MLP experiment.

The main conclusion is therefore not that one method is always best. The stronger conclusion is:

> Adaptive control changes the balance of physics-informed training and should be evaluated through separate loss components, not only through one total loss value.

## Corrected CFD GPR Surrogate Results

The corrected CFD surrogate experiments use Gaussian Process Regression with case-wise holdout evaluation. GPR is used instead of an MLP because it is faster for a compact sampled training set and reports predictive uncertainty.

| Dataset | Holdout case | Main observation | Compact GPR check |
| --- | --- | --- | --- |
| Temperature-change | `T_hot = 1780 K` | Smoother thermal sweep, useful baseline surrogate check | `T` relative RMSE `0.0561`; highest relative RMSE in `u_z` |
| Crucible-rotation | `omega = -4 rpm` | Rotating-boundary case-wise generalization check | `T` relative RMSE `0.0634`; highest relative RMSE in `u_z` |
| Crystal-rotation | `omega = 7 rpm` | Harder prediction, especially for velocity components | `T` relative RMSE `0.0958`; highest relative RMSE in `u_z` |

The crystal-rotation case is harder because the corrected sweep shows stronger meridional-flow changes around the 6 to 7 rpm region.

The code supports both corrected GitHub repository layouts and the raw `data.zip` layout shared during thesis development. Selected generated GPR result files are included under `results/cfd_surrogate_gpr/`.

The uncertainty plots should be read as diagnostic support, not as proof of physical correctness. They help reviewers see where the surrogate is less confident and where additional CFD cases may be useful.

## What the Results Support

The results support three defensible claims:

1. PINNs can be trained on benchmark and simplified crystal-growth-oriented PDE constraints under local-compute conditions.
2. Adaptive and agentic control can improve PDE residual minimization in selected cases, but the trade-off with boundary or initial-condition loss must be reported.
3. Corrected CFD sweep data can be used for supervised surrogate validation and for grounding the simplified PINN study in realistic Czochralski simulation fields.

## What the Results Do Not Claim

The results do not claim that:

- the PINN fully replaces high-fidelity CFD
- AgenticPINN is universally better than all other trainers
- the simplified crystal-growth PDE is a complete industrial Czochralski furnace model
- one-parameter CFD sweeps capture all multi-parameter process interactions
