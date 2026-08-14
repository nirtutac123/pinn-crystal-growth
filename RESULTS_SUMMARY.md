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

## Corrected CFD Surrogate Results

The corrected CFD surrogate experiments use case-wise holdout evaluation.

| Dataset | Holdout case | Main observation |
| --- | --- | --- |
| Temperature-change | `T_hot = 1780 K` | Low relative errors across all target variables |
| Crucible-rotation | `omega = -4 rpm` | Low relative errors across all target variables |
| Crystal-rotation | `omega = 7 rpm` | Harder prediction, especially for `u_r` and `u_z` |

The crystal-rotation case is harder because the corrected sweep shows stronger meridional-flow changes around the 6 to 7 rpm region.

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
