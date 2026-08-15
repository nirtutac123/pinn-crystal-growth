# Dataset Guide

This repository uses two kinds of data.

## 1. PINN Training Points

The main PINN experiments do not rely on a conventional labelled dataset. Instead, the code samples:

- PDE collocation points inside the domain
- boundary-condition points
- initial-condition points where applicable

The loss function checks whether the neural-network predictions satisfy the governing equations and the boundary or initial constraints.

## 2. Corrected Czochralski CFD Sweep Data

The supervised CFD surrogate experiment uses corrected external Czochralski simulation datasets shared during thesis development. These data are used to connect the simplified PINN study to CFD-style crystal-growth fields.

Required external folders:

```text
external_repos/CZ_Study_TempChange/Steady/
external_repos/CZ_study_Crucible_Sweep/
external_repos/CZ_Crystal_Sweep/
```

The loader also accepts the extracted raw zip layout:

```text
data/temperature/
data/crucible/
data/crystal/
```

When using this layout, pass the parent directory with `--data_root`:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /path/to/extracted --validate_only
```

The datasets represent separate one-parameter sweeps:

| Dataset | Process parameter | Example holdout |
| --- | --- | --- |
| Temperature-change | hot-boundary temperature `T_hot` | `1780 K` |
| Crucible-rotation | crucible rotation `omega` | `-4 rpm` |
| Crystal-rotation | crystal rotation `omega` | `7 rpm` |

Each CFD case is expected to contain spatial coordinates and physical fields:

```text
r, z, u_r, u_z, u_swirl, p, T
```

The supervised surrogate maps:

```text
(r, z, case_parameter) -> (u_r, u_z, u_swirl, p, T)
```

## Why Case-Wise Holdout?

The surrogate experiment holds out one complete CFD case for testing. This is stricter than randomly splitting rows because the model must generalize to an unseen process setting.

## Important Data Correction

Earlier CFD sweep data had a boundary-condition issue in which one surface was frozen. The final repository workflow uses the corrected temperature, crucible-rotation, and crystal-rotation sweep datasets.

The previous swirl-change dataset is not used for final thesis claims.
