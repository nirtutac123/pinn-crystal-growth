import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from core.cfd_data import TARGET_COLUMNS


def fit_gpr_surrogate(x_train_scaled, y_train_scaled, seed: int):
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=np.ones(x_train_scaled.shape[1]), length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        optimizer=None,
        random_state=seed,
    )
    model.fit(x_train_scaled, y_train_scaled)
    return model


def evaluate_gpr(model, x_test_scaled, y_test, y_scaler):
    pred_scaled, std_scaled = model.predict(x_test_scaled, return_std=True)
    pred = y_scaler.inverse_transform(pred_scaled)
    if std_scaled.ndim == 1:
        std = np.repeat(std_scaled[:, None], y_test.shape[1], axis=1) * y_scaler.std
    else:
        std = std_scaled * y_scaler.std

    errors = pred - y_test
    metrics = []
    for idx, target in enumerate(TARGET_COLUMNS):
        err = errors[:, idx]
        true = y_test[:, idx]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        denom = float(np.std(true)) if np.std(true) > 1e-12 else 1.0
        uncertainty = std[:, idx]
        abs_err = np.abs(err)
        corr = (
            float(np.corrcoef(abs_err, uncertainty)[0, 1])
            if np.std(abs_err) > 1e-12 and np.std(uncertainty) > 1e-12
            else 0.0
        )
        metrics.append(
            {
                "target": target,
                "mae": mae,
                "rmse": rmse,
                "relative_rmse_std": rmse / denom,
                "mean_predictive_std": float(np.mean(uncertainty)),
                "max_predictive_std": float(np.max(uncertainty)),
                "abs_error_uncertainty_corr": corr,
            }
        )
    return pred, std, metrics


def plot_gpr_outputs(out_dir: Path, y_test, pred, std, test_frame, holdout_case):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(15, 3), constrained_layout=True)
    for idx, target in enumerate(TARGET_COLUMNS):
        axes[idx].scatter(y_test[:, idx], pred[:, idx], s=4, alpha=0.35)
        low = min(y_test[:, idx].min(), pred[:, idx].min())
        high = max(y_test[:, idx].max(), pred[:, idx].max())
        axes[idx].plot([low, high], [low, high], color="black", linewidth=1)
        axes[idx].set_title(target)
        axes[idx].set_xlabel("CFD")
        axes[idx].set_ylabel("GPR")
    fig.suptitle(f"GPR holdout prediction parity: {holdout_case.case_name}")
    fig.savefig(out_dir / f"{holdout_case.dataset}_holdout_parity.png", dpi=300)
    plt.close(fig)

    for target in ["T", "u_swirl"]:
        idx = TARGET_COLUMNS.index(target)
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
        values = [y_test[:, idx], pred[:, idx], np.abs(pred[:, idx] - y_test[:, idx]), std[:, idx]]
        titles = [f"CFD {target}", f"GPR {target}", f"Absolute error {target}", f"Uncertainty {target}"]
        cmaps = ["inferno", "inferno", "magma", "viridis"]
        for ax, vals, title, cmap in zip(axes, values, titles, cmaps):
            sc = ax.scatter(test_frame["r"], test_frame["z"], c=vals, s=3, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            fig.colorbar(sc, ax=ax)
        fig.suptitle(f"{holdout_case.dataset} holdout case: {holdout_case.case_name}")
        fig.savefig(out_dir / f"{holdout_case.dataset}_holdout_{target}_field_uncertainty.png", dpi=300)
        plt.close(fig)


def write_gpr_metrics(out_dir: Path, args, train_cases, holdout_case, metrics, model):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "surrogate_model": "GaussianProcessRegressor",
        "holdout_case": holdout_case.case_name,
        "holdout_parameter": holdout_case.case_parameter,
        "train_cases": [case.case_name for case in train_cases],
        "max_rows_per_case": args.max_rows_per_case,
        "data_root": str(Path(args.data_root).resolve()),
        "kernel": str(model.kernel_),
        "metrics": metrics,
    }
    with open(out_dir / f"{holdout_case.dataset}_gpr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{holdout_case.dataset}_gpr_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "mae",
                "rmse",
                "relative_rmse_std",
                "mean_predictive_std",
                "max_predictive_std",
                "abs_error_uncertainty_corr",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "| target | rmse | relative_rmse_std | mean_predictive_std |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {target} | {rmse:.6e} | {relative_rmse_std:.4f} | {mean_predictive_std:.6e} |".format(**row)
        )
    (out_dir / f"{holdout_case.dataset}_gpr_metrics.md").write_text("\n".join(lines) + "\n")
