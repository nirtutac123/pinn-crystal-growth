#!/usr/bin/env python3
"""Generate publication-ready SVG figures from the final saved experiment data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PINN_ROOT = ROOT / "results" / "pinn_final"
GPR_ROOT = ROOT / "results" / "cfd_surrogate_gpr"
OUT_DIR = ROOT / "results" / "svg_figures"

METHODS = ["baseline", "gn", "agentic"]
METHOD_LABELS = {
    "baseline": "Baseline PINN",
    "gn": "GNPINN",
    "agentic": "AgenticPINN",
}
METHOD_COLORS = {
    "baseline": "#0072B2",
    "gn": "#E69F00",
    "agentic": "#009E73",
}
CONFIGS = ["crystal_mlp", "heat_mlp", "heat_siren"]
CONFIG_LABELS = {
    "crystal_mlp": "Crystal-growth\nMLP",
    "heat_mlp": "Heat equation\nMLP",
    "heat_siren": "Heat equation\nSIREN",
}
DATASETS = ["temperature", "crucible", "crystal"]
DATASET_LABELS = {
    "temperature": "Temperature sweep",
    "crucible": "Crucible rotation",
    "crystal": "Crystal rotation",
}
DATASET_COLORS = {
    "temperature": "#0072B2",
    "crucible": "#CC79A7",
    "crystal": "#D55E00",
}
TARGETS = ["u_r", "u_z", "u_swirl", "p", "T"]
TARGET_LABELS = {
    "u_r": "u_r",
    "u_z": "u_z",
    "u_swirl": "u_theta",
    "p": "p",
    "T": "T",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "svg.fonttype": "path",
            "savefig.bbox": "tight",
        }
    )


def read_pinn_summary() -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    with (PINN_ROOT / "summary.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = dict(raw)
            for key in [
                "epochs",
                "collocation_points",
                "final_total_loss",
                "final_pde_loss",
                "final_bc_ic_loss",
                "final_lr",
            ]:
                value = raw.get(key, "")
                row[key] = float(value) if value else np.nan
            rows[str(raw["run_dir"])] = row
    return rows


def read_history(config: str, method: str) -> dict[str, list[float]]:
    path = PINN_ROOT / f"{config}_{method}_history.json"
    return json.loads(path.read_text())


def read_gpr_metrics() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in DATASETS:
        path = GPR_ROOT / dataset / f"{dataset}_gpr_metrics.csv"
        result[dataset] = {}
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                result[dataset][raw["target"]] = {
                    key: float(value) for key, value in raw.items() if key != "target"
                }
    return result


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / filename,
        format="svg",
        facecolor="white",
        transparent=False,
        metadata={"Creator": "pinn-crystal-growth publication figure generator"},
    )
    plt.close(fig)


def grouped_values(summary, metric: str, config: str) -> list[float]:
    return [float(summary[f"{config}_{method}"][metric]) for method in METHODS]


def plot_final_total_loss(summary) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(CONFIGS))
    width = 0.23
    for offset, method in enumerate(METHODS):
        values = [float(summary[f"{config}_{method}"]["final_total_loss"]) for config in CONFIGS]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Final total loss (log scale)")
    ax.set_xticks(x, [CONFIG_LABELS[c] for c in CONFIGS])
    ax.set_title("Final total loss across PINN configurations")
    ax.legend(ncols=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "pinn_final_total_loss.svg")


def plot_component_losses(summary) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)
    x = np.arange(len(CONFIGS))
    width = 0.23
    panels = [
        ("final_pde_loss", "PDE residual loss"),
        ("final_bc_ic_loss", "Boundary / initial-condition loss"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for offset, method in enumerate(METHODS):
            values = [float(summary[f"{config}_{method}"][metric]) for config in CONFIGS]
            ax.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, [CONFIG_LABELS[c] for c in CONFIGS])
        ax.set_ylabel("Final component loss (log scale)")
    axes[0].legend(ncols=3, frameon=False, loc="upper left")
    fig.suptitle("Component-level PINN evaluation reveals different trade-offs")
    fig.tight_layout()
    save(fig, "pinn_component_losses.svg")


def plot_tradeoff(summary) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    markers = {"crystal_mlp": "o", "heat_mlp": "s", "heat_siren": "^"}
    for config in CONFIGS:
        for method in METHODS:
            row = summary[f"{config}_{method}"]
            ax.scatter(
                float(row["final_pde_loss"]),
                float(row["final_bc_ic_loss"]),
                s=85,
                marker=markers[config],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=METHOD_COLORS[method],
               markeredgecolor="white", markersize=8, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    config_handles = [
        Line2D([0], [0], marker=markers[config], linestyle="none", color="#555555",
               markersize=7, label=CONFIG_LABELS[config].replace("\n", " "))
        for config in CONFIGS
    ]
    method_legend = ax.legend(
        handles=method_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="upper left",
        title="Trainer",
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=config_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="lower right",
        title="Problem / network",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final PDE residual loss")
    ax.set_ylabel("Final boundary / initial-condition loss")
    ax.set_title("PDE satisfaction and boundary accuracy reveal different trade-offs")
    fig.tight_layout()
    save(fig, "pinn_pde_bc_tradeoff.svg")


def plot_convergence(config: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method in METHODS:
        history = read_history(config, method)
        values = np.asarray(history["total_loss"], dtype=float)
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, color=METHOD_COLORS[method], label=METHOD_LABELS[method], linewidth=1.25)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss (log scale)")
    ax.set_title(title)
    ax.legend(ncols=3, frameon=False)
    fig.tight_layout()
    save(fig, filename)


def plot_agentic_dynamics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, config, title in [
        (axes[0], "crystal_mlp", "Crystal-growth MLP"),
        (axes[1], "heat_mlp", "Heat-equation MLP"),
    ]:
        history = read_history(config, "agentic")
        n = len(history["pde_weight"])
        progress = np.linspace(0, 100, n)
        ax.plot(progress, history["pde_weight"], color="#D55E00", label="PDE weight")
        ax.plot(progress, history["bc_ic_weight"], color="#0072B2", label="BC/IC weight")
        ax.set_yscale("log")
        ax.set_xlabel("Training progress (%)")
        ax.set_ylabel("Adaptive loss weight (log scale)")
        ax.set_title(title)
        ax.legend(frameon=False, loc="center right")
    fig.suptitle("AgenticPINN controller weight dynamics")
    fig.tight_layout(pad=0.6)
    save(fig, "agentic_controller_dynamics.svg")


def plot_gpr_relative_rmse(gpr) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    x = np.arange(len(TARGETS))
    width = 0.24
    for offset, dataset in enumerate(DATASETS):
        values = [gpr[dataset][target]["relative_rmse_std"] for target in TARGETS]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=DATASET_LABELS[dataset],
            color=DATASET_COLORS[dataset],
        )
    ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGETS])
    ax.set_ylabel("Relative RMSE (RMSE / target standard deviation)")
    ax.set_title("Case-wise GPR holdout error across CFD targets")
    ax.legend(ncols=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "gpr_relative_rmse.svg")


def plot_gpr_uncertainty_correlation(gpr) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    x = np.arange(len(TARGETS))
    width = 0.24
    for offset, dataset in enumerate(DATASETS):
        values = [gpr[dataset][target]["abs_error_uncertainty_corr"] for target in TARGETS]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=DATASET_LABELS[dataset],
            color=DATASET_COLORS[dataset],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGETS])
    ax.set_ylabel("Correlation: absolute error vs predictive uncertainty")
    ax.set_ylim(-0.15, 0.55)
    ax.set_title("Uncertainty is informative for some targets, but not uniformly calibrated")
    ax.legend(ncols=3, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "gpr_uncertainty_error_correlation.svg")


def plot_paper_overview(summary, gpr) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    x = np.arange(len(METHODS))
    width = 0.34
    pde = grouped_values(summary, "final_pde_loss", "crystal_mlp")
    bc = grouped_values(summary, "final_bc_ic_loss", "crystal_mlp")
    axes[0].bar(x - width / 2, pde, width, color="#D55E00", label="PDE residual")
    axes[0].bar(x + width / 2, bc, width, color="#0072B2", label="BC/IC loss")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=12)
    axes[0].set_ylabel("Final loss component (log scale)")
    axes[0].set_title("(a) Crystal-growth MLP trade-off")
    axes[0].legend(frameon=False)

    t_values = [gpr[dataset]["T"]["relative_rmse_std"] for dataset in DATASETS]
    axes[1].bar(
        np.arange(len(DATASETS)),
        t_values,
        color=[DATASET_COLORS[d] for d in DATASETS],
    )
    axes[1].set_xticks(np.arange(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS], rotation=12)
    axes[1].set_ylabel(r"Temperature relative RMSE")
    axes[1].set_title("(b) Corrected CFD case-wise holdout")
    for idx, value in enumerate(t_values):
        axes[1].text(idx, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    axes[1].set_ylim(0, max(t_values) * 1.28)

    fig.suptitle("Component-level PINN findings and CFD surrogate validation")
    fig.tight_layout(pad=0.6)
    save(fig, "paper_results_overview.svg")


def main() -> None:
    configure_style()
    summary = read_pinn_summary()
    gpr = read_gpr_metrics()

    plot_final_total_loss(summary)
    plot_component_losses(summary)
    plot_tradeoff(summary)
    plot_convergence("crystal_mlp", "Crystal-growth MLP convergence", "crystal_mlp_convergence.svg")
    plot_convergence("heat_mlp", "Heat-equation MLP convergence", "heat_mlp_convergence.svg")
    plot_agentic_dynamics()
    plot_gpr_relative_rmse(gpr)
    plot_gpr_uncertainty_correlation(gpr)
    plot_paper_overview(summary, gpr)

    generated = sorted(path.name for path in OUT_DIR.glob("*.svg"))
    print(f"Generated {len(generated)} SVG figures in {OUT_DIR}")
    for name in generated:
        print(f"- {name}")


if __name__ == "__main__":
    main()
