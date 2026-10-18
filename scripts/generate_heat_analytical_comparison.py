"""Reproduce the heat MLP runs and compare them with the analytical solution."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equations import Heat1D
from models import AgenticPINN, GNPINN, MLP, PINN
from utils import exact_solution_heat_1d, set_seed


SEED = 42
EPOCHS = 2000
COLLOCATION_POINTS = 1500
LEARNING_RATE = 1e-3
GRID_SIZE = 201


def train_model(trainer_class):
    set_seed(SEED)
    device = torch.device("cpu")
    equation = Heat1D(
        domain_ranges={"x": (0, 1), "t": (0, 1)},
        diffusivity=1.0,
        device=device,
    )
    model = MLP(2, 1, hidden_layers=4, neurons_per_layer=50, activation="tanh")
    trainer = trainer_class(model, equation, device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = trainer.train(
        optimizer,
        n_collocation_points=COLLOCATION_POINTS,
        n_epochs=EPOCHS,
        log_interval=500,
    )
    return model.eval(), history


def evaluate(model, points, exact):
    with torch.no_grad():
        prediction = model(points).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()
    difference = prediction - exact
    rmse = float(np.sqrt(np.mean(difference**2)))
    relative_l2 = float(np.linalg.norm(difference.ravel()) / np.linalg.norm(exact.ravel()))
    return prediction, rmse, relative_l2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "appendix_fields",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(0, 1, GRID_SIZE)
    t = torch.linspace(0, 1, GRID_SIZE)
    x_grid, t_grid = torch.meshgrid(x, t, indexing="xy")
    points = torch.stack([x_grid.reshape(-1), t_grid.reshape(-1)], dim=1)
    exact = exact_solution_heat_1d(x_grid, t_grid).cpu().numpy()

    methods = [
        ("Baseline PINN", PINN),
        ("GNPINN", GNPINN),
        ("AgenticPINN", AgenticPINN),
    ]
    fields = [("Analytical solution", exact)]
    metrics = {
        "evaluation_grid": [GRID_SIZE, GRID_SIZE],
        "seed": SEED,
        "epochs": EPOCHS,
        "collocation_points": COLLOCATION_POINTS,
        "learning_rate": LEARNING_RATE,
        "models": {},
    }

    for name, trainer_class in methods:
        print(f"\nTraining {name}")
        model, history = train_model(trainer_class)
        prediction, rmse, relative_l2 = evaluate(model, points, exact)
        fields.append((name, prediction))
        metrics["models"][name] = {
            "rmse": rmse,
            "relative_l2": relative_l2,
            "final_pde_loss": float(history["pde_loss"][-1]),
            "final_bc_ic_loss": float(history["bc_ic_loss"][-1]),
            "final_weighted_total": float(history["total_loss"][-1]),
        }

    lower = min(float(np.min(field)) for _, field in fields)
    upper = max(float(np.max(field)) for _, field in fields)
    levels = np.linspace(lower, upper, 25)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), sharex=True, sharey=True)
    axes = axes.ravel()
    contour = None
    for index, ((name, field), axis) in enumerate(zip(fields, axes)):
        contour = axis.contourf(x_grid.numpy(), t_grid.numpy(), field, levels=levels, cmap="viridis", extend="both")
        if index == 0:
            axis.set_title(name)
        else:
            model_metrics = metrics["models"][name]
            axis.set_title(f"{name}\nRMSE={model_metrics['rmse']:.3e}")
        axis.set_aspect("equal")
    axes[2].set_xlabel(r"Spatial coordinate $x$")
    axes[3].set_xlabel(r"Spatial coordinate $x$")
    axes[0].set_ylabel(r"Time $t$")
    axes[2].set_ylabel(r"Time $t$")
    colorbar = fig.colorbar(contour, ax=axes.tolist(), fraction=0.035, pad=0.035)
    colorbar.set_label(r"Temperature $T(x,t)$")
    fig.suptitle("Heat-equation analytical benchmark and PINN predictions", y=0.985, fontsize=11)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.10, top=0.88, wspace=0.13, hspace=0.30)

    svg_path = output_dir / "heat_mlp_method_comparison.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / "heat_analytical_benchmark_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {svg_path}")
    print(f"Saved {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
