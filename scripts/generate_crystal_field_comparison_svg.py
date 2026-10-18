"""Create a paper-readable SVG comparison of the final crystal PINN fields."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MLP


TIME_VALUE = 0.20
GRID_SIZE = 161
METHODS = [
    ("Baseline PINN", "crystal_mlp_baseline"),
    ("GNPINN", "crystal_mlp_gn"),
    ("AgenticPINN", "crystal_mlp_agentic"),
]


def load_fields(run_name, points, shape, checkpoint_root):
    model = MLP(3, 4, hidden_layers=4, neurons_per_layer=50, activation="tanh")
    checkpoint = checkpoint_root / run_name / "crystal_growth_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Generate the final crystal runs "
            "or pass their directory with --checkpoint-root."
        )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        prediction = model(points).reshape(*shape, 4).numpy()
    u_x = prediction[..., 0]
    u_y = prediction[..., 1]
    pressure = prediction[..., 2]
    temperature = prediction[..., 3]
    speed = np.sqrt(u_x**2 + u_y**2)
    return {
        "u_x": u_x,
        "u_y": u_y,
        "speed": speed,
        "pressure": pressure,
        "temperature": temperature,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "thesis_runs" / "final",
        help="Directory containing the three final crystal run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "appendix_fields",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, GRID_SIZE)
    y = np.linspace(0.0, 1.0, GRID_SIZE)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    points = torch.tensor(
        np.column_stack(
            [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, TIME_VALUE)]
        ),
        dtype=torch.float32,
    )
    interface = 0.2 + 0.05 * np.sin(np.pi * x_grid) + 0.1 * TIME_VALUE
    solid_mask = y_grid < interface

    results = [
        (name, load_fields(run, points, x_grid.shape, checkpoint_root))
        for name, run in METHODS
    ]
    variable_rows = [
        ("speed", r"Velocity magnitude $|\mathbf{u}|$", "viridis"),
        ("pressure", r"Pressure $p$", "coolwarm"),
        ("temperature", r"Temperature $T$", "inferno"),
    ]

    ranges = {}
    for key, _, _ in variable_rows:
        values = np.concatenate([fields[key][~solid_mask] for _, fields in results])
        ranges[key] = (float(np.min(values)), float(np.max(values)))

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.0), sharex=True, sharey=True)
    row_contours = []

    for row, (key, row_title, cmap) in enumerate(variable_rows):
        lower, upper = ranges[key]
        levels = np.linspace(lower, upper, 24)
        contour = None
        for column, (method_name, fields) in enumerate(results):
            axis = axes[row, column]
            masked = np.ma.array(fields[key], mask=solid_mask)
            contour = axis.contourf(
                x_grid,
                y_grid,
                masked,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            if row == 0:
                axis.set_title(method_name, fontweight="bold")
                stride = 7
                axis.streamplot(
                    x[::stride],
                    y[::stride],
                    np.ma.array(fields["u_x"], mask=solid_mask)[::stride, ::stride],
                    np.ma.array(fields["u_y"], mask=solid_mask)[::stride, ::stride],
                    color="white",
                    density=0.75,
                    linewidth=0.45,
                    arrowsize=0.45,
                )
            axis.plot(x, interface[0], color="black", linewidth=0.7)
            axis.fill_between(x, 0.0, interface[0], color="0.82", zorder=5)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks([0.0, 0.5, 1.0])
            axis.set_yticks([0.0, 0.5, 1.0])
            axis.set_aspect("equal")
            if column == 0:
                axis.set_ylabel(r"Vertical coordinate $y$")
            if row == 2:
                axis.set_xlabel(r"Horizontal coordinate $x$")
        row_contours.append((contour, row_title))

    for row, (contour, label) in enumerate(row_contours):
        colorbar = fig.colorbar(
            contour,
            ax=axes[row, :].tolist(),
            fraction=0.025,
            pad=0.025,
            aspect=24,
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        rf"Simplified crystal-growth fields at $t={TIME_VALUE:.2f}$",
        y=0.985,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.08, top=0.93, wspace=0.16, hspace=0.15)

    output_path = output_dir / "crystal_mlp_fields_t020_comparison.svg"
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
