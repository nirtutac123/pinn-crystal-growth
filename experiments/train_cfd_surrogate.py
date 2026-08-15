import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "external_repos"
TEMP_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_Study_TempChange" / "Steady"
CRUCIBLE_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_study_Crucible_Sweep"
CRYSTAL_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_Crystal_Sweep"
OUT_ROOT = ROOT / "results" / "cfd_surrogate"
FEATURE_COLUMNS = ["r", "z", "case_parameter"]
TARGET_COLUMNS = ["u_r", "u_z", "u_swirl", "p", "T"]


@dataclass
class CaseData:
    dataset: str
    case_name: str
    case_parameter: float
    path: Path
    frame: pd.DataFrame


class Standardizer:
    def fit(self, values: np.ndarray):
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers = []
        current_dim = in_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def infer_temperature(path: Path) -> float:
    if path.name == "cz_baseline.csv":
        return 1750.0
    match = re.search(r"temp\s*([0-9]+)", path.name)
    if not match:
        match = re.search(r"cz_([0-9]+)", path.name)
    if not match:
        raise ValueError(f"Could not infer temperature from {path.name}")
    return float(match.group(1))


def infer_crystal_rpm(path: Path) -> float:
    match = re.search(r"crystal_([0-9]+)rpm", path.name)
    if not match:
        raise ValueError(f"Could not infer crystal rpm from {path.name}")
    return float(match.group(1))


def infer_crucible_rpm(path: Path) -> float:
    match = re.search(r"crucible_m([0-9]+(?:p[0-9]+)?)rpm", path.name)
    if not match:
        raise ValueError(f"Could not infer crucible rpm from {path.name}")
    return -float(match.group(1).replace("p", "."))


def dataset_roots(data_root: Path):
    """Return dataset folders for either repo-clone or extracted data.zip layouts."""
    candidates = {
        "temperature": [
            data_root / "temperature",
            data_root / "data" / "temperature",
            data_root / "CZ_Study_TempChange" / "Steady",
            TEMP_REPO_ROOT,
        ],
        "crucible": [
            data_root / "crucible",
            data_root / "data" / "crucible",
            data_root / "CZ_study_Crucible_Sweep",
            CRUCIBLE_REPO_ROOT,
        ],
        "crystal": [
            data_root / "crystal",
            data_root / "data" / "crystal",
            data_root / "CZ_Crystal_Sweep",
            CRYSTAL_REPO_ROOT,
        ],
    }
    resolved = {}
    for dataset, paths in candidates.items():
        resolved[dataset] = next((path for path in paths if path.exists()), paths[0])
    return resolved


def require_columns(frame: pd.DataFrame, path: Path):
    missing = [column for column in FEATURE_COLUMNS[:2] + TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def read_case_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, path)
    return frame


def load_temperature_cases(data_root: Path) -> List[CaseData]:
    temp_root = dataset_roots(data_root)["temperature"]
    if not temp_root.exists():
        raise FileNotFoundError(f"Missing temperature dataset folder: {temp_root}")
    cases = []
    for path in sorted(temp_root.glob("*.csv")):
        if path.name == "cz_baseline.csv" and any(p.stem in {"cz_(temp 1745)", "cz_1745"} for p in temp_root.glob("*.csv")):
            continue
        parameter = infer_temperature(path)
        frame = read_case_frame(path)
        frame["case_parameter"] = parameter
        cases.append(CaseData("temperature", path.stem, parameter, path, frame))
    return cases


def load_crystal_cases(data_root: Path) -> List[CaseData]:
    crystal_root = dataset_roots(data_root)["crystal"]
    if not crystal_root.exists():
        raise FileNotFoundError(f"Missing crystal-rotation dataset folder: {crystal_root}")
    cases = []
    for path in sorted(crystal_root.glob("*.csv")):
        parameter = infer_crystal_rpm(path)
        frame = read_case_frame(path)
        frame["case_parameter"] = parameter
        cases.append(CaseData("crystal", path.stem, parameter, path, frame))
    return cases


def load_crucible_cases(data_root: Path) -> List[CaseData]:
    crucible_root = dataset_roots(data_root)["crucible"]
    if not crucible_root.exists():
        raise FileNotFoundError(f"Missing crucible-rotation dataset folder: {crucible_root}")
    cases = []
    for path in sorted(crucible_root.glob("*.csv")):
        parameter = infer_crucible_rpm(path)
        frame = read_case_frame(path)
        frame["case_parameter"] = parameter
        cases.append(CaseData("crucible", path.stem, parameter, path, frame))
    return cases


def select_cases(dataset: str, data_root: Path) -> List[CaseData]:
    if dataset == "temperature":
        return load_temperature_cases(data_root)
    if dataset == "crystal":
        return load_crystal_cases(data_root)
    if dataset == "crucible":
        return load_crucible_cases(data_root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def validate_cases(cases: List[CaseData]):
    rows = []
    for case in cases:
        rows.append(
            {
                "case_name": case.case_name,
                "case_parameter": case.case_parameter,
                "rows": len(case.frame),
                "path": str(case.path),
            }
        )
    return rows


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(max_rows, random_state=seed).copy()


def prepare_split(cases: List[CaseData], holdout: Optional[str], max_rows_per_case: int, seed: int):
    if holdout is None:
        holdout_case = cases[-1]
    else:
        matches = [case for case in cases if case.case_name == holdout or str(case.case_parameter) == holdout]
        if not matches:
            available = ", ".join(case.case_name for case in cases)
            raise ValueError(f"Unknown holdout case '{holdout}'. Available cases: {available}")
        holdout_case = matches[0]

    train_cases = [case for case in cases if case.case_name != holdout_case.case_name]
    train_frames = [
        sample_frame(case.frame, max_rows_per_case, seed + idx)
        for idx, case in enumerate(train_cases)
    ]
    test_frame = holdout_case.frame.copy()

    train_df = pd.concat(train_frames, ignore_index=True)
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(y_train)
    return train_cases, holdout_case, x_scaler, y_scaler, x_train, y_train, x_test, y_test, test_frame


def train_model(x_train, y_train, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SurrogateMLP(
        in_dim=x_train.shape[1],
        out_dim=y_train.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        history.append({"epoch": epoch, "train_mse_scaled": running / max(seen, 1)})
    return model, history


def evaluate(model, x_test_scaled, y_test, y_scaler, args):
    device = torch.device(args.device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_test_scaled).to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled)
    errors = pred - y_test
    metrics = []
    for idx, target in enumerate(TARGET_COLUMNS):
        err = errors[:, idx]
        true = y_test[:, idx]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        denom = float(np.std(true)) if np.std(true) > 1e-12 else 1.0
        metrics.append(
            {
                "target": target,
                "mae": mae,
                "rmse": rmse,
                "relative_rmse_std": rmse / denom,
            }
        )
    return pred, metrics


def plot_outputs(out_dir: Path, history, y_test, pred, test_frame, holdout_case: CaseData):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(hist_df["epoch"], hist_df["train_mse_scaled"])
    plt.xlabel("Epoch")
    plt.ylabel("Scaled training MSE")
    plt.title(f"CFD surrogate training loss: {holdout_case.dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{holdout_case.dataset}_training_loss.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(15, 3), constrained_layout=True)
    for idx, target in enumerate(TARGET_COLUMNS):
        axes[idx].scatter(y_test[:, idx], pred[:, idx], s=4, alpha=0.35)
        low = min(y_test[:, idx].min(), pred[:, idx].min())
        high = max(y_test[:, idx].max(), pred[:, idx].max())
        axes[idx].plot([low, high], [low, high], color="black", linewidth=1)
        axes[idx].set_title(target)
        axes[idx].set_xlabel("CFD")
        axes[idx].set_ylabel("MLP")
    fig.suptitle(f"Holdout prediction parity: {holdout_case.case_name}")
    fig.savefig(out_dir / f"{holdout_case.dataset}_holdout_parity.png", dpi=300)
    plt.close(fig)

    for target in ["T", "u_swirl"]:
        idx = TARGET_COLUMNS.index(target)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        values = [y_test[:, idx], pred[:, idx], np.abs(pred[:, idx] - y_test[:, idx])]
        titles = [f"CFD {target}", f"Predicted {target}", f"Absolute error {target}"]
        cmaps = ["inferno", "inferno", "magma"]
        for ax, vals, title, cmap in zip(axes, values, titles, cmaps):
            sc = ax.scatter(test_frame["r"], test_frame["z"], c=vals, s=3, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("r (m)")
            ax.set_ylabel("z (m)")
            fig.colorbar(sc, ax=ax)
        fig.suptitle(f"{holdout_case.dataset} holdout case: {holdout_case.case_name}")
        fig.savefig(out_dir / f"{holdout_case.dataset}_holdout_{target}_field.png", dpi=300)
        plt.close(fig)


def write_metrics(out_dir: Path, args, train_cases, holdout_case, history, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "holdout_case": holdout_case.case_name,
        "holdout_parameter": holdout_case.case_parameter,
        "train_cases": [case.case_name for case in train_cases],
        "epochs": args.epochs,
        "max_rows_per_case": args.max_rows_per_case,
        "hidden_layers": args.hidden_layers,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "data_root": str(Path(args.data_root).resolve()),
        "final_train_mse_scaled": history[-1]["train_mse_scaled"],
        "metrics": metrics,
    }
    with open(out_dir / f"{holdout_case.dataset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{holdout_case.dataset}_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "mae", "rmse", "relative_rmse_std"])
        writer.writeheader()
        writer.writerows(metrics)
    pd.DataFrame(history).to_csv(out_dir / f"{holdout_case.dataset}_history.csv", index=False)


def run(args):
    data_root = Path(args.data_root).expanduser()
    cases = select_cases(args.dataset, data_root)
    if len(cases) < 2:
        raise RuntimeError("At least two cases are required for a train/holdout split.")
    if args.validate_only:
        rows = validate_cases(cases)
        print(f"Dataset: {args.dataset}")
        print(f"Cases found: {len(rows)}")
        for row in rows:
            print(f"- {row['case_name']}: parameter={row['case_parameter']}, rows={row['rows']}")
        return

    train_cases, holdout_case, x_scaler, y_scaler, x_train, y_train, x_test, y_test, test_frame = prepare_split(
        cases, args.holdout, args.max_rows_per_case, args.seed
    )
    x_train_scaled = x_scaler.transform(x_train).astype(np.float32)
    y_train_scaled = y_scaler.transform(y_train).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)

    model, history = train_model(x_train_scaled, y_train_scaled, args)
    pred, metrics = evaluate(model, x_test_scaled, y_test, y_scaler, args)

    out_dir = OUT_ROOT / holdout_case.dataset
    write_metrics(out_dir, args, train_cases, holdout_case, history, metrics)
    plot_outputs(out_dir, history, y_test, pred, test_frame, holdout_case)
    torch.save(model.state_dict(), out_dir / f"{holdout_case.dataset}_surrogate.pt")

    print(f"Dataset: {holdout_case.dataset}")
    print(f"Holdout case: {holdout_case.case_name} ({holdout_case.case_parameter})")
    print(f"Training cases: {len(train_cases)}")
    print(f"Final scaled train MSE: {history[-1]['train_mse_scaled']:.6e}")
    for row in metrics:
        print(
            f"{row['target']}: MAE={row['mae']:.6e}, "
            f"RMSE={row['rmse']:.6e}, relative RMSE={row['relative_rmse_std']:.4f}"
        )
    print(f"Wrote outputs to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a supervised MLP surrogate on certified Czochralski CFD sweep datasets."
    )
    parser.add_argument("--dataset", choices=["temperature", "crucible", "crystal"], default="temperature")
    parser.add_argument(
        "--data_root",
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "Root containing corrected CFD data. Supports cloned repos under external_repos/ "
            "or extracted data.zip layouts containing temperature/, crucible/, and crystal/."
        ),
    )
    parser.add_argument("--holdout", default=None, help="Case name or case parameter to reserve for testing.")
    parser.add_argument("--validate_only", action="store_true", help="Load data and print detected cases without training.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max_rows_per_case", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
