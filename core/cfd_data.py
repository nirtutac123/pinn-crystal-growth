import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "external_repos"
TEMP_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_Study_TempChange" / "Steady"
CRUCIBLE_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_study_Crucible_Sweep"
CRYSTAL_REPO_ROOT = DEFAULT_DATA_ROOT / "CZ_Crystal_Sweep"

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
    return [
        {
            "case_name": case.case_name,
            "case_parameter": case.case_parameter,
            "rows": len(case.frame),
            "path": str(case.path),
        }
        for case in cases
    ]


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(max_rows, random_state=seed).copy()


def prepare_case_holdout(cases: List[CaseData], holdout: Optional[str], max_rows_per_case: int, seed: int):
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
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_test = test_frame[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(y_train)
    return train_cases, holdout_case, x_scaler, y_scaler, x_train, y_train, x_test, y_test, test_frame
