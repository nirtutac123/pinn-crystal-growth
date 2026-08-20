import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cfd_data import DEFAULT_DATA_ROOT, prepare_case_holdout, select_cases, validate_cases
from core.gpr_surrogate import evaluate_gpr, fit_gpr_surrogate, plot_gpr_outputs, write_gpr_metrics


OUT_ROOT = ROOT / "results" / "cfd_surrogate_gpr"


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

    train_cases, holdout_case, x_scaler, y_scaler, x_train, y_train, x_test, y_test, test_frame = prepare_case_holdout(
        cases, args.holdout, args.max_rows_per_case, args.seed
    )
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_test_scaled = x_scaler.transform(x_test)

    model = fit_gpr_surrogate(x_train_scaled, y_train_scaled, args.seed)
    pred, std, metrics = evaluate_gpr(model, x_test_scaled, y_test, y_scaler)

    out_dir = OUT_ROOT / holdout_case.dataset
    write_gpr_metrics(out_dir, args, train_cases, holdout_case, metrics, model)
    plot_gpr_outputs(out_dir, y_test, pred, std, test_frame, holdout_case)

    print(f"Dataset: {holdout_case.dataset}")
    print("Surrogate: Gaussian Process Regression")
    print(f"Holdout case: {holdout_case.case_name} ({holdout_case.case_parameter})")
    print(f"Training cases: {len(train_cases)}")
    for row in metrics:
        print(
            f"{row['target']}: RMSE={row['rmse']:.6e}, "
            f"relative RMSE={row['relative_rmse_std']:.4f}, "
            f"mean uncertainty={row['mean_predictive_std']:.6e}"
        )
    print(f"Wrote GPR outputs to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a fast Gaussian Process Regression surrogate on corrected Czochralski CFD sweep data "
            "and write holdout metrics plus uncertainty plots."
        )
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
    parser.add_argument("--validate_only", action="store_true", help="Load data and print detected cases without fitting.")
    parser.add_argument(
        "--max_rows_per_case",
        type=int,
        default=80,
        help="Subsampled rows per training case. Keep modest because exact GPR scales cubically with rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
