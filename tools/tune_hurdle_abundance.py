import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_hurdle_abundance import choose_alpha, fit_ridge, predict_ridge, rmse, mae, r2


GRID_LIBRARY = {
    "broad_low": np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=np.float64),
    "mid": np.array([1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=np.float64),
    "high": np.array([1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=np.float64),
    "wide": np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64),
    "very_high": np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0], dtype=np.float64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hurdle abundance hyperparameters on validation performance.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-idx", required=True)
    parser.add_argument("--val-idx", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument(
        "--presence-grid-names",
        default="mid,high,wide",
        help="Comma-separated predefined grid names for the presence stage.",
    )
    parser.add_argument(
        "--positive-grid-names",
        default="broad_low,mid,wide",
        help="Comma-separated predefined grid names for the positive-abundance stage.",
    )
    return parser.parse_args()


def evaluate_combo(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_log_train: np.ndarray,
    y_log_val: np.ndarray,
    y_raw_train: np.ndarray,
    y_raw_val: np.ndarray,
    presence_grid: np.ndarray,
    positive_grid: np.ndarray,
    labels: list[str],
) -> tuple[pd.DataFrame, dict]:
    y_pres_train = (y_raw_train > 0).astype(np.float64)
    y_pres_val = (y_raw_val > 0).astype(np.float64)

    rows = []
    y_pred_expected = np.zeros_like(y_raw_val)

    for i, label in enumerate(labels):
        alpha_presence = choose_alpha(
            x_train,
            y_pres_train[:, i],
            x_val,
            y_pres_val[:, i],
            presence_grid,
            clip_min=0.0,
            clip_max=1.0,
        )
        coef_presence, intercept_presence = fit_ridge(x_train, y_pres_train[:, i], alpha_presence)
        pred_presence = np.clip(predict_ridge(x_val, coef_presence, intercept_presence), a_min=0.0, a_max=1.0)

        pos_train_mask = y_pres_train[:, i] > 0
        pos_val_mask = y_pres_val[:, i] > 0
        if pos_train_mask.sum() >= 2 and pos_val_mask.sum() >= 1:
            alpha_positive = choose_alpha(
                x_train[pos_train_mask],
                y_log_train[pos_train_mask, i],
                x_val[pos_val_mask],
                y_log_val[pos_val_mask, i],
                positive_grid,
                clip_min=0.0,
                clip_max=None,
            )
            coef_positive, intercept_positive = fit_ridge(
                x_train[pos_train_mask],
                y_log_train[pos_train_mask, i],
                alpha_positive,
            )
            pred_positive_log = np.clip(predict_ridge(x_val, coef_positive, intercept_positive), a_min=0.0, a_max=None)
        else:
            alpha_positive = np.nan
            pred_positive_log = np.full(len(x_val), float(y_log_train[pos_train_mask, i].mean()) if pos_train_mask.any() else 0.0)

        expected_raw = pred_presence * np.expm1(pred_positive_log)
        y_pred_expected[:, i] = expected_raw

        rows.append(
            {
                "species": label,
                "alpha_presence": float(alpha_presence),
                "alpha_positive": float(alpha_positive) if not np.isnan(alpha_positive) else np.nan,
                "rmse_raw": rmse(y_raw_val[:, i], expected_raw),
                "mae_raw": mae(y_raw_val[:, i], expected_raw),
                "r2_raw": r2(y_raw_val[:, i], expected_raw),
            }
        )

    metrics = pd.DataFrame(rows)
    summary = {
        "mean_rmse_raw": float(metrics["rmse_raw"].mean()),
        "mean_mae_raw": float(metrics["mae_raw"].mean()),
        "mean_r2_raw": float(metrics["r2_raw"].dropna().mean()),
    }
    return metrics, summary


def grid_to_string(arr: np.ndarray) -> str:
    return ",".join(f"{x:g}" for x in arr.tolist())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)
    labels = json.loads(Path(args.labels_json).read_text())

    y_log = data[:, : args.label_dim]
    x = data[:, args.label_dim :]
    y_raw = np.expm1(y_log)

    x_train, x_val = x[train_idx], x[val_idx]
    y_log_train, y_log_val = y_log[train_idx], y_log[val_idx]
    y_raw_train, y_raw_val = y_raw[train_idx], y_raw[val_idx]

    presence_names = [name.strip() for name in args.presence_grid_names.split(",") if name.strip()]
    positive_names = [name.strip() for name in args.positive_grid_names.split(",") if name.strip()]

    rows = []
    best = None
    best_metrics = None
    for p_name, q_name in itertools.product(presence_names, positive_names):
        p_grid = GRID_LIBRARY[p_name]
        q_grid = GRID_LIBRARY[q_name]
        metrics, summary = evaluate_combo(
            x_train,
            x_val,
            y_log_train,
            y_log_val,
            y_raw_train,
            y_raw_val,
            p_grid,
            q_grid,
            labels,
        )
        row = {
            "presence_grid_name": p_name,
            "positive_grid_name": q_name,
            "presence_alpha_grid": grid_to_string(p_grid),
            "positive_alpha_grid": grid_to_string(q_grid),
            **summary,
        }
        rows.append(row)
        if best is None or summary["mean_rmse_raw"] < best["mean_rmse_raw"]:
            best = row
            best_metrics = metrics

    search_df = pd.DataFrame(rows).sort_values("mean_rmse_raw", ascending=True)
    search_df.to_csv(out_dir / "hurdle_tuning_search.csv", index=False)
    (out_dir / "best_hyperparameters.json").write_text(json.dumps(best, indent=2))
    if best_metrics is not None:
        best_metrics.to_csv(out_dir / "best_validation_species_metrics.csv", index=False)

    print(out_dir / "hurdle_tuning_search.csv")
    print(out_dir / "best_hyperparameters.json")
    if best_metrics is not None:
        print(out_dir / "best_validation_species_metrics.csv")


if __name__ == "__main__":
    main()
