import argparse
import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-species abundance regressors on log1p counts.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-idx", required=True)
    parser.add_argument("--val-idx", required=True)
    parser.add_argument("--test-idx", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument("--model-name", default="ridge_numpy")
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - num / denom


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    xtx = x_centered.T @ x_centered
    reg = np.eye(x.shape[1], dtype=np.float64) * alpha
    coef = np.linalg.solve(xtx + reg, x_centered.T @ y_centered)
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def predict_ridge(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return x @ coef + intercept


def choose_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
) -> float:
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        coef, intercept = fit_ridge(x_train, y_train, float(alpha))
        pred = np.clip(predict_ridge(x_val, coef, intercept), a_min=0.0, a_max=None)
        score = rmse(y_val, pred)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def plot_metric(df: pd.DataFrame, value_col: str, title: str, path: Path, color: str) -> None:
    plot_df = df.sort_values(value_col, ascending=False).copy()
    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["species"], plot_df[value_col], color=color)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def observed_vs_predicted_grid(
    species: list[str],
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    path: Path,
) -> None:
    n = len(species)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    max_val = max(float(y_true_raw.max()), float(y_pred_raw.max()))
    for i, label in enumerate(species):
        ax = axes[i]
        ax.scatter(y_true_raw[:, i], y_pred_raw[:, i], s=18, alpha=0.65, color="#1f6f8b")
        ax.plot([0, max_val], [0, max_val], linestyle="--", color="#555555", linewidth=1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(Path(args.labels_json).read_text())
    feature_names = json.loads(Path(args.features_json).read_text())
    data = np.load(args.data)
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)
    test_idx = np.load(args.test_idx)

    y = data[:, : args.label_dim]
    x = data[:, args.label_dim :]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_fit = x[np.sort(np.concatenate([train_idx, val_idx]))]
    y_fit = y[np.sort(np.concatenate([train_idx, val_idx]))]
    x_test, y_test = x[test_idx], y[test_idx]

    rows = []
    model_payload = {}
    y_pred_test = np.zeros_like(y_test)
    alphas = np.logspace(-3, 3, 9)

    for i, label in enumerate(labels):
        best_alpha = choose_alpha(x_train, y_train[:, i], x_val, y_val[:, i], alphas)
        coef, intercept = fit_ridge(x_fit, y_fit[:, i], best_alpha)
        pred_log = np.clip(predict_ridge(x_test, coef, intercept), a_min=0.0, a_max=None)
        y_pred_test[:, i] = pred_log

        true_raw = np.expm1(y_test[:, i])
        pred_raw = np.expm1(pred_log)

        rows.append(
            {
                "species": label,
                "support_nonzero_test": int((true_raw > 0).sum()),
                "alpha": float(best_alpha),
                "rmse_log1p": rmse(y_test[:, i], pred_log),
                "rmse_raw": rmse(true_raw, pred_raw),
                "mae_raw": mae(true_raw, pred_raw),
                "r2_raw": r2(true_raw, pred_raw),
            }
        )
        model_payload[label] = {
            "alpha": float(best_alpha),
            "coef": coef,
            "intercept": float(intercept),
        }

    metrics = pd.DataFrame(rows)
    metrics_csv = out_dir / "species_abundance_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    summary = {
        "model_name": args.model_name,
        "n_species": len(labels),
        "n_features": len(feature_names),
        "mean_rmse_raw": float(metrics["rmse_raw"].mean()),
        "median_rmse_raw": float(metrics["rmse_raw"].median()),
        "mean_mae_raw": float(metrics["mae_raw"].mean()),
        "mean_r2_raw": float(metrics["r2_raw"].dropna().mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_metric(metrics, "rmse_raw", "Per-Species RMSE (Raw Count Scale)", out_dir / "species_rmse_raw.png", "#c65d2e")
    plot_metric(metrics, "mae_raw", "Per-Species MAE (Raw Count Scale)", out_dir / "species_mae_raw.png", "#d98c2b")
    plot_metric(metrics, "r2_raw", "Per-Species R^2 (Raw Count Scale)", out_dir / "species_r2_raw.png", "#1f6f8b")
    plot_metric(metrics, "support_nonzero_test", "Nonzero Test Samples by Species", out_dir / "species_support_nonzero.png", "#567d46")
    observed_vs_predicted_grid(labels, np.expm1(y_test), np.expm1(y_pred_test), out_dir / "observed_vs_predicted_grid.png")

    joblib.dump(
        {
            "models": model_payload,
            "feature_names": feature_names,
            "label_names": labels,
        },
        out_dir / "ridge_numpy_models.joblib",
    )
    np.save(out_dir / "y_test_log1p.npy", y_test)
    np.save(out_dir / "y_pred_log1p.npy", y_pred_test)
    np.save(out_dir / "test_idx.npy", test_idx)

    print(metrics_csv)
    print(out_dir / "summary.json")
    print(out_dir / "species_rmse_raw.png")
    print(out_dir / "species_mae_raw.png")
    print(out_dir / "species_r2_raw.png")
    print(out_dir / "species_support_nonzero.png")
    print(out_dir / "observed_vs_predicted_grid.png")


if __name__ == "__main__":
    main()
