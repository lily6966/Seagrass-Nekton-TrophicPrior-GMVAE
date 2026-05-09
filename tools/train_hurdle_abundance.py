import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hurdle-style abundance models on log1p counts.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-idx", required=True)
    parser.add_argument("--val-idx", required=True)
    parser.add_argument("--test-idx", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument(
        "--presence-alpha-grid",
        default="1e-3,1e-2,1e-1,1,10,100,1000",
        help="Comma-separated ridge alpha candidates for the presence stage.",
    )
    parser.add_argument(
        "--positive-alpha-grid",
        default="1e-3,1e-2,1e-1,1,10,100,1000",
        help="Comma-separated ridge alpha candidates for the positive-abundance stage.",
    )
    parser.add_argument("--rare-positive-oversample", action="store_true", help="Oversample rare positive training rows for each target.")
    parser.add_argument("--rare-positive-threshold", default=20, type=int, help="Treat targets with at most this many positive training samples as rare.")
    parser.add_argument("--rare-positive-max-multiplier", default=5, type=int, help="Maximum duplication multiplier for rare positive rows.")
    parser.add_argument("--feature-mask-rate", default=0.0, type=float, help="Append one masked copy of training rows with this masking rate.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed for augmentation.")
    return parser.parse_args()


def parse_alpha_grid(raw: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float64)


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
    clip_min: float,
    clip_max: float | None,
) -> float:
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        coef, intercept = fit_ridge(x_train, y_train, float(alpha))
        pred = predict_ridge(x_val, coef, intercept)
        pred = np.clip(pred, a_min=clip_min, a_max=clip_max)
        score = rmse(y_val, pred)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(np.float64)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (recall_pos + recall_neg)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall_pos / (precision + recall_pos) if (precision + recall_pos) > 0 else 0.0
    return balanced_acc, f1


def apply_feature_mask(x: np.ndarray, mask_rate: float, rng: np.random.Generator) -> np.ndarray:
    if mask_rate <= 0:
        return x.copy()
    mask = rng.random(x.shape) > mask_rate
    return x * mask.astype(np.float64)


def augment_training_rows(
    x: np.ndarray,
    y_target: np.ndarray,
    rare_positive_oversample: bool,
    rare_positive_threshold: int,
    rare_positive_max_multiplier: int,
    feature_mask_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_aug = x.copy()
    y_aug = y_target.copy()

    if rare_positive_oversample:
        support = int((y_target > 0).sum())
        if 0 < support <= rare_positive_threshold:
            multiplier = min(rare_positive_max_multiplier, int(np.ceil(rare_positive_threshold / support)))
            pos_mask = y_target > 0
            if multiplier > 1 and pos_mask.any():
                x_pos = np.repeat(x[pos_mask], multiplier - 1, axis=0)
                y_pos = np.repeat(y_target[pos_mask], multiplier - 1, axis=0)
                x_aug = np.concatenate([x_aug, x_pos], axis=0)
                y_aug = np.concatenate([y_aug, y_pos], axis=0)

    if feature_mask_rate > 0:
        x_masked = apply_feature_mask(x_aug, feature_mask_rate, rng)
        x_aug = np.concatenate([x_aug, x_masked], axis=0)
        y_aug = np.concatenate([y_aug, y_aug.copy()], axis=0)

    return x_aug, y_aug


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

    y_log = data[:, : args.label_dim]
    x = data[:, args.label_dim :]
    y_raw = np.expm1(y_log)
    y_presence = (y_raw > 0).astype(np.float64)

    fit_idx = np.sort(np.concatenate([train_idx, val_idx]))
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_log_train, y_log_val, y_log_test = y_log[train_idx], y_log[val_idx], y_log[test_idx]
    y_raw_test = y_raw[test_idx]
    y_pres_train, y_pres_val, y_pres_test = y_presence[train_idx], y_presence[val_idx], y_presence[test_idx]

    presence_alphas = parse_alpha_grid(args.presence_alpha_grid)
    positive_alphas = parse_alpha_grid(args.positive_alpha_grid)
    rng = np.random.default_rng(args.seed)
    rows = []
    y_pred_expected = np.zeros_like(y_raw_test)
    y_pred_presence = np.zeros_like(y_pres_test)
    y_pred_positive_log = np.zeros_like(y_log_test)

    for i, label in enumerate(labels):
        x_train_presence, y_train_presence = augment_training_rows(
            x_train,
            y_pres_train[:, i],
            args.rare_positive_oversample,
            args.rare_positive_threshold,
            args.rare_positive_max_multiplier,
            args.feature_mask_rate,
            rng,
        )
        alpha_presence = choose_alpha(
            x_train_presence,
            y_train_presence,
            x_val,
            y_pres_val[:, i],
            presence_alphas,
            clip_min=0.0,
            clip_max=1.0,
        )
        x_fit_presence, y_fit_presence = augment_training_rows(
            x[fit_idx],
            y_presence[fit_idx, i],
            args.rare_positive_oversample,
            args.rare_positive_threshold,
            args.rare_positive_max_multiplier,
            args.feature_mask_rate,
            rng,
        )
        coef_presence, intercept_presence = fit_ridge(x_fit_presence, y_fit_presence, alpha_presence)
        pred_presence = np.clip(predict_ridge(x_test, coef_presence, intercept_presence), a_min=0.0, a_max=1.0)
        y_pred_presence[:, i] = pred_presence

        pos_train_mask = y_pres_train[:, i] > 0
        pos_val_mask = y_pres_val[:, i] > 0
        if pos_train_mask.sum() >= 2 and pos_val_mask.sum() >= 1:
            x_train_positive, y_train_positive = augment_training_rows(
                x_train[pos_train_mask],
                y_log_train[pos_train_mask, i],
                args.rare_positive_oversample,
                args.rare_positive_threshold,
                args.rare_positive_max_multiplier,
                args.feature_mask_rate,
                rng,
            )
            alpha_positive = choose_alpha(
                x_train_positive,
                y_train_positive,
                x_val[pos_val_mask],
                y_log_val[pos_val_mask, i],
                positive_alphas,
                clip_min=0.0,
                clip_max=None,
            )
            fit_mask = y_presence[fit_idx, i] > 0
            x_fit_pos, y_fit_pos = augment_training_rows(
                x[fit_idx][fit_mask],
                y_log[fit_idx, i][fit_mask],
                args.rare_positive_oversample,
                args.rare_positive_threshold,
                args.rare_positive_max_multiplier,
                args.feature_mask_rate,
                rng,
            )
            coef_positive, intercept_positive = fit_ridge(x_fit_pos, y_fit_pos, alpha_positive)
            pred_positive_log = np.clip(predict_ridge(x_test, coef_positive, intercept_positive), a_min=0.0, a_max=None)
        else:
            alpha_positive = float("nan")
            pred_positive_log = np.full(len(test_idx), float(y_log_train[pos_train_mask, i].mean()) if pos_train_mask.any() else 0.0)
            coef_positive = np.zeros(x.shape[1], dtype=np.float64)
            intercept_positive = float(pred_positive_log[0])

        y_pred_positive_log[:, i] = pred_positive_log
        expected_raw = pred_presence * np.expm1(pred_positive_log)
        y_pred_expected[:, i] = expected_raw

        balanced_acc, f1_presence = binary_metrics(y_pres_test[:, i], pred_presence)
        rows.append(
            {
                "species": label,
                "support_nonzero_test": int(y_pres_test[:, i].sum()),
                "alpha_presence": float(alpha_presence),
                "alpha_positive": float(alpha_positive) if not np.isnan(alpha_positive) else np.nan,
                "presence_balanced_acc": balanced_acc,
                "presence_f1": f1_presence,
                "rmse_raw": rmse(y_raw_test[:, i], expected_raw),
                "mae_raw": mae(y_raw_test[:, i], expected_raw),
                "r2_raw": r2(y_raw_test[:, i], expected_raw),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_csv = out_dir / "species_hurdle_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    summary = {
        "model_name": "hurdle_ridge_numpy",
        "n_species": len(labels),
        "n_features": len(feature_names),
        "presence_alpha_grid": args.presence_alpha_grid,
        "positive_alpha_grid": args.positive_alpha_grid,
        "mean_rmse_raw": float(metrics["rmse_raw"].mean()),
        "mean_mae_raw": float(metrics["mae_raw"].mean()),
        "mean_r2_raw": float(metrics["r2_raw"].dropna().mean()),
        "mean_presence_balanced_acc": float(metrics["presence_balanced_acc"].mean()),
        "rare_positive_oversample": bool(args.rare_positive_oversample),
        "rare_positive_threshold": int(args.rare_positive_threshold),
        "rare_positive_max_multiplier": int(args.rare_positive_max_multiplier),
        "feature_mask_rate": float(args.feature_mask_rate),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_metric(metrics, "rmse_raw", "Per-Species Hurdle RMSE (Raw Count Scale)", out_dir / "species_rmse_raw.png", "#c65d2e")
    plot_metric(metrics, "mae_raw", "Per-Species Hurdle MAE (Raw Count Scale)", out_dir / "species_mae_raw.png", "#d98c2b")
    plot_metric(metrics, "r2_raw", "Per-Species Hurdle R^2 (Raw Count Scale)", out_dir / "species_r2_raw.png", "#1f6f8b")
    plot_metric(metrics, "presence_balanced_acc", "Per-Species Presence Balanced Accuracy", out_dir / "species_presence_balanced_acc.png", "#567d46")
    plot_metric(metrics, "support_nonzero_test", "Nonzero Test Samples by Species", out_dir / "species_support_nonzero.png", "#7a5c61")
    observed_vs_predicted_grid(labels, y_raw_test, y_pred_expected, out_dir / "observed_vs_predicted_grid.png")

    np.save(out_dir / "y_test_raw.npy", y_raw_test)
    np.save(out_dir / "y_pred_expected_raw.npy", y_pred_expected)
    np.save(out_dir / "y_pred_presence.npy", y_pred_presence)
    np.save(out_dir / "y_pred_positive_log.npy", y_pred_positive_log)

    print(metrics_csv)
    print(out_dir / "summary.json")
    print(out_dir / "species_rmse_raw.png")
    print(out_dir / "species_mae_raw.png")
    print(out_dir / "species_r2_raw.png")
    print(out_dir / "species_presence_balanced_acc.png")
    print(out_dir / "species_support_nonzero.png")
    print(out_dir / "observed_vs_predicted_grid.png")


if __name__ == "__main__":
    main()
