import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import VAE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clean_label(label: str) -> str:
    return label.replace("niche_label__", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CV label-level metrics.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--fold-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument("--latent-dim", default=32, type=int)
    parser.add_argument("--z-dim", default=32, type=int)
    parser.add_argument("--emb-size", default=256, type=int)
    parser.add_argument("--keep-prob", default=0.5, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


def build_model(args: argparse.Namespace, checkpoint: Path) -> VAE:
    model_args = SimpleNamespace(
        feature_dim=args.feature_dim,
        label_dim=args.label_dim,
        latent_dim=args.latent_dim,
        z_dim=args.z_dim,
        emb_size=args.emb_size,
        keep_prob=args.keep_prob,
    )
    model = VAE(model_args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def predict(model: VAE, x: np.ndarray, label_dim: int, batch_size: int = 64) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        dummy_label = torch.zeros((batch.shape[0], label_dim), device=device)
        dummy_label[:, 0] = 1.0
        with torch.no_grad():
            output = model(dummy_label, batch)
            prob = torch.sigmoid(output["feat_out"]).cpu().numpy()
        preds.append(prob)
    return np.concatenate(preds, axis=0)


def compute_fold_label_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    threshold: float,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    y_pred = (y_score >= threshold).astype(int)
    support = y_true.sum(axis=0)
    rows = []
    for i, label in enumerate(labels):
        auc = np.nan
        if np.unique(y_true[:, i]).size > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
            except Exception:
                pass
        aupr = np.nan
        try:
            aupr = average_precision_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
        f1 = np.nan
        if y_true[:, i].sum() > 0:
            try:
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            except Exception:
                pass
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "label": clean_label(label),
                "support": int(support[i]),
                "roc_auc": auc,
                "aupr": aupr,
                "f1_at_0_5": f1,
            }
        )
    return pd.DataFrame(rows)


def barplot(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    plot_df = df.sort_values(value_col, ascending=False).copy()
    fig_h = max(6, 0.35 * len(plot_df))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["label"], plot_df[value_col], color="#1f6f8b")
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def support_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.sort_values("support_mean", ascending=False).copy()
    fig_h = max(6, 0.35 * len(plot_df))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["label"], plot_df["support_mean"], color="#d98c2b")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Positive Test Samples Across Folds")
    ax.set_title("CV Mean Test Support by Label")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    labels = json.loads(Path(args.labels_json).read_text())
    fold_df = pd.read_csv(args.fold_metrics)

    all_rows = []
    for _, row in fold_df.iterrows():
        checkpoint = ROOT / row["best_checkpoint"]
        split_root = Path(args.fold_metrics).parent / "splits" / f"repeat_{int(row['repeat'])}" / f"fold_{int(row['fold'])}"
        test_idx = np.load(split_root / "test_idx.npy")
        y_true = data[test_idx, : args.label_dim]
        x = data[test_idx, args.label_dim :]
        model = build_model(args, checkpoint)
        y_score = predict(model, x, args.label_dim)
        all_rows.append(
            compute_fold_label_metrics(
                y_true,
                y_score,
                labels,
                args.threshold,
                int(row["repeat"]),
                int(row["fold"]),
            )
        )

    per_fold = pd.concat(all_rows, ignore_index=True)
    per_fold.to_csv(out_dir / "new_cv_label_metrics_by_fold.csv", index=False)

    summary = (
        per_fold.groupby("label")
        .agg(
            support_mean=("support", "mean"),
            support_min=("support", "min"),
            support_max=("support", "max"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_std=("aupr", "std"),
            f1_at_0_5_mean=("f1_at_0_5", "mean"),
            f1_at_0_5_std=("f1_at_0_5", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "new_cv_label_metrics_summary.csv", index=False)

    barplot(summary.fillna(0.0), "roc_auc_mean", "CV Mean ROC AUC by Label", out_dir / "new_cv_species_roc_auc.png")
    barplot(summary.fillna(0.0), "aupr_mean", "CV Mean AUPR by Label", out_dir / "new_cv_species_aupr.png")
    barplot(summary.fillna(0.0), "f1_at_0_5_mean", "CV Mean F1 at 0.5 by Label", out_dir / "new_cv_species_f1_at_0_5.png")
    support_plot(summary.fillna(0.0), out_dir / "new_cv_species_support.png")

    print(out_dir / "new_cv_label_metrics_by_fold.csv")
    print(out_dir / "new_cv_label_metrics_summary.csv")


if __name__ == "__main__":
    main()
