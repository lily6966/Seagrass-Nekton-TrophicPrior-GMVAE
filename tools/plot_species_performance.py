import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import VAE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMMON_NAME_MAP = {
    "Penaeus aztecus": "Brown shrimp",
    "Penaeus duorarum": "Pink shrimp",
    "Penaeus setiferus": "White shrimp",
    "Callinectes sapidus": "Blue crab",
    "Sciaenops ocellatus": "Red drum",
    "Cynoscion nebulosus": "Spotted seatrout",
    "Lutjanus griseus": "Gray snapper",
    "Lutjanus synagris": "Lane snapper",
}


def clean_label(label: str) -> str:
    cleaned = label.replace("niche_label__", "")
    return COMMON_NAME_MAP.get(cleaned, cleaned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-species model performance.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-idx", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument("--latent-dim", required=True, type=int)
    parser.add_argument("--z-dim", required=True, type=int)
    parser.add_argument("--emb-size", required=True, type=int)
    parser.add_argument("--keep-prob", default=0.5, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> VAE:
    model_args = SimpleNamespace(
        feature_dim=args.feature_dim,
        label_dim=args.label_dim,
        latent_dim=args.latent_dim,
        z_dim=args.z_dim,
        emb_size=args.emb_size,
        keep_prob=args.keep_prob,
    )
    model = VAE(model_args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
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


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, labels: list[str], threshold: float) -> pd.DataFrame:
    rows = []
    y_pred = (y_score >= threshold).astype(int)
    support = y_true.sum(axis=0)
    for i, label in enumerate(labels):
        auc = np.nan
        try:
            if np.unique(y_true[:, i]).size > 1:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
        aupr = np.nan
        try:
            aupr = average_precision_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
        f1 = np.nan
        try:
            if y_true[:, i].sum() > 0:
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        except Exception:
            pass
        rows.append(
            {
                "species": clean_label(label),
                "support": int(support[i]),
                "roc_auc": auc,
                "aupr": aupr,
                "f1_at_0.5": f1,
            }
        )
    return pd.DataFrame(rows)


def barplot(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    plot_df = df.sort_values(value_col, ascending=False).copy()
    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["species"], plot_df[value_col], color="#1f6f8b")
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def support_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.sort_values("support", ascending=False).copy()
    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["species"], plot_df["support"], color="#d98c2b")
    ax.invert_yaxis()
    ax.set_xlabel("Positive Test Samples")
    ax.set_title("Test Support by Species")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _curve_grid_shape(n_panels: int) -> tuple[int, int]:
    n_cols = 2 if n_panels <= 4 else 3
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def roc_curve_grid(y_true: np.ndarray, y_score: np.ndarray, labels: list[str], metrics: pd.DataFrame, path: Path) -> None:
    n_labels = len(labels)
    n_rows, n_cols = _curve_grid_shape(n_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    metric_lookup = metrics.set_index("species")
    for i, label in enumerate(labels):
        ax = axes[i]
        clean = clean_label(label)
        yt = y_true[:, i]
        ys = y_score[:, i]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        if np.unique(yt).size > 1:
            fpr, tpr, _ = roc_curve(yt, ys)
            ax.plot(fpr, tpr, color="#1f6f8b", linewidth=2)
        else:
            ax.text(0.5, 0.5, "ROC undefined\n(single class)", ha="center", va="center", fontsize=10)
        auc_val = metric_lookup.loc[clean, "roc_auc"]
        support = int(metric_lookup.loc[clean, "support"])
        auc_text = "NA" if pd.isna(auc_val) else f"{auc_val:.3f}"
        ax.set_title(f"{clean}\nAUC={auc_text}, n+={support}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax in axes[n_labels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pr_curve_grid(y_true: np.ndarray, y_score: np.ndarray, labels: list[str], metrics: pd.DataFrame, path: Path) -> None:
    n_labels = len(labels)
    n_rows, n_cols = _curve_grid_shape(n_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    metric_lookup = metrics.set_index("species")
    for i, label in enumerate(labels):
        ax = axes[i]
        clean = clean_label(label)
        yt = y_true[:, i]
        ys = y_score[:, i]
        positives = int(yt.sum())
        if positives > 0:
            precision, recall, _ = precision_recall_curve(yt, ys)
            ax.plot(recall, precision, color="#d98c2b", linewidth=2)
            baseline = positives / len(yt)
            ax.axhline(baseline, linestyle="--", color="#999999", linewidth=1)
        else:
            ax.text(0.5, 0.5, "PR undefined\n(no positives)", ha="center", va="center", fontsize=10)
        aupr_val = metric_lookup.loc[clean, "aupr"]
        aupr_text = "NA" if pd.isna(aupr_val) else f"{aupr_val:.3f}"
        ax.set_title(f"{clean}\nAUPR={aupr_text}, n+={positives}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax in axes[n_labels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(Path(args.labels_json).read_text())
    data = np.load(args.data)
    test_idx = np.load(args.test_idx)

    y_true = data[test_idx, : args.label_dim]
    x = data[test_idx, args.label_dim :]

    model = build_model(args)
    y_score = predict(model, x, args.label_dim)
    metrics = compute_metrics(y_true, y_score, labels, args.threshold)

    csv_path = out_dir / "species_metrics.csv"
    metrics.to_csv(csv_path, index=False)

    auc_path = out_dir / "species_roc_auc.png"
    aupr_path = out_dir / "species_aupr.png"
    f1_path = out_dir / "species_f1_at_0_5.png"
    support_path = out_dir / "species_support.png"
    roc_curve_path = out_dir / "species_roc_curves.png"
    pr_curve_path = out_dir / "species_pr_curves.png"

    barplot(metrics.fillna(0.0), "roc_auc", "Per-Species ROC AUC", auc_path)
    barplot(metrics.fillna(0.0), "aupr", "Per-Species AUPR", aupr_path)
    barplot(metrics.fillna(0.0), "f1_at_0.5", "Per-Species F1 at 0.5", f1_path)
    support_plot(metrics, support_path)
    roc_curve_grid(y_true, y_score, labels, metrics, roc_curve_path)
    pr_curve_grid(y_true, y_score, labels, metrics, pr_curve_path)

    print(csv_path)
    print(auc_path)
    print(aupr_path)
    print(f1_path)
    print(support_path)
    print(roc_curve_path)
    print(pr_curve_path)


if __name__ == "__main__":
    main()
