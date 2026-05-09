import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model import VAE
from utils import get_feat, get_label
from tools.process_nekton_seagrass import (
    load_frame,
    build_sample_table,
    parse_dates,
    choose_species_label,
    SLED_CSV,
    TRAWL_CSV,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMMON_NAMES = {
    "Penaeus aztecus": "Brown shrimp",
    "Penaeus duorarum": "Pink shrimp",
    "Penaeus setiferus": "White shrimp",
    "Callinectes sapidus": "Blue crab",
    "Sciaenops ocellatus": "Red drum",
    "Cynoscion nebulosus": "Spotted seatrout",
    "Lutjanus griseus": "Gray snapper",
    "Lutjanus synagris": "Lane snapper",
}

DEFAULT_FEATURES = [
    "light_attenuation_av",
    "depth",
    "secchi",
    "temp",
    "salinity",
    "do",
    "macro_weight",
    "seagrass_richness",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run 1D partial dependence for a saved GMVAE checkpoint.")
    parser.add_argument("--data", required=True, help="Path to *_data.npy file")
    parser.add_argument("--index", required=True, help="Path to index file used for PDP background samples")
    parser.add_argument("--feature-columns", required=True, help="Path to feature_columns JSON")
    parser.add_argument("--label-columns", required=True, help="Path to label_columns JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for CSVs and figures")
    parser.add_argument("--label-dim", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--emb-size", type=int, default=512)
    parser.add_argument("--keep-prob", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=21)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--original-scale", action="store_true", help="Convert PDP x-axis from standardized input scale back to original feature units.")
    return parser.parse_args()


def load_model(args):
    model_args = argparse.Namespace(
        label_dim=args.label_dim,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        z_dim=args.z_dim,
        emb_size=args.emb_size,
        keep_prob=args.keep_prob,
        reg="gmvae",
        gear_residualize=False,
        gear_interaction_design=False,
        gear_feature_idx=None,
        meta_offset=0,
    )
    model = VAE(model_args).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    return model


def predict_probabilities(model, labels, features, batch_size):
    outputs = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        feat_batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        label_batch = torch.from_numpy(labels[start:end]).float().to(DEVICE)
        dummy_label = torch.zeros(label_batch.shape[1], device=DEVICE)
        dummy_label[0] = 1.0
        mask = label_batch.sum(dim=1) > 0
        if not mask.all():
            label_batch = label_batch.clone()
            label_batch[~mask] = dummy_label
        with torch.no_grad():
            prob = torch.sigmoid(model(label_batch, feat_batch)["feat_out"]).cpu().numpy()
        outputs.append(prob)
    return np.concatenate(outputs, axis=0)


def sanitize_filename(name):
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
        .replace("(", "")
        .replace(")", "")
    )


def infer_full_species_scaler():
    sled_raw = load_frame(SLED_CSV)
    trawl_raw = load_frame(TRAWL_CSV)
    positive_sled = pd.to_numeric(sled_raw["species_abundance"], errors="coerce").fillna(0.0) > 0
    positive_trawl = pd.to_numeric(trawl_raw["species_abundance"], errors="coerce").fillna(0.0) > 0
    all_label_cols = sorted(
        set(choose_species_label(sled_raw[positive_sled]).tolist())
        | set(choose_species_label(trawl_raw[positive_trawl]).tolist())
    )
    sample_table = pd.concat(
        [
            build_sample_table(sled_raw, "pull_id", 1),
            build_sample_table(trawl_raw, "tow_id", 0),
        ],
        ignore_index=True,
        sort=False,
    )
    feature_df = sample_table.drop(columns=["sample_id"] + all_label_cols).copy()
    feature_df = parse_dates(feature_df)
    categorical_cols = ["site", "station", "sample_period", "substrate"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    numeric_part = feature_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    medians = numeric_part.median().fillna(0.0)
    numeric_part = numeric_part.fillna(medians)
    means = numeric_part.mean()
    stds = numeric_part.std(ddof=0).replace(0, 1.0)
    return means.to_dict(), stds.to_dict()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    index = np.load(args.index)
    feature_columns = json.loads(Path(args.feature_columns).read_text())
    label_columns = json.loads(Path(args.label_columns).read_text())

    feature_name_to_idx = {name: i for i, name in enumerate(feature_columns)}
    selected_features = [f for f in args.features if f in feature_name_to_idx]
    missing_features = [f for f in args.features if f not in feature_name_to_idx]
    if not selected_features:
        raise ValueError("None of the requested features are present in the feature matrix.")

    model = load_model(args)
    labels = get_label(data, index, 0, args.label_dim).astype(np.float32)
    base_features = get_feat(data, index, 0, args.label_dim, args.feature_dim).astype(np.float32)
    original_means = {}
    original_stds = {}
    if args.original_scale:
        original_means, original_stds = infer_full_species_scaler()

    records = []
    summary = []
    for feature_name in selected_features:
        feat_idx = feature_name_to_idx[feature_name]
        observed = base_features[:, feat_idx]
        lo = float(np.quantile(observed, args.quantile_low))
        hi = float(np.quantile(observed, args.quantile_high))
        if np.isclose(lo, hi):
            grid = np.array([lo], dtype=np.float32)
        else:
            grid = np.linspace(lo, hi, args.grid_points, dtype=np.float32)

        pdp_values = []
        for grid_value in grid:
            modified = base_features.copy()
            modified[:, feat_idx] = grid_value
            probs = predict_probabilities(model, labels, modified, args.batch_size)
            mean_probs = probs.mean(axis=0)
            pdp_values.append(mean_probs)
            original_value = float(grid_value)
            if args.original_scale and feature_name in original_means and feature_name in original_stds:
                original_value = float((grid_value * original_stds[feature_name]) + original_means[feature_name])
            for label_name, mean_prob in zip(label_columns, mean_probs):
                records.append(
                    {
                        "feature_name": feature_name,
                        "feature_value_zscore": float(grid_value),
                        "feature_value_original": original_value,
                        "target_label": label_name,
                        "target_common_name": COMMON_NAMES.get(label_name, label_name),
                        "mean_predicted_probability": float(mean_prob),
                    }
                )

        pdp_values = np.vstack(pdp_values)
        effect_size = pdp_values.max(axis=0) - pdp_values.min(axis=0)
        for label_name, effect in zip(label_columns, effect_size):
            summary.append(
                {
                    "feature_name": feature_name,
                    "target_label": label_name,
                    "target_common_name": COMMON_NAMES.get(label_name, label_name),
                    "feature_range_original_low": float((grid.min() * original_stds[feature_name]) + original_means[feature_name]) if args.original_scale and feature_name in original_means and feature_name in original_stds else float(grid.min()),
                    "feature_range_original_high": float((grid.max() * original_stds[feature_name]) + original_means[feature_name]) if args.original_scale and feature_name in original_means and feature_name in original_stds else float(grid.max()),
                    "effect_size_probability_range": float(effect),
                }
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_x = grid
        xlabel = f"{feature_name} (standardized input)"
        if args.original_scale and feature_name in original_means and feature_name in original_stds:
            plot_x = (grid * original_stds[feature_name]) + original_means[feature_name]
            xlabel = f"{feature_name} (original units)"
        for label_idx, label_name in enumerate(label_columns):
            ax.plot(
                plot_x,
                pdp_values[:, label_idx],
                linewidth=2,
                label=COMMON_NAMES.get(label_name, label_name),
            )
        ax.set_title(f"Partial Dependence: {feature_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean predicted probability")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"pdp_{sanitize_filename(feature_name)}.png", dpi=200)
        plt.close(fig)

    detail_df = pd.DataFrame.from_records(records)
    summary_df = (
        pd.DataFrame.from_records(summary)
        .sort_values(["feature_name", "effect_size_probability_range"], ascending=[True, False])
        .reset_index(drop=True)
    )
    detail_df.to_csv(out_dir / "partial_dependence_curves.csv", index=False)
    summary_df.to_csv(out_dir / "partial_dependence_effect_sizes.csv", index=False)

    ncols = 2
    nrows = int(np.ceil(len(selected_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax_idx, feature_name in enumerate(selected_features):
        ax = axes[ax_idx]
        subset = detail_df[detail_df["feature_name"] == feature_name]
        x_col = "feature_value_original" if args.original_scale else "feature_value_zscore"
        for label_name in label_columns:
            label_subset = subset[subset["target_label"] == label_name]
            ax.plot(
                label_subset[x_col],
                label_subset["mean_predicted_probability"],
                linewidth=1.7,
                label=COMMON_NAMES.get(label_name, label_name),
            )
        ax.set_title(feature_name)
        ax.set_xlabel("Original feature value" if args.original_scale else "Standardized feature value")
        ax.set_ylabel("Mean predicted probability")
        ax.grid(alpha=0.25)
    for ax in axes[len(selected_features):]:
        ax.axis("off")
    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_for_legend, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Partial Dependence Panels", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_dir / "partial_dependence_panel.png", dpi=200)
    plt.close(fig)

    metadata = {
        "data": str(Path(args.data).resolve()),
        "index": str(Path(args.index).resolve()),
        "feature_columns": str(Path(args.feature_columns).resolve()),
        "label_columns": str(Path(args.label_columns).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_features": selected_features,
        "missing_requested_features": missing_features,
        "grid_points": args.grid_points,
        "quantile_low": args.quantile_low,
        "quantile_high": args.quantile_high,
        "original_scale": bool(args.original_scale),
        "note": "Numeric features are standardized in this dataset. When original_scale=true, feature_value_original is reconstructed from the raw mixed-gear full-species preprocessing means and standard deviations.",
    }
    (out_dir / "partial_dependence_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
