from argparse import Namespace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from model import VAE
from utils import get_feat, get_label
from tools.process_nekton_seagrass import (
    SLED_CSV,
    TRAWL_CSV,
    build_sample_table,
    choose_species_label,
    load_frame,
    parse_dates,
    make_splits,
)


DATA_DIR = ROOT / "data" / "nekton-seagrass"
TRANSFER_DIR = DATA_DIR / "transfer_learning"
DATA_PATH = TRANSFER_DIR / "nekton_seagrass_finetune_fisheries8_nogear_seagrass_data.npy"
INDEX_PATH = TRANSFER_DIR / "nekton_seagrass_finetune_fisheries8_nogear_seagrass_test_idx.npy"
FEATURE_COLUMNS_PATH = TRANSFER_DIR / "feature_columns_finetune_fisheries8_nogear_seagrass.json"
LABEL_COLUMNS_PATH = TRANSFER_DIR / "label_columns_finetune_fisheries8_seagrass.json"
CHECKPOINT_PATH = (
    ROOT
    / "model"
    / "model_transfer_learning_finetune_fisheries8_fullgroupconsistency_nogear_matchvae261_v1"
    / "lr-0.0005_lr-decay_0.50_lr-times_3.0_nll-0.50_l2-1.00_c-0.00"
    / "vae-660"
)
OUT_DIR = DATA_DIR / "pdp_environmental_by_site_no_red_drum_white_shrimp"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMMON_NAMES = {
    "Penaeus aztecus": "Brown shrimp",
    "Penaeus duorarum": "Pink shrimp",
    "Callinectes sapidus": "Blue crab",
    "Cynoscion nebulosus": "Spotted seatrout",
    "Lutjanus griseus": "Gray snapper",
    "Lutjanus synagris": "Lane snapper",
}

SPECIES_ORDER = [
    "Brown shrimp",
    "Pink shrimp",
    "Blue crab",
    "Spotted seatrout",
    "Gray snapper",
    "Lane snapper",
]

SITE_ORDER = ["AP", "CB", "CH", "CK", "LA", "LM"]

FEATURES = ["salinity", "do", "temp", "seagrass_richness", "secchi"]

FEATURE_LABELS = {
    "salinity": "Salinity (ppt)",
    "do": "Dissolved oxygen (mg/L)",
    "temp": "Temperature (C)",
    "seagrass_richness": "Seagrass richness (count)",
    "secchi": "Secchi depth (cm)",
}


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


def build_test_site_series() -> pd.Series:
    sled = load_frame(SLED_CSV)
    trawl = load_frame(TRAWL_CSV)
    combined = pd.concat(
        [
            build_sample_table(sled, "pull_id", 1),
            build_sample_table(trawl, "tow_id", 0),
        ],
        ignore_index=True,
        sort=False,
    )
    _, _, test_idx = make_splits(len(combined))
    return combined.iloc[test_idx]["site"].reset_index(drop=True)


def load_model(feature_dim: int, label_dim: int) -> VAE:
    args = Namespace(
        label_dim=label_dim,
        feature_dim=feature_dim,
        latent_dim=64,
        z_dim=64,
        emb_size=512,
        keep_prob=0.5,
        reg="gmvae",
        gear_residualize=False,
        gear_interaction_design=False,
        gear_feature_idx=None,
        meta_offset=0,
    )
    model = VAE(args).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict_probabilities(model: VAE, labels: np.ndarray, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        feat_batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        label_batch = torch.from_numpy(labels[start:end]).float().to(DEVICE)
        dummy = torch.zeros(label_batch.shape[1], device=DEVICE)
        dummy[0] = 1.0
        mask = label_batch.sum(dim=1) > 0
        if not mask.all():
            label_batch = label_batch.clone()
            label_batch[~mask] = dummy
        with torch.no_grad():
            prob = torch.sigmoid(model(label_batch, feat_batch)["feat_out"]).cpu().numpy()
        outputs.append(prob)
    return np.concatenate(outputs, axis=0)


def compute_site_curves(
    site: str,
    labels: np.ndarray,
    base_features: np.ndarray,
    feature_columns: list[str],
    label_columns: list[str],
    model: VAE,
    means: dict[str, float],
    stds: dict[str, float],
) -> pd.DataFrame:
    feature_name_to_idx = {name: i for i, name in enumerate(feature_columns)}
    records = []
    for feature_name in FEATURES:
        feat_idx = feature_name_to_idx[feature_name]
        observed = base_features[:, feat_idx]
        lo = float(np.quantile(observed, 0.05))
        hi = float(np.quantile(observed, 0.95))
        if np.isclose(lo, hi):
            grid = np.array([lo], dtype=np.float32)
        else:
            grid = np.linspace(lo, hi, 21, dtype=np.float32)

        for grid_value in grid:
            modified = base_features.copy()
            modified[:, feat_idx] = grid_value
            probs = predict_probabilities(model, labels, modified)
            mean_probs = probs.mean(axis=0)
            original_value = float((grid_value * stds[feature_name]) + means[feature_name])
            for label_name, mean_prob in zip(label_columns, mean_probs):
                if label_name not in COMMON_NAMES:
                    continue
                records.append(
                    {
                        "site": site,
                        "feature_name": feature_name,
                        "feature_value_zscore": float(grid_value),
                        "feature_value_original": original_value,
                        "target_label": label_name,
                        "target_common_name": COMMON_NAMES[label_name],
                        "mean_predicted_probability": float(mean_prob),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_site_curves(site: str, curves: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(
        len(SPECIES_ORDER),
        len(FEATURES),
        figsize=(16, 18),
        squeeze=False,
    )
    for row_idx, species in enumerate(SPECIES_ORDER):
        species_df = curves[curves["target_common_name"] == species]
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["feature_name"] == feature].sort_values("feature_value_original")
            ax.plot(
                sub["feature_value_original"],
                sub["mean_predicted_probability"],
                color="#1f77b4",
                linewidth=1.8,
            )
            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=8)
            if row_idx == 0:
                ax.set_title(FEATURE_LABELS[feature], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(species, fontsize=9)
            else:
                ax.set_ylabel("")

    fig.suptitle(
        f"Environmental Partial Dependence in Original Units ({site})",
        y=0.995,
        fontsize=16,
    )
    fig.text(0.02, 0.5, "Mean predicted probability", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = pd.read_json(FEATURE_COLUMNS_PATH, typ="series").tolist()
    label_columns = pd.read_json(LABEL_COLUMNS_PATH, typ="series").tolist()
    data = np.load(DATA_PATH)
    index = np.load(INDEX_PATH)
    labels = get_label(data, index, 0, len(label_columns)).astype(np.float32)
    base_features = get_feat(data, index, 0, len(label_columns), len(feature_columns)).astype(np.float32)
    site_series = build_test_site_series()

    means, stds = infer_full_species_scaler()
    model = load_model(len(feature_columns), len(label_columns))

    written = []
    all_curves = []
    for site in SITE_ORDER:
        mask = site_series.to_numpy() == site
        if not mask.any():
            continue
        site_labels = labels[mask]
        site_features = base_features[mask]
        curves = compute_site_curves(
            site,
            site_labels,
            site_features,
            feature_columns,
            label_columns,
            model,
            means,
            stds,
        )
        out_csv = OUT_DIR / f"pdp_environmental_original_units_{site}_no_red_drum_white_shrimp.csv"
        out_pdf = OUT_DIR / f"pdp_environmental_original_units_{site}_no_red_drum_white_shrimp.pdf"
        curves.to_csv(out_csv, index=False)
        plot_site_curves(site, curves, out_pdf)
        written.extend([out_csv, out_pdf])
        all_curves.append(curves)

    if all_curves:
        combined_csv = OUT_DIR / "pdp_environmental_original_units_all_sites_split_no_red_drum_white_shrimp.csv"
        pd.concat(all_curves, ignore_index=True).to_csv(combined_csv, index=False)
        written.append(combined_csv)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
