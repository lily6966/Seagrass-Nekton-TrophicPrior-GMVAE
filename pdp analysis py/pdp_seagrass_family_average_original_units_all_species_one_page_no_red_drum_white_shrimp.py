from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys


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
OUT_CSV = DATA_DIR / "pdp_seagrass_family_average_original_units_all_species_one_page_no_red_drum_white_shrimp.csv"
OUT_PDF = DATA_DIR / "pdp_seagrass_family_average_original_units_all_species_one_page_no_red_drum_white_shrimp.pdf"

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

FAMILY_FEATURES = {
    "Shoots": [
        "tt_stationsp_shoots_m",
        "hd_stationsp_shoots_m",
        "he_stationsp_shoots_m",
        "hw_stationsp_shoots_m",
        "rm_stationsp_shoots_m",
        "sf_stationsp_shoots_m",
    ],
    "Above biomass": [
        "tt_station_above_wt_m",
        "hd_station_above_wt_m",
        "he_station_above_wt_m",
        "hw_station_above_wt_m",
        "rm_station_above_wt_m",
        "sf_station_above_wt_m",
    ],
    "Below biomass": [
        "tt_station_below_wt_m",
        "hd_station_below_wt_m",
        "he_station_below_wt_m",
        "hw_station_below_wt_m",
        "rm_station_below_wt_m",
        "sf_station_below_wt_m",
    ],
    "Canopy": [
        "station_tt_can",
        "station_hw_can",
        "station_sf_can",
        "station_he_can",
        "station_hd_can",
        "station_rm_can",
        "station_sg_can",
    ],
    "Leaf width": [
        "tt_stationsp_leaf_width",
        "hd_stationsp_leaf_width",
        "he_stationsp_leaf_width",
        "hw_stationsp_leaf_width",
        "rm_stationsp_leaf_width",
        "sf_stationsp_leaf_width",
    ],
    "Leaves per shoot": [
        "tt_stationsp_lps",
        "hd_stationsp_lps",
        "he_stationsp_lps",
        "hw_stationsp_lps",
        "rm_stationsp_lps",
        "sf_stationsp_lps",
    ],
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


def main() -> None:
    feature_columns = pd.read_json(FEATURE_COLUMNS_PATH, typ="series").tolist()
    label_columns = pd.read_json(LABEL_COLUMNS_PATH, typ="series").tolist()
    data = np.load(DATA_PATH)
    index = np.load(INDEX_PATH)
    labels = get_label(data, index, 0, len(label_columns)).astype(np.float32)
    base_features = get_feat(data, index, 0, len(label_columns), len(feature_columns)).astype(np.float32)
    model = load_model(len(feature_columns), len(label_columns))

    means, stds = infer_full_species_scaler()
    feature_to_idx = {name: i for i, name in enumerate(feature_columns)}
    original_values = {
        name: (base_features[:, feature_to_idx[name]] * stds[name]) + means[name]
        for name in feature_columns
        if name in means and name in stds
    }

    records = []
    for family_name, member_features in FAMILY_FEATURES.items():
        member_indices = [feature_to_idx[f] for f in member_features]
        member_original = np.column_stack([original_values[f] for f in member_features])
        family_average = member_original.mean(axis=1)
        grid = np.linspace(np.quantile(family_average, 0.05), np.quantile(family_average, 0.95), 21)

        composition = member_original.mean(axis=0)
        if composition.sum() <= 0:
            composition = np.ones_like(composition)
        composition = composition / composition.sum()

        n_members = member_original.shape[1]
        for grid_value in grid:
            modified = base_features.copy()
            current_average = member_original.mean(axis=1, keepdims=True)
            scaled_original = np.zeros_like(member_original)
            positive_mask = current_average[:, 0] > 1e-8
            scaled_original[positive_mask] = member_original[positive_mask] * (
                grid_value / current_average[positive_mask]
            )
            scaled_original[~positive_mask] = composition * (grid_value * n_members)

            for local_idx, feature_name in enumerate(member_features):
                standardized = (scaled_original[:, local_idx] - means[feature_name]) / stds[feature_name]
                modified[:, member_indices[local_idx]] = standardized.astype(np.float32)

            probs = predict_probabilities(model, labels, modified)
            mean_probs = probs.mean(axis=0)
            for label_name, mean_prob in zip(label_columns, mean_probs):
                if label_name not in COMMON_NAMES:
                    continue
                records.append(
                    {
                        "family_name": family_name,
                        "family_average_original": float(grid_value),
                        "target_label": label_name,
                        "target_common_name": COMMON_NAMES[label_name],
                        "mean_predicted_probability": float(mean_prob),
                    }
                )

    curves = pd.DataFrame.from_records(records)
    curves.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(
        len(SPECIES_ORDER),
        len(FAMILY_FEATURES),
        figsize=(16, 18),
        squeeze=False,
    )
    for row_idx, species in enumerate(SPECIES_ORDER):
        species_df = curves[curves["target_common_name"] == species]
        for col_idx, family_name in enumerate(FAMILY_FEATURES):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["family_name"] == family_name].sort_values("family_average_original")
            ax.plot(
                sub["family_average_original"],
                sub["mean_predicted_probability"],
                color="#1f77b4",
                linewidth=1.8,
            )
            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=8)
            if row_idx == 0:
                ax.set_title(family_name, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(species, fontsize=9)
            else:
                ax.set_ylabel("")

    fig.suptitle(
        "Average Seagrass-Family PDPs in Original Units",
        y=0.995,
        fontsize=16,
    )
    fig.text(0.02, 0.5, "Mean predicted probability", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(OUT_CSV)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
