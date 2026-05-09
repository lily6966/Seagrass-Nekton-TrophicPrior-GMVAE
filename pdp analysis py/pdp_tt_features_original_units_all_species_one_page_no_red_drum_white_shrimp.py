from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CURVES_CSV = (
    ROOT
    / "output"
    / "partial_dependence"
    / "vae660_requested_predictor_groups_original_scale"
    / "partial_dependence_curves_no_red_drum_white_shrimp.csv"
)
OUT_PDF = (
    Path(__file__).resolve().parent
    / "pdp_tt_features_original_units_all_species_one_page_no_red_drum_white_shrimp.pdf"
)

SPECIES_ORDER = [
    "Brown shrimp",
    "Pink shrimp",
    "Blue crab",
    "Spotted seatrout",
    "Gray snapper",
    "Lane snapper",
]

FEATURES = [
    "tt_stationsp_shoots_m",
    "tt_station_above_wt_m",
    "tt_station_below_wt_m",
    "station_tt_can",
    "tt_stationsp_leaf_width",
    "tt_stationsp_lps",
]

FEATURE_LABELS = {
    "tt_stationsp_shoots_m": "TT shoots",
    "tt_station_above_wt_m": "TT above biomass",
    "tt_station_below_wt_m": "TT below biomass",
    "station_tt_can": "TT canopy",
    "tt_stationsp_leaf_width": "TT leaf width",
    "tt_stationsp_lps": "TT leaves per shoot",
}


def main() -> None:
    df = pd.read_csv(CURVES_CSV)
    df = df[df["feature_name"].isin(FEATURES)].copy()

    fig, axes = plt.subplots(
        len(SPECIES_ORDER),
        len(FEATURES),
        figsize=(18, 18),
        squeeze=False,
    )

    for row_idx, species in enumerate(SPECIES_ORDER):
        species_df = df[df["target_common_name"] == species]
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["feature_name"] == feature].sort_values(
                "feature_value_original"
            )
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
        "TT Seagrass-Feature PDPs in Original Units",
        y=0.995,
        fontsize=16,
    )
    fig.text(0.02, 0.5, "Mean predicted probability", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(OUT_PDF)


if __name__ == "__main__":
    main()
