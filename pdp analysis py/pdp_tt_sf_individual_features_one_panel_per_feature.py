from pathlib import Path
import math

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
OUT_PDF = Path(__file__).resolve().parent / "pdp_tt_sf_individual_features_one_panel_per_feature.pdf"

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
    "sf_stationsp_shoots_m",
    "tt_station_above_wt_m",
    "sf_station_above_wt_m",
    "tt_station_below_wt_m",
    "sf_station_below_wt_m",
    "station_tt_can",
    "station_sf_can",
    "tt_stationsp_leaf_width",
    "sf_stationsp_leaf_width",
    "tt_stationsp_lps",
    "sf_stationsp_lps",
]

FEATURE_LABELS = {
    "tt_stationsp_shoots_m": "TT shoots",
    "sf_stationsp_shoots_m": "SF shoots",
    "tt_station_above_wt_m": "TT above biomass",
    "sf_station_above_wt_m": "SF above biomass",
    "tt_station_below_wt_m": "TT below biomass",
    "sf_station_below_wt_m": "SF below biomass",
    "station_tt_can": "TT canopy",
    "station_sf_can": "SF canopy",
    "tt_stationsp_leaf_width": "TT leaf width",
    "sf_stationsp_leaf_width": "SF leaf width",
    "tt_stationsp_lps": "TT leaves per shoot",
    "sf_stationsp_lps": "SF leaves per shoot",
}


def main() -> None:
    df = pd.read_csv(CURVES_CSV)
    df = df[df["feature_name"].isin(FEATURES)].copy()

    ncols = 2
    nrows = math.ceil(len(FEATURES) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        subset = df[df["feature_name"] == feature]
        for species in SPECIES_ORDER:
            sub = subset[subset["target_common_name"] == species].sort_values("feature_value_original")
            ax.plot(
                sub["feature_value_original"],
                sub["mean_predicted_probability"],
                linewidth=2.0,
                label=species,
            )
        ax.set_title(FEATURE_LABELS[feature], fontsize=11)
        ax.set_xlabel("Original feature units", fontsize=10)
        ax.set_ylabel("Mean predicted probability", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes[len(FEATURES):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("TT and SF Individual-Feature PDPs by Feature", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(OUT_PDF)


if __name__ == "__main__":
    main()
