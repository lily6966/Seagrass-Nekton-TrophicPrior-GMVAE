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
OUT_PDF = Path(__file__).resolve().parent / "pdp_tt_sf_individual_features_panels_by_species.pdf"

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

COLORS = {
    "tt_stationsp_shoots_m": "#1f77b4",
    "sf_stationsp_shoots_m": "#9ecae1",
    "tt_station_above_wt_m": "#2ca02c",
    "sf_station_above_wt_m": "#98df8a",
    "tt_station_below_wt_m": "#d62728",
    "sf_station_below_wt_m": "#ff9896",
    "station_tt_can": "#9467bd",
    "station_sf_can": "#c5b0d5",
    "tt_stationsp_leaf_width": "#ff7f0e",
    "sf_stationsp_leaf_width": "#ffbb78",
    "tt_stationsp_lps": "#8c564b",
    "sf_stationsp_lps": "#c49c94",
}


def main() -> None:
    df = pd.read_csv(CURVES_CSV)
    df = df[df["feature_name"].isin(FEATURES)].copy()

    fig, axes = plt.subplots(
        len(SPECIES_ORDER),
        len(FEATURES),
        figsize=(28, 18),
        squeeze=False,
    )

    for row_idx, species in enumerate(SPECIES_ORDER):
        species_df = df[df["target_common_name"] == species]
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["feature_name"] == feature].sort_values("feature_value_original")
            ax.plot(
                sub["feature_value_original"],
                sub["mean_predicted_probability"],
                linewidth=1.8,
                color=COLORS[feature],
            )
            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=8)
            if row_idx == 0:
                ax.set_title(FEATURE_LABELS[feature], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(species, fontsize=9)
            else:
                ax.set_ylabel("")

    fig.suptitle("TT and SF Individual-Feature PDPs in Original Units", y=0.995, fontsize=16)
    fig.text(0.02, 0.5, "Mean predicted probability", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(OUT_PDF)


if __name__ == "__main__":
    main()
