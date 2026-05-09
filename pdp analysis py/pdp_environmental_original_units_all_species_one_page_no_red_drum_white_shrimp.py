from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CURVES_CSV = (
    ROOT
    / "output"
    / "partial_dependence"
    / "vae660_environmental_only_original_scale"
    / "partial_dependence_curves_no_red_drum_white_shrimp.csv"
)
OUT_PDF = (
    Path(__file__).resolve().parent
    / "pdp_environmental_original_units_all_species_one_page_no_red_drum_white_shrimp.pdf"
)

SPECIES_ORDER = [
    "Brown shrimp",
    "Pink shrimp",
    "Blue crab",
    "Spotted seatrout",
    "Gray snapper",
    "Lane snapper",
]

FEATURES = ["salinity", "do", "temp", "seagrass_richness", "secchi"]

FEATURE_LABELS = {
    "salinity": "Salinity (ppt)",
    "do": "Dissolved oxygen (mg/L)",
    "temp": "Temperature (C)",
    "seagrass_richness": "Seagrass richness (count)",
    "secchi": "Secchi depth (cm)",
}


def main() -> None:
    df = pd.read_csv(CURVES_CSV)

    fig, axes = plt.subplots(
        len(SPECIES_ORDER),
        len(FEATURES),
        figsize=(16, 18),
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
                ax.set_ylabel(
                    f"{species}",
                    fontsize=9,
                )
            else:
                ax.set_ylabel("")

    fig.suptitle(
        "Environmental Partial Dependence in Original Units",
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
