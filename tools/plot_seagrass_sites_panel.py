import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SITE_ORDER = ["AP", "CB", "CH", "CK", "LA", "LM"]


def clean_label(label: str) -> str:
    cleaned = label.replace("niche_label__", "")
    friendly = {
        "Penaeus aztecus": "Brown shrimp",
        "Penaeus duorarum": "Pink shrimp",
        "Penaeus setiferus": "White shrimp",
        "Callinectes sapidus": "Blue crab",
        "Sciaenops ocellatus": "Red drum",
        "Cynoscion nebulosus": "Spotted seatrout",
        "Lutjanus griseus": "Gray snapper",
        "Lutjanus synagris": "Lane snapper",
        "planktivore": "Planktivore",
        "benthic_invertivore": "Benthic invertivore",
        "piscivore": "Piscivore",
        "omnivore": "Omnivore",
        "herbivore_detritivore": "Herbivore/detritivore",
        "shrimp_crab_invertivore": "Shrimp/crab invertivore",
        "unknown": "Unknown",
    }
    return friendly.get(cleaned, cleaned)


def clean_row_label(value: str, row_col: str) -> str:
    if row_col == "seagrass_group":
        mapping = {
            "tt": "Thalassia testudinum",
            "hw": "Halodule wrightii",
            "sf": "Syringodium filiforme",
            "rm": "Ruppia maritima",
            "he": "Halophila engelmannii",
            "hd": "Halophila decipiens",
            "sg": "Seagrass aggregate",
            "richness": "Seagrass richness",
        }
        return mapping.get(value, value)
    if row_col == "feature_type":
        mapping = {
            "above_wt": "Above-ground biomass",
            "below_wt": "Below-ground biomass",
            "canopy": "Canopy",
            "cover": "Percent cover",
            "ep_wt": "Epiphyte weight",
            "la": "Leaf area",
            "lai": "Leaf area index",
            "leaf_length": "Leaf length",
            "leaf_width": "Leaf width",
            "lps": "Leaves per shoot",
            "richness": "Seagrass richness",
            "shoot_structure": "Shoot structure",
            "shoots": "Shoots",
        }
        return mapping.get(value, value.replace("_", " ").title())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one multi-panel seagrass summary figure with a shared legend across sites.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--row-col", required=True, choices=["seagrass_group", "feature_type"])
    parser.add_argument("--value-col", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def matrix_for_region(df: pd.DataFrame, region: str, row_col: str) -> pd.DataFrame:
    subset = df[df["region"] == region].copy()
    subset["target_label"] = subset["target_label"].map(clean_label)
    subset[row_col] = subset[row_col].map(lambda v: clean_row_label(v, row_col))
    m = subset.pivot(index=row_col, columns="target_label", values=args.value_col)
    return m


def main() -> None:
    global args
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    regions = [r for r in SITE_ORDER if r in set(df["region"].unique())]

    row_order = [clean_row_label(v, args.row_col) for v in list(df[args.row_col].drop_duplicates())]
    matrices = []
    vmax = 0.0
    for region in regions:
        subset = df[df["region"] == region].copy()
        subset["target_label"] = subset["target_label"].map(clean_label)
        subset[args.row_col] = subset[args.row_col].map(lambda v: clean_row_label(v, args.row_col))
        m = subset.pivot(index=args.row_col, columns="target_label", values=args.value_col)
        m = m.reindex(index=row_order)
        matrices.append((region, m))
        if len(m.columns):
            vmax = max(vmax, float(m.max().max()))

    fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=18)

    im = None
    for ax, (region, m) in zip(axes.flatten(), matrices):
        data = np.ma.masked_invalid(m.values.astype(float))
        x = np.arange(len(m.columns) + 1)
        y = np.arange(len(m.index) + 1)
        im = ax.pcolormesh(
            x,
            y,
            data,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            shading="flat",
            edgecolors="white",
            linewidth=0.5,
        )
        ax.set_xlim(0, len(m.columns))
        ax.set_ylim(len(m.index), 0)
        ax.set_xticks(np.arange(len(m.columns)) + 0.5)
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(m.index)) + 0.5)
        ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(region)
        ax.set_xlabel("Target Label")
    for ax in axes[:, 0]:
        ax.set_ylabel(args.row_col.replace("_", " ").title())

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.92)
        cbar.set_label("Summed Mean Absolute Attribution")

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
