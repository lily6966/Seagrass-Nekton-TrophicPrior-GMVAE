import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_environment_predictor_heatmaps import filter_environment, select_top_environment_predictors


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one multi-panel environmental heatmap with shared legend across sites.")
    parser.add_argument("--by-region", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--exclude-context", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--tick-fontsize", type=int, default=8)
    parser.add_argument("--label-fontsize", type=int, default=10)
    parser.add_argument("--title-fontsize", type=int, default=12)
    parser.add_argument("--suptitle-fontsize", type=int, default=18)
    parser.add_argument("--colorbar-fontsize", type=int, default=10)
    return parser.parse_args()


def matrix_for_region(df: pd.DataFrame, region: str, predictors: list[str]) -> pd.DataFrame:
    subset = df[(df["region"] == region) & (df["feature_name"].isin(predictors))].copy()
    subset["target_label"] = subset["target_label"].map(clean_label)
    m = subset.pivot(index="feature_name", columns="target_label", values="mean_abs_attribution")
    return m.reindex(index=predictors)


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.by_region)
    env = filter_environment(df, args.exclude_context)
    predictors = select_top_environment_predictors(env, args.top_n)

    matrices = []
    vmax = 0.0
    regions = [r for r in SITE_ORDER if r in set(env["region"].unique())]
    for region in regions:
        m = matrix_for_region(env, region, predictors)
        matrices.append((region, m))
        if len(m.columns):
            vmax = max(vmax, float(m.max().max()))

    fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=args.suptitle_fontsize)

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
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=args.tick_fontsize)
        ax.set_yticks(np.arange(len(m.index)) + 0.5)
        ax.set_yticklabels(m.index, fontsize=args.tick_fontsize)
        ax.set_title(region, fontsize=args.title_fontsize)
        ax.set_xlabel("Target Label", fontsize=args.label_fontsize)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predictor", fontsize=args.label_fontsize)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.92)
        cbar.set_label("Mean Absolute Attribution", fontsize=args.colorbar_fontsize)
        cbar.ax.tick_params(labelsize=args.tick_fontsize)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
