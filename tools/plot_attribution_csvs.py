import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot integrated gradients attribution CSV outputs.")
    parser.add_argument("--by-region", required=True, help="Path to integrated_gradients_by_region.csv")
    parser.add_argument("--group-totals", required=True, help="Path to integrated_gradients_group_totals.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG plots")
    parser.add_argument("--top-n", type=int, default=8, help="Top features per region/group to plot")
    return parser.parse_args()


def plot_group_totals(df: pd.DataFrame, out_dir: Path) -> Path:
    if "environment" not in df.columns:
        df["environment"] = 0.0
    if "cooccurrence" not in df.columns:
        df["cooccurrence"] = 0.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(df))
    ax.bar(x, df["environment"], label="Environment", color="#1f6f8b")
    ax.bar(x, df["cooccurrence"], bottom=df["environment"], label="Co-occurrence", color="#d98c2b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["region"])
    ax.set_ylabel("Total Mean Absolute Attribution")
    ax.set_xlabel("Region")
    ax.set_title(f"Attribution Share by Region\n{df['target_label'].iloc[0]}")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "attribution_group_totals.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_top_features(df: pd.DataFrame, out_dir: Path, feature_group: str, top_n: int) -> Path:
    regions = sorted(df["region"].unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(9, 3.8 * len(regions)))
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        subset = (
            df[(df["region"] == region) & (df["feature_group"] == feature_group)]
            .sort_values("mean_abs_attribution", ascending=True)
            .tail(top_n)
        )
        color = "#1f6f8b" if feature_group == "environment" else "#d98c2b"
        ax.barh(subset["feature_name"], subset["mean_abs_attribution"], color=color)
        ax.set_title(f"{region}: Top {top_n} {feature_group} predictors")
        ax.set_xlabel("Mean Absolute Attribution")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    path = out_dir / f"top_{feature_group}_predictors.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_region = pd.read_csv(args.by_region)
    group_totals = pd.read_csv(args.group_totals)
    if "environment" not in group_totals.columns:
        group_totals["environment"] = 0.0
    if "cooccurrence" not in group_totals.columns:
        group_totals["cooccurrence"] = 0.0

    paths = [
        plot_group_totals(group_totals, out_dir),
        plot_top_features(by_region, out_dir, "environment", args.top_n),
        plot_top_features(by_region, out_dir, "cooccurrence", args.top_n),
    ]

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
