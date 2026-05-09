import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOCATION_FEATURES = [
    "start_lat",
    "start_long",
    "end_lat",
    "end_long",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot location predictor importance heatmaps by region and label.")
    parser.add_argument("--by-region", required=True, help="Path to integrated_gradients_by_region.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG heatmaps")
    return parser.parse_args()


def filter_location(df: pd.DataFrame) -> pd.DataFrame:
    env = df[df["feature_group"] == "environment"].copy()
    env = env[env["feature_name"].isin(LOCATION_FEATURES)]
    return env


def draw_region_heatmap(df: pd.DataFrame, region: str, out_dir: Path) -> Path:
    subset = df[df["region"] == region].copy()
    matrix = subset.pivot(index="feature_name", columns="target_label", values="mean_abs_attribution")
    matrix = matrix.reindex(index=LOCATION_FEATURES)

    fig_w = max(10, 0.45 * len(matrix.columns) + 3)
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Target Label")
    ax.set_ylabel("Location Predictor")
    ax.set_title(f"Location Predictor Importance\nRegion: {region}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Attribution")
    fig.tight_layout()

    path = out_dir / f"location_predictors_heatmap_{region}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_combined_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    subset = df.copy()
    subset["region_label"] = subset["region"] + " | " + subset["target_label"]
    matrix = subset.pivot(index="feature_name", columns="region_label", values="mean_abs_attribution")
    matrix = matrix.reindex(index=LOCATION_FEATURES)

    fig_w = max(14, 0.35 * len(matrix.columns) + 4)
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Region | Target Label")
    ax.set_ylabel("Location Predictor")
    ax.set_title("Location Predictor Importance Across Regions and Labels")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Attribution")
    fig.tight_layout()

    path = out_dir / "location_predictors_heatmap_combined.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.by_region)
    loc = filter_location(df)

    paths = [draw_combined_heatmap(loc, out_dir)]
    for region in sorted(loc["region"].unique()):
        paths.append(draw_region_heatmap(loc, region, out_dir))

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
