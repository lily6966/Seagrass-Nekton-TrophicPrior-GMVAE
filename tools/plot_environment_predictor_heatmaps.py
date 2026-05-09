import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    parser = argparse.ArgumentParser(description="Plot environmental predictor importance heatmaps by region and species.")
    parser.add_argument("--by-region", required=True, help="Path to integrated_gradients_by_region.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG heatmaps")
    parser.add_argument("--top-n", type=int, default=12, help="Number of environmental predictors to display")
    parser.add_argument("--exclude-context", action="store_true", help="Exclude location, time, sample, and gear predictors.")
    return parser.parse_args()


def filter_environment(df: pd.DataFrame, exclude_context: bool) -> pd.DataFrame:
    env = df[df["feature_group"] == "environment"].copy()
    env = env[~env["feature_name"].str.endswith("__missing")]
    if not exclude_context:
        return env

    excluded_exact = {
        "gear_is_sled",
        "start_lat",
        "start_long",
        "end_lat",
        "end_long",
        "date_year",
        "date_month",
        "date_dayofyear",
    }
    excluded_prefixes = ("site_", "station_", "sample_period_")
    mask = ~env["feature_name"].isin(excluded_exact)
    mask &= ~env["feature_name"].str.startswith(excluded_prefixes)
    return env[mask].copy()


def select_top_environment_predictors(df: pd.DataFrame, top_n: int) -> list[str]:
    scores = (
        df.groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    return scores.head(top_n)["feature_name"].tolist()


def draw_region_heatmap(df: pd.DataFrame, region: str, predictors: list[str], out_dir: Path, vmax: float) -> Path:
    subset = df[(df["region"] == region) & (df["feature_name"].isin(predictors))].copy()
    subset["target_label"] = subset["target_label"].map(clean_label)
    matrix = subset.pivot(index="feature_name", columns="target_label", values="mean_abs_attribution")
    matrix = matrix.reindex(index=predictors)

    fig_w = max(10, 0.45 * len(matrix.columns) + 3)
    fig_h = max(5, 0.45 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Target Species")
    ax.set_ylabel("Environmental Predictor")
    ax.set_title(f"Environmental Predictor Importance\nRegion: {region}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Attribution")
    fig.tight_layout()

    path = out_dir / f"environment_predictors_heatmap_{region}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_combined_heatmap(df: pd.DataFrame, predictors: list[str], out_dir: Path) -> Path:
    subset = df[df["feature_name"].isin(predictors)].copy()
    subset["target_label"] = subset["target_label"].map(clean_label)
    subset["region_species"] = subset["region"] + " | " + subset["target_label"]
    matrix = subset.pivot(index="feature_name", columns="region_species", values="mean_abs_attribution")
    matrix = matrix.reindex(index=predictors)

    fig_w = max(14, 0.35 * len(matrix.columns) + 4)
    fig_h = max(5, 0.45 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(matrix.max().max()) if len(matrix.columns) else 0.0
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Region | Target Species")
    ax.set_ylabel("Environmental Predictor")
    ax.set_title("Environmental Predictor Importance Across Regions and Species")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Attribution")
    fig.tight_layout()

    path = out_dir / "environment_predictors_heatmap_combined.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.by_region)
    env = filter_environment(df, args.exclude_context)

    predictors = select_top_environment_predictors(env, args.top_n)
    vmax = 0.0
    if predictors:
        vmax = float(
            env[env["feature_name"].isin(predictors)]["mean_abs_attribution"].max()
        )
    paths = [draw_combined_heatmap(env, predictors, out_dir)]
    for region in sorted(env["region"].unique()):
        paths.append(draw_region_heatmap(env, region, predictors, out_dir, vmax))

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
