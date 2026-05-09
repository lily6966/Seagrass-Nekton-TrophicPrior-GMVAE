import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_label(label: str) -> str:
    return label.replace("niche_label__", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare environmental predictor importance heatmaps for sled vs trawl.")
    parser.add_argument("--sled-csv", required=True, help="Sled-only integrated_gradients_by_region.csv")
    parser.add_argument("--trawl-csv", required=True, help="Trawl-only integrated_gradients_by_region.csv")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--exclude-context", action="store_true", help="Exclude location, time, sample, and gear predictors.")
    return parser.parse_args()


def filter_environment(df: pd.DataFrame, exclude_context: bool) -> pd.DataFrame:
    env = df[(df["feature_group"] == "environment") & (~df["feature_name"].str.endswith("__missing"))].copy()
    env = env[env["feature_name"] != "gear_is_sled"]
    if not exclude_context:
        return env

    excluded_exact = {
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


def top_predictors(df1: pd.DataFrame, df2: pd.DataFrame, top_n: int, exclude_context: bool) -> list[str]:
    both = pd.concat([df1, df2], ignore_index=True)
    env = filter_environment(both, exclude_context)
    scores = (
        env.groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    return scores.head(top_n)["feature_name"].tolist()


def matrix(df: pd.DataFrame, predictors: list[str], exclude_context: bool) -> pd.DataFrame:
    env = filter_environment(df, exclude_context)
    env = env[env["feature_name"].isin(predictors)].copy()
    env["target_label"] = env["target_label"].map(clean_label)
    env["region_species"] = env["region"] + " | " + env["target_label"]
    m = env.pivot(index="feature_name", columns="region_species", values="mean_abs_attribution")
    return m.reindex(index=predictors)


def main() -> None:
    args = parse_args()
    sled = pd.read_csv(args.sled_csv)
    trawl = pd.read_csv(args.trawl_csv)
    predictors = top_predictors(sled, trawl, args.top_n, args.exclude_context)

    sled_m = matrix(sled, predictors, args.exclude_context)
    trawl_m = matrix(trawl, predictors, args.exclude_context)
    vmax = max(float(sled_m.max().max()), float(trawl_m.max().max()))

    fig, axes = plt.subplots(1, 2, figsize=(24, 6), constrained_layout=True)
    for ax, m, title in [
        (axes[0], sled_m, "Sled Predictions"),
        (axes[1], trawl_m, "Trawl Predictions"),
    ]:
        im = ax.imshow(m.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index)
        ax.set_title(title)
        ax.set_xlabel("Region | Target Species")
    axes[0].set_ylabel("Environmental Predictor")
    cbar = fig.colorbar(im, ax=axes, shrink=0.95)
    cbar.set_label("Mean Absolute Attribution")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
