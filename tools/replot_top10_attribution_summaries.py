import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TROPHIC_DEFAULTS = [
    "/Users/liyingnceas/GitHub/Multi-taxa-GM-VAE/output/attribution/transfer_learning_pretrain_feedpos11_allspecies_sled_nogear_site/integrated_gradients_by_region.csv",
    "/Users/liyingnceas/GitHub/Multi-taxa-GM-VAE/output/attribution/transfer_learning_pretrain_feedpos11_allspecies_trawl_nogear_site/integrated_gradients_by_region.csv",
]

SPECIES_DEFAULTS = [
    "/Users/liyingnceas/GitHub/Multi-taxa-GM-VAE/output/attribution/new_transfer_finetune_fisheries8_fullgroupconsistency_site_sled/integrated_gradients_by_region.csv",
    "/Users/liyingnceas/GitHub/Multi-taxa-GM-VAE/output/attribution/new_transfer_finetune_fisheries8_fullgroupconsistency_site_trawl/integrated_gradients_by_region.csv",
]


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
    }
    return friendly.get(cleaned, cleaned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot top-10 attribution heatmaps and averaged barplots for trophic groups and fishery species."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exclude-context", action="store_true", default=True)
    parser.add_argument("--trophic-csvs", nargs="*", default=TROPHIC_DEFAULTS)
    parser.add_argument("--species-csvs", nargs="*", default=SPECIES_DEFAULTS)
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


def load_and_average(csv_paths: list[str], exclude_context: bool, exclude_targets: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = filter_environment(df, exclude_context)
        df["target_label"] = df["target_label"].map(clean_label)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if exclude_targets:
        combined = combined[~combined["target_label"].isin(exclude_targets)].copy()
    averaged = (
        combined.groupby(["feature_name", "target_label"], as_index=False)["mean_abs_attribution"]
        .mean()
    )
    return averaged


def select_top_features(df: pd.DataFrame, top_n: int) -> list[str]:
    scores = (
        df.groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    return scores.head(top_n)["feature_name"].tolist()


def draw_heatmap(df: pd.DataFrame, predictors: list[str], out_path: Path, title: str, x_label: str) -> None:
    matrix = df[df["feature_name"].isin(predictors)].pivot(
        index="feature_name", columns="target_label", values="mean_abs_attribution"
    )
    matrix = matrix.reindex(index=predictors)

    fig_w = max(10, 0.55 * len(matrix.columns) + 3)
    fig_h = max(5, 0.45 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(matrix.max().max()) if len(matrix.columns) else 0.0
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Environmental Predictor")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_average_barplot(df: pd.DataFrame, predictors: list[str], out_path: Path, title: str) -> None:
    scores = (
        df[df["feature_name"].isin(predictors)]
        .groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=True)
    )
    fig_h = max(5, 0.45 * len(scores) + 1)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(scores["feature_name"], scores["mean_abs_attribution"], color="#1f77b4")
    ax.set_xlabel("Mean Absolute Attribution")
    ax.set_ylabel("Environmental Predictor")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_combined_average_barplots(
    trophic_df: pd.DataFrame,
    trophic_predictors: list[str],
    species_df: pd.DataFrame,
    species_predictors: list[str],
    out_path: Path,
) -> None:
    trophic_scores = (
        trophic_df[trophic_df["feature_name"].isin(trophic_predictors)]
        .groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=True)
    )
    species_scores = (
        species_df[species_df["feature_name"].isin(species_predictors)]
        .groupby("feature_name", as_index=False)["mean_abs_attribution"]
        .mean()
        .sort_values("mean_abs_attribution", ascending=True)
    )

    fig_h = max(6, 0.38 * max(len(trophic_scores), len(species_scores)) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_h), constrained_layout=True)

    axes[0].barh(
        trophic_scores["feature_name"],
        trophic_scores["mean_abs_attribution"],
        color="#1f77b4",
    )
    axes[0].set_title("Average Across All Trophic Groups")
    axes[0].set_xlabel("Mean Absolute Attribution")
    axes[0].set_ylabel("Environmental Predictor")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(
        species_scores["feature_name"],
        species_scores["mean_abs_attribution"],
        color="#ff7f0e",
    )
    axes[1].set_title("Average Across 6 Fishery Species")
    axes[1].set_xlabel("Mean Absolute Attribution")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", alpha=0.25)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trophic_df = load_and_average(args.trophic_csvs, args.exclude_context)
    trophic_top = select_top_features(trophic_df, args.top_n)
    trophic_df.to_csv(out_dir / "trophic_groups_averaged_by_target.csv", index=False)
    draw_heatmap(
        trophic_df,
        trophic_top,
        out_dir / "trophic_groups_top10_heatmap.png",
        "Top 10 Environmental Predictors Averaged Across Trophic Groups",
        "Trophic Group",
    )
    draw_average_barplot(
        trophic_df,
        trophic_top,
        out_dir / "trophic_groups_top10_average_barplot.png",
        "Average Attribution Across All Trophic Groups",
    )

    species_df = load_and_average(
        args.species_csvs,
        args.exclude_context,
        exclude_targets={"Red drum", "White shrimp"},
    )
    species_top = select_top_features(species_df, args.top_n)
    species_df.to_csv(out_dir / "fishery_species6_averaged_by_target.csv", index=False)
    draw_heatmap(
        species_df,
        species_top,
        out_dir / "fishery_species6_top10_heatmap.png",
        "Top 10 Environmental Predictors Averaged Across 6 Fishery Species",
        "Fishery-Important Species",
    )
    draw_average_barplot(
        species_df,
        species_top,
        out_dir / "fishery_species6_top10_average_barplot.png",
        "Average Attribution Across 6 Fishery Species",
    )
    draw_combined_average_barplots(
        trophic_df,
        trophic_top,
        species_df,
        species_top,
        out_dir / "combined_average_barplots.png",
    )

    for path in [
        out_dir / "trophic_groups_averaged_by_target.csv",
        out_dir / "trophic_groups_top10_heatmap.png",
        out_dir / "trophic_groups_top10_average_barplot.png",
        out_dir / "fishery_species6_averaged_by_target.csv",
        out_dir / "fishery_species6_top10_heatmap.png",
        out_dir / "fishery_species6_top10_average_barplot.png",
        out_dir / "combined_average_barplots.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
