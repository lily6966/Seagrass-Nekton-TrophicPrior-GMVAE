import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot region-by-species heatmaps from integrated gradients group totals.")
    parser.add_argument("--group-totals", required=True, help="Path to integrated_gradients_group_totals.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG heatmaps")
    return parser.parse_args()


def draw_heatmap(matrix: pd.DataFrame, title: str, out_path: Path) -> None:
    fig_w = max(8, 0.45 * len(matrix.columns) + 2)
    fig_h = max(4, 0.55 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Species")
    ax.set_ylabel("Region")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Total Mean Absolute Attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.group_totals)
    if "environment" not in df.columns:
        df["environment"] = 0.0
    if "cooccurrence" not in df.columns:
        df["cooccurrence"] = 0.0
    env = df.pivot(index="region", columns="target_label", values="environment").sort_index()
    co = df.pivot(index="region", columns="target_label", values="cooccurrence").sort_index()

    env_path = out_dir / "heatmap_environment_by_region_species.png"
    co_path = out_dir / "heatmap_cooccurrence_by_region_species.png"

    draw_heatmap(env, "Environmental Importance by Region and Species", env_path)
    draw_heatmap(co, "Co-occurrence Importance by Region and Species", co_path)

    print(env_path)
    print(co_path)


if __name__ == "__main__":
    main()
