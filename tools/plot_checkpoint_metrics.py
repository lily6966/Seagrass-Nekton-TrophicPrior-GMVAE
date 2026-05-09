from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "species_performance" / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_balanced" / "new_checkpoint_test_metrics_table.csv"
OUT_DIR = ROOT / "output" / "species_performance" / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_balanced"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    metrics = ["ha", "ebF1", "miF1", "maF1", "p_at_1"]
    display = {
        "ha": "HA",
        "ebF1": "ebF1",
        "miF1": "miF1",
        "maF1": "maF1",
        "p_at_1": "P@1",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(df))]

    for ax, metric in zip(axes, metrics):
        ax.bar(df["checkpoint"], df[metric], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(display[metric])
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for i, value in enumerate(df[metric]):
            ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("New Feed-Position Grouped Model: Checkpoint Test Metrics", fontsize=14)

    png_path = OUT_DIR / "new_checkpoint_test_metrics_comparison.png"
    pdf_path = OUT_DIR / "new_checkpoint_test_metrics_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
