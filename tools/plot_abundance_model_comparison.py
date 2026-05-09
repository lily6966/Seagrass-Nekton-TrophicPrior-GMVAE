import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two abundance model runs by species.")
    parser.add_argument("--env-metrics", required=True)
    parser.add_argument("--coocc-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = pd.read_csv(args.env_metrics).rename(
        columns={"rmse_raw": "rmse_raw_env", "r2_raw": "r2_raw_env", "mae_raw": "mae_raw_env"}
    )
    coocc = pd.read_csv(args.coocc_metrics).rename(
        columns={"rmse_raw": "rmse_raw_coocc", "r2_raw": "r2_raw_coocc", "mae_raw": "mae_raw_coocc"}
    )
    merged = env.merge(coocc, on="species", suffixes=("_env", "_coocc"))
    merged["rmse_gain"] = merged["rmse_raw_env"] - merged["rmse_raw_coocc"]
    merged["r2_gain"] = merged["r2_raw_coocc"] - merged["r2_raw_env"]
    merged["mae_gain"] = merged["mae_raw_env"] - merged["mae_raw_coocc"]
    merged.to_csv(out_dir / "abundance_model_comparison.csv", index=False)

    plot_df = merged.sort_values("r2_gain", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.35 * len(plot_df))), constrained_layout=True)

    axes[0].barh(plot_df["species"], plot_df["rmse_gain"], color="#2b7a78")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#444444", linewidth=1)
    axes[0].set_title("RMSE Improvement from Adding Co-occurrence")
    axes[0].set_xlabel("Positive means co-occurrence RMSE is lower")

    axes[1].barh(plot_df["species"], plot_df["r2_gain"], color="#1f6f8b")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#444444", linewidth=1)
    axes[1].set_title("R^2 Gain from Adding Co-occurrence")
    axes[1].set_xlabel("Positive means co-occurrence R^2 is higher")

    fig.savefig(out_dir / "abundance_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(out_dir / "abundance_model_comparison.csv")
    print(out_dir / "abundance_model_comparison.png")


if __name__ == "__main__":
    main()
