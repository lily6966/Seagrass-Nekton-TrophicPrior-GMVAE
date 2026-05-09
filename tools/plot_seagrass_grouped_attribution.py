import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SEAGRASS_CODE_PREFIXES = ("tt_", "hd_", "he_", "hw_", "rm_", "sf_", "station_tt_", "station_hw_", "station_sf_", "station_he_", "station_hd_", "station_rm_", "station_sg_")
SEAGRASS_CODES = ("tt", "hd", "he", "hw", "rm", "sf", "sg")


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
    parser = argparse.ArgumentParser(description="Plot averaged seagrass attribution summaries.")
    parser.add_argument("--by-region", required=True, help="Path to integrated_gradients_by_region.csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def is_seagrass_feature(name: str) -> bool:
    if name.endswith("__missing"):
        return False
    if name == "seagrass_richness":
        return True
    return name.startswith(SEAGRASS_CODE_PREFIXES)


def extract_code(name: str) -> str:
    if name == "seagrass_richness":
        return "richness"
    for code in SEAGRASS_CODES:
        if name.startswith(f"{code}_") or name.startswith(f"station_{code}_"):
            return code
    return "other"


def extract_feature_type(name: str) -> str:
    if name == "seagrass_richness":
        return "richness"
    cleaned = name
    for prefix in ("station_",):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    for code in SEAGRASS_CODES:
        if cleaned.startswith(f"{code}_"):
            cleaned = cleaned[len(code) + 1 :]
            break
    if cleaned.startswith("stationsp_"):
        cleaned = cleaned[len("stationsp_") :]
    if cleaned.startswith("station_"):
        cleaned = cleaned[len("station_") :]
    family_rules = [
        ("leaf_mlength", "leaf_length"),
        ("leaf_sd_width", "leaf_width"),
        ("leaf_width", "leaf_width"),
        ("sd_mcanopy", "canopy"),
        ("se_mcanopy", "canopy"),
        ("mcanopy", "canopy"),
        ("lai_se", "lai"),
        ("lai_sd", "lai"),
        ("lai", "lai"),
        ("la_sd", "la"),
        ("la", "la"),
        ("lps_sd", "lps"),
        ("lps", "lps"),
        ("sd_shoots_m", "shoots"),
        ("se_shoots_m", "shoots"),
        ("shoots_m", "shoots"),
        ("sd_shoots", "shoots"),
        ("se_shoots", "shoots"),
        ("shoots", "shoots"),
        ("sd_above_wt_m", "above_wt"),
        ("above_wt_m", "above_wt"),
        ("sd_below_wt_m", "below_wt"),
        ("below_wt_m", "below_wt"),
        ("sd_ep_wt_gg", "ep_wt"),
        ("ep_wt_gg", "ep_wt"),
        ("sht_ratio", "shoot_structure"),
        ("sht_m", "shoot_structure"),
        ("pcov", "cover"),
        ("can", "canopy"),
    ]
    for suffix, family in family_rules:
        if cleaned.endswith(suffix) or cleaned == suffix:
            return family
    return cleaned


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["feature_group"] == "environment"].copy()
    out = out[out["feature_name"].map(is_seagrass_feature)].copy()
    out["seagrass_group"] = out["feature_name"].map(extract_code)
    out["feature_type"] = out["feature_name"].map(extract_feature_type)
    return out


def average_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["region", "target_label", group_col], as_index=False)["mean_abs_attribution"]
        .sum()
        .rename(columns={"mean_abs_attribution": "sum_mean_abs_attribution"})
    )
    grouped["target_label"] = grouped["target_label"].map(clean_label)
    return grouped


def draw_heatmap(df: pd.DataFrame, row_col: str, value_col: str, title: str, path: Path) -> None:
    df = df.copy()
    df["region_label"] = df["region"] + " | " + df["target_label"]
    matrix = df.pivot(index=row_col, columns="region_label", values=value_col)
    vmax = float(matrix.max().max()) if len(matrix.columns) else 0.0

    fig_w = max(14, 0.35 * len(matrix.columns) + 4)
    fig_h = max(4.5, 0.45 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Region | Target Label")
    ax.set_ylabel(row_col.replace("_", " ").title())
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Summed Mean Absolute Attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_region_heatmaps(df: pd.DataFrame, row_col: str, value_col: str, title_prefix: str, out_dir: Path, stem: str) -> list[Path]:
    vmax = float(df[value_col].max()) if not df.empty else 0.0
    paths = []
    for region in sorted(df["region"].unique()):
        subset = df[df["region"] == region].copy()
        matrix = subset.pivot(index=row_col, columns="target_label", values=value_col)

        fig_w = max(10, 0.45 * len(matrix.columns) + 3)
        fig_h = max(4.5, 0.45 * len(matrix.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel("Target Label")
        ax.set_ylabel(row_col.replace("_", " ").title())
        ax.set_title(f"{title_prefix}\nRegion: {region}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Summed Mean Absolute Attribution")
        fig.tight_layout()

        path = out_dir / f"{stem}_{region}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.by_region)
    sea = prep(df)

    by_code = average_by_group(sea, "seagrass_group")
    by_type = average_by_group(sea, "feature_type")

    by_code_csv = out_dir / "seagrass_group_sum_importance.csv"
    by_type_csv = out_dir / "seagrass_feature_type_sum_importance.csv"
    by_code.to_csv(by_code_csv, index=False)
    by_type.to_csv(by_type_csv, index=False)

    code_png = out_dir / "seagrass_group_sum_importance.png"
    type_png = out_dir / "seagrass_feature_type_sum_importance.png"
    draw_heatmap(by_code, "seagrass_group", "sum_mean_abs_attribution", "Summed Importance by Seagrass Group", code_png)
    draw_heatmap(by_type, "feature_type", "sum_mean_abs_attribution", "Summed Importance by Seagrass Feature Family", type_png)
    region_paths = []
    region_paths.extend(
        draw_region_heatmaps(
            by_code,
            "seagrass_group",
            "sum_mean_abs_attribution",
            "Summed Importance by Seagrass Group",
            out_dir,
            "seagrass_group_sum_importance",
        )
    )
    region_paths.extend(
        draw_region_heatmaps(
            by_type,
            "feature_type",
            "sum_mean_abs_attribution",
            "Summed Importance by Seagrass Feature Family",
            out_dir,
            "seagrass_feature_type_sum_importance",
        )
    )

    print(by_code_csv)
    print(by_type_csv)
    print(code_png)
    print(type_png)
    for path in region_paths:
        print(path)


if __name__ == "__main__":
    main()
