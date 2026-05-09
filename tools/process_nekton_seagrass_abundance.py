from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nekton-seagrass"
SLED_CSV = DATA_DIR / "combined_sled_species_sp.csv"
TRAWL_CSV = DATA_DIR / "combined_trawl_species.csv"
TOP20_LABELS_JSON = DATA_DIR / "label_columns_top20_restfeat.json"
SEED = 1


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def seagrass_columns(df: pd.DataFrame, gear_is_sled: int) -> list[str]:
    if gear_is_sled:
        prefixes = ("tt_", "hd_", "he_", "hw_", "rm_", "sf_")
    else:
        prefixes = (
            "station_tt_",
            "station_hw_",
            "station_sf_",
            "station_he_",
            "station_hd_",
            "station_rm_",
            "station_sg_",
        )
    return [c for c in df.columns if c.startswith(prefixes)]


def choose_species_label(df: pd.DataFrame) -> pd.Series:
    sci = df["species_scientific"].fillna("").astype(str).str.strip()
    common = df["species_common"].fillna("").astype(str).str.strip()
    label = sci.where(sci.ne(""), common)
    return label.replace({"": "Unknown species"})


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.to_datetime(df["date"], errors="coerce")
    df["date_year"] = parsed.dt.year.astype(float)
    df["date_month"] = parsed.dt.month.astype(float)
    df["date_dayofyear"] = parsed.dt.dayofyear.astype(float)
    return df.drop(columns=["date"])


def build_feature_matrix(df: pd.DataFrame, drop_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    feature_df = df.drop(columns=drop_cols).copy()
    feature_df = parse_dates(feature_df)

    categorical_cols = ["site", "station", "sample_period", "substrate"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    numeric_part = feature_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    missing_flags = numeric_part.isna().astype(float)
    missing_flags.columns = [f"{c}__missing" for c in numeric_part.columns]

    medians = numeric_part.median().fillna(0.0)
    numeric_part = numeric_part.fillna(medians)
    means = numeric_part.mean()
    stds = numeric_part.std(ddof=0).replace(0, 1.0)
    numeric_part = (numeric_part - means) / stds

    cat_part = pd.get_dummies(
        feature_df[categorical_cols].fillna("Missing").astype(str),
        prefix=categorical_cols,
        dtype=float,
    )

    final = pd.concat([numeric_part, missing_flags, cat_part], axis=1).astype(np.float64)
    return final.to_numpy(), list(final.columns)


def scale_dense_matrix(values: np.ndarray, prefix: str, names: list[str]) -> tuple[np.ndarray, list[str]]:
    frame = pd.DataFrame(values, columns=names)
    medians = frame.median().fillna(0.0)
    frame = frame.fillna(medians)
    means = frame.mean()
    stds = frame.std(ddof=0).replace(0, 1.0)
    frame = (frame - means) / stds
    return frame.to_numpy(dtype=np.float64), [f"{prefix}{name}" for name in names]


def build_sample_table(df: pd.DataFrame, sample_id_col: str, gear_is_sled: int, label_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["sample_id"] = df[sample_id_col].astype(str)
    df["gear_is_sled"] = float(gear_is_sled)
    df["species_label"] = choose_species_label(df)
    df["species_abundance"] = pd.to_numeric(df["species_abundance"], errors="coerce").fillna(0.0)

    sample_level_cols = [
        "sample_id",
        "site",
        "station",
        "sample_period",
        "substrate",
        "date",
        "light_attenuation_av",
        "depth",
        "secchi",
        "temp",
        "salinity",
        "do",
        "start_lat",
        "start_long",
        "end_lat",
        "end_long",
        "macro_weight",
        "gear_is_sled",
    ]

    if "station_seagrass_richness" in df.columns:
        df["seagrass_richness"] = df["station_seagrass_richness"]
    elif "seagrass_richness" in df.columns:
        df["seagrass_richness"] = df["seagrass_richness"]
    else:
        df["seagrass_richness"] = np.nan
    sample_level_cols.append("seagrass_richness")
    sample_level_cols.extend(seagrass_columns(df, gear_is_sled))

    samples = df[sample_level_cols].groupby("sample_id", as_index=False).first()

    abundances = (
        df.groupby(["sample_id", "species_label"])["species_abundance"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=label_cols, fill_value=0.0)
        .reset_index()
    )

    out = samples.merge(abundances, on="sample_id", how="left")
    out[label_cols] = out[label_cols].fillna(0.0)
    return out


def build_output_paths(stem_suffix: str) -> dict[str, Path]:
    stem = f"nekton_seagrass_{stem_suffix}"
    return {
        "data": DATA_DIR / f"{stem}_data.npy",
        "train": DATA_DIR / f"{stem}_train_idx.npy",
        "val": DATA_DIR / f"{stem}_val_idx.npy",
        "test": DATA_DIR / f"{stem}_test_idx.npy",
        "count": DATA_DIR / f"count_{stem_suffix}.txt",
        "features": DATA_DIR / f"feature_columns_{stem_suffix}.json",
        "labels": DATA_DIR / f"label_columns_{stem_suffix}.json",
        "info": DATA_DIR / f"processing_info_{stem_suffix}.json",
        "samples": DATA_DIR / f"sample_metadata_{stem_suffix}.csv",
    }


def make_splits(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_rows)
    n_train = int(round(n_rows * 0.72))
    n_val = int(round(n_rows * 0.18))
    train_idx = np.sort(idx[:n_train])
    val_idx = np.sort(idx[n_train:n_train + n_val])
    test_idx = np.sort(idx[n_train + n_val :])
    return train_idx, val_idx, test_idx


def write_dataset(
    combined: pd.DataFrame,
    label_cols: list[str],
    all_species_cols: list[str],
    cooccurrence_mode: str,
    out_paths: dict[str, Path],
) -> None:
    labels_raw = combined[label_cols].astype(np.float64)
    labels = np.log1p(labels_raw.to_numpy())

    feature_base, feature_cols = build_feature_matrix(combined, ["sample_id"] + all_species_cols)
    features = feature_base

    if cooccurrence_mode != "none":
        other_species_cols = [c for c in all_species_cols if c not in set(label_cols)]
        other_species = combined[other_species_cols].astype(np.float64).to_numpy()
        if cooccurrence_mode == "presence":
            other_species = (other_species > 0).astype(np.float64)
            prefix = "species_presence__"
        else:
            other_species = np.log1p(other_species)
            prefix = "species_log1p__"
        scaled_other, other_cols = scale_dense_matrix(other_species, prefix, other_species_cols)
        features = np.concatenate([features, scaled_other], axis=1)
        feature_cols = feature_cols + other_cols

    data = np.concatenate([labels, features], axis=1)
    train_idx, val_idx, test_idx = make_splits(len(combined))

    np.save(out_paths["data"], data)
    np.save(out_paths["train"], train_idx)
    np.save(out_paths["val"], val_idx)
    np.save(out_paths["test"], test_idx)

    out_paths["count"].write_text(f"{features.shape[1]}\n{labels.shape[1]}\n")
    out_paths["features"].write_text(json.dumps(feature_cols, indent=2))
    out_paths["labels"].write_text(json.dumps(label_cols, indent=2))
    out_paths["samples"].write_text(
        combined[["sample_id", "site", "station", "sample_period", "substrate", "gear_is_sled"]].to_csv(index=False)
    )
    out_paths["info"].write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_samples": int(len(combined)),
                "feature_dim": int(features.shape[1]),
                "label_dim": int(labels.shape[1]),
                "label_transform": "log1p(raw species_abundance aggregated to haul/tow level)",
                "top20_source": str(TOP20_LABELS_JSON.name),
                "cooccurrence_mode": cooccurrence_mode,
                "gear_feature": "gear_is_sled (1=sled, 0=trawl)",
                "files": {key: str(path.name) for key, path in out_paths.items()},
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build haul-level abundance datasets for nekton-seagrass.")
    parser.add_argument(
        "--cooccurrence-mode",
        choices=["none", "presence", "abundance"],
        default="none",
        help="How to encode non-target species as predictor features.",
    )
    parser.add_argument(
        "--labels-json",
        default=str(TOP20_LABELS_JSON),
        help="JSON file containing the target species labels to model.",
    )
    parser.add_argument(
        "--output-stem-suffix",
        default="",
        help="Optional custom stem suffix. If omitted, a default based on cooccurrence mode is used.",
    )
    args = parser.parse_args()

    labels_json = Path(args.labels_json)
    label_cols = json.loads(labels_json.read_text())

    sled = load_frame(SLED_CSV)
    trawl = load_frame(TRAWL_CSV)

    all_species_cols = sorted(
        set(choose_species_label(sled).tolist()) | set(choose_species_label(trawl).tolist())
    )

    sled_samples = build_sample_table(sled, "pull_id", 1, all_species_cols)
    trawl_samples = build_sample_table(trawl, "tow_id", 0, all_species_cols)
    combined = pd.concat([sled_samples, trawl_samples], ignore_index=True, sort=False)
    combined[all_species_cols] = combined[all_species_cols].fillna(0.0)

    if args.output_stem_suffix:
        stem_suffix = args.output_stem_suffix
    elif args.cooccurrence_mode == "none":
        stem_suffix = "abundance_top20_env_seagrass"
    else:
        stem_suffix = f"abundance_top20_coocc_{args.cooccurrence_mode}_seagrass"

    out_paths = build_output_paths(stem_suffix)
    write_dataset(combined, label_cols, all_species_cols, args.cooccurrence_mode, out_paths)

    print(out_paths["data"])
    print(out_paths["train"])
    print(out_paths["val"])
    print(out_paths["test"])


if __name__ == "__main__":
    main()
