from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nekton-seagrass"

SLED_CSV = DATA_DIR / "combined_sled_species_sp.csv"
TRAWL_CSV = DATA_DIR / "combined_trawl_species.csv"

SEED = 1


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def seagrass_columns(df: pd.DataFrame, gear_is_sled: int) -> list[str]:
    if gear_is_sled:
        prefixes = ("tt_", "hd_", "he_", "hw_", "rm_", "sf_")
    else:
        prefixes = ("station_tt_", "station_hw_", "station_sf_", "station_he_", "station_hd_", "station_rm_", "station_sg_")
    return [c for c in df.columns if c.startswith(prefixes)]


def choose_species_label(df: pd.DataFrame) -> pd.Series:
    sci = df["species_scientific"].fillna("").astype(str).str.strip()
    common = df["species_common"].fillna("").astype(str).str.strip()
    label = sci.where(sci.ne(""), common)
    label = label.replace({"": "Unknown species"})
    return label


def build_sample_table(df: pd.DataFrame, sample_id_col: str, gear_is_sled: int) -> pd.DataFrame:
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

    positive = df[df["species_abundance"] > 0].copy()
    positive["present"] = 1.0

    labels = (
        positive.pivot_table(
            index="sample_id",
            columns="species_label",
            values="present",
            aggfunc="max",
            fill_value=0.0,
        )
        .sort_index(axis=1)
        .reset_index()
    )

    out = samples.merge(labels, on="sample_id", how="left")
    label_cols = [c for c in out.columns if c not in sample_level_cols]
    if label_cols:
        out[label_cols] = out[label_cols].fillna(0.0)
    return out


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

    final = pd.concat([numeric_part, missing_flags, cat_part], axis=1)
    final = final.astype(np.float64)
    return final.to_numpy(), list(final.columns)


def make_splits(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_rows)
    n_train = int(round(n_rows * 0.72))
    n_val = int(round(n_rows * 0.18))
    train_idx = np.sort(idx[:n_train])
    val_idx = np.sort(idx[n_train:n_train + n_val])
    test_idx = np.sort(idx[n_train + n_val:])
    return train_idx, val_idx, test_idx


def build_output_paths(suffix: str) -> dict[str, Path]:
    stem = f"nekton_seagrass{suffix}"
    count_name = "count.txt" if not suffix else f"count{suffix}.txt"
    return {
        "data": DATA_DIR / f"{stem}_data.npy",
        "train": DATA_DIR / f"{stem}_train_idx.npy",
        "val": DATA_DIR / f"{stem}_val_idx.npy",
        "test": DATA_DIR / f"{stem}_test_idx.npy",
        "count": DATA_DIR / count_name,
        "features": DATA_DIR / f"feature_columns{suffix}.json",
        "labels": DATA_DIR / f"label_columns{suffix}.json",
        "info": DATA_DIR / f"processing_info{suffix}.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-labels", type=int, default=0, help="Keep only the top-k most abundant labels; 0 keeps all labels.")
    parser.add_argument(
        "--labels-json",
        default="",
        help="Optional JSON file with an explicit label list to keep as targets.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional explicit output suffix, e.g. _fisheries_joint_seagrass.",
    )
    parser.add_argument(
        "--use-other-species-as-features",
        action="store_true",
        help="When top-k labels are used, append the remaining species presence columns as predictor features.",
    )
    parser.add_argument(
        "--include-seagrass-columns",
        action="store_true",
        help="Preserve gear-specific seagrass habitat columns from the raw files as predictors.",
    )
    args = parser.parse_args()

    suffix = "" if args.top_k_labels <= 0 else f"_top{args.top_k_labels}"
    if args.labels_json:
        suffix = suffix or "_custom"
    if (args.top_k_labels > 0 or args.labels_json) and args.use_other_species_as_features:
        suffix += "_restfeat"
    if args.include_seagrass_columns:
        suffix += "_seagrass"
    if args.output_suffix:
        suffix = args.output_suffix
    out = build_output_paths(suffix)

    sled = load_frame(SLED_CSV)
    trawl = load_frame(TRAWL_CSV)

    sled_samples = build_sample_table(sled, sample_id_col="pull_id", gear_is_sled=1)
    trawl_samples = build_sample_table(trawl, sample_id_col="tow_id", gear_is_sled=0)
    if not args.include_seagrass_columns:
        sled_drop = seagrass_columns(sled, 1)
        trawl_drop = seagrass_columns(trawl, 0)
        sled_samples = sled_samples.drop(columns=[c for c in sled_drop if c in sled_samples.columns])
        trawl_samples = trawl_samples.drop(columns=[c for c in trawl_drop if c in trawl_samples.columns])

    combined = pd.concat([sled_samples, trawl_samples], ignore_index=True, sort=False)

    raw_species_labels = sorted(
        set(choose_species_label(sled[pd.to_numeric(sled["species_abundance"], errors="coerce").fillna(0.0) > 0]).tolist())
        | set(choose_species_label(trawl[pd.to_numeric(trawl["species_abundance"], errors="coerce").fillna(0.0) > 0]).tolist())
    )
    all_label_cols = [c for c in raw_species_labels if c in combined.columns]
    label_cols = all_label_cols
    if args.labels_json:
        requested = set(json.loads(Path(args.labels_json).read_text()))
        label_cols = [c for c in all_label_cols if c in requested]
    elif args.top_k_labels > 0:
        positive_rows = pd.concat(
            [
                sled.assign(species_label=choose_species_label(sled), species_abundance=pd.to_numeric(sled["species_abundance"], errors="coerce").fillna(0.0))[["species_label", "species_abundance"]],
                trawl.assign(species_label=choose_species_label(trawl), species_abundance=pd.to_numeric(trawl["species_abundance"], errors="coerce").fillna(0.0))[["species_label", "species_abundance"]],
            ],
            ignore_index=True,
        )
        top_labels = (
            positive_rows.groupby("species_label")["species_abundance"]
            .sum()
            .sort_values(ascending=False)
            .head(args.top_k_labels)
            .index.tolist()
        )
        label_cols = [c for c in all_label_cols if c in set(top_labels)]

    other_species_feature_names: list[str] = []
    other_species_features = None
    if (args.top_k_labels > 0 or args.labels_json) and args.use_other_species_as_features:
        other_label_cols = [c for c in all_label_cols if c not in set(label_cols)]
        if other_label_cols:
            other_species_features = combined[other_label_cols].fillna(0.0).astype(np.float64).to_numpy()
            other_species_feature_names = [f"species_feature__{c}" for c in other_label_cols]

    combined[label_cols] = combined[label_cols].fillna(0.0)
    labels = combined[label_cols].astype(np.float64).to_numpy()
    features, feature_cols = build_feature_matrix(combined, ["sample_id"] + all_label_cols)
    if other_species_features is not None:
        features = np.concatenate([features, other_species_features], axis=1)
        feature_cols = feature_cols + other_species_feature_names

    data = np.concatenate([labels, features], axis=1)
    train_idx, val_idx, test_idx = make_splits(len(combined))

    np.save(out["data"], data)
    np.save(out["train"], train_idx)
    np.save(out["val"], val_idx)
    np.save(out["test"], test_idx)

    out["count"].write_text(f"{features.shape[1]}\n{labels.shape[1]}\n")
    out["features"].write_text(json.dumps(feature_cols, indent=2))
    out["labels"].write_text(json.dumps(label_cols, indent=2))
    out["info"].write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_samples": int(len(combined)),
                "n_sled_samples": int(len(sled_samples)),
                "n_trawl_samples": int(len(trawl_samples)),
                "feature_dim": int(features.shape[1]),
                "label_dim": int(labels.shape[1]),
                "top_k_labels": int(args.top_k_labels),
                "use_other_species_as_features": bool(args.use_other_species_as_features),
                "include_seagrass_columns": bool(args.include_seagrass_columns),
                "na_strategy": "Median imputation for numeric predictors plus per-feature missingness indicators; categorical predictors one-hot encoded with explicit Missing category; labels built as species presence from rows with species_abundance > 0.",
                "gear_feature": "gear_is_sled (1=sled, 0=trawl)",
                "files": {
                    "data": str(out["data"].name),
                    "train_idx": str(out["train"].name),
                    "val_idx": str(out["val"].name),
                    "test_idx": str(out["test"].name),
                },
            },
            indent=2,
        )
    )

    print("saved", out["data"])
    print("samples", len(combined), "features", features.shape[1], "labels", labels.shape[1])
    print("splits", len(train_idx), len(val_idx), len(test_idx))
    print("nan in output", bool(np.isnan(data).any()))


if __name__ == "__main__":
    main()
