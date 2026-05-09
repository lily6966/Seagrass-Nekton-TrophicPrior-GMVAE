from pathlib import Path
import json

import numpy as np
import pandas as pd

from process_nekton_seagrass_abundance import (
    DATA_DIR,
    SLED_CSV,
    TRAWL_CSV,
    build_feature_matrix,
    build_sample_table,
    choose_species_label,
    load_frame,
    make_splits,
)


FOCAL_LABELS = [
    "Penaeus aztecus",
    "Penaeus duorarum",
    "Callinectes sapidus",
    "Cynoscion nebulosus",
    "Lutjanus griseus",
    "Lutjanus synagris",
]

LOOKUP_CSV = DATA_DIR / "nekton_grouping_lookup.csv"
ALL_LABELS_JSON = DATA_DIR / "label_columns_seagrass.json"


def build_group_abundance(raw_df: pd.DataFrame, sample_id_col: str, lookup: pd.DataFrame, group_names: list[str]) -> pd.DataFrame:
    df = raw_df.copy()
    df["sample_id"] = df[sample_id_col].astype(str)
    df["species_label"] = choose_species_label(df)
    df["species_abundance"] = pd.to_numeric(df["species_abundance"], errors="coerce").fillna(0.0)

    grouped = df.merge(lookup, left_on="species_label", right_on="taxon", how="left")
    grouped["assigned_group"] = grouped["assigned_group"].fillna("unknown")
    grouped = grouped[~grouped["species_label"].isin(FOCAL_LABELS)].copy()

    out = (
        grouped.groupby(["sample_id", "assigned_group"])["species_abundance"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=group_names, fill_value=0.0)
        .reset_index()
    )
    rename = {g: f"group_abundance__{g}" for g in group_names}
    return out.rename(columns=rename)


def main() -> None:
    lookup = pd.read_csv(LOOKUP_CSV)
    lookup = lookup.copy()
    lookup["assigned_group"] = lookup["assigned_group"].fillna("unknown")
    group_names = sorted(lookup["assigned_group"].unique().tolist())

    sled = load_frame(SLED_CSV)
    trawl = load_frame(TRAWL_CSV)

    all_species_cols = json.loads(ALL_LABELS_JSON.read_text())
    sled_samples = build_sample_table(sled, "pull_id", 1, all_species_cols)
    trawl_samples = build_sample_table(trawl, "tow_id", 0, all_species_cols)
    combined = pd.concat([sled_samples, trawl_samples], ignore_index=True, sort=False)
    combined[all_species_cols] = combined[all_species_cols].fillna(0.0)

    group_sled = build_group_abundance(sled, "pull_id", lookup, group_names)
    group_trawl = build_group_abundance(trawl, "tow_id", lookup, group_names)
    group_labels = pd.concat([group_sled, group_trawl], ignore_index=True, sort=False)

    combined = combined.merge(group_labels, on="sample_id", how="left")
    group_label_cols = [f"group_abundance__{g}" for g in group_names]
    combined[group_label_cols] = combined[group_label_cols].fillna(0.0)

    label_cols = [c for c in FOCAL_LABELS if c in combined.columns] + group_label_cols
    labels_raw = combined[label_cols].astype(np.float64)
    labels_log = np.log1p(labels_raw.to_numpy())

    features, feature_cols = build_feature_matrix(combined, ["sample_id"] + label_cols + all_species_cols)
    data = np.concatenate([labels_log, features], axis=1)
    train_idx, val_idx, test_idx = make_splits(len(combined))

    suffix = "abundance_fisheries6_plus_feedpos_env_seagrass"
    np.save(DATA_DIR / f"nekton_seagrass_{suffix}_data.npy", data)
    np.save(DATA_DIR / f"nekton_seagrass_{suffix}_train_idx.npy", train_idx)
    np.save(DATA_DIR / f"nekton_seagrass_{suffix}_val_idx.npy", val_idx)
    np.save(DATA_DIR / f"nekton_seagrass_{suffix}_test_idx.npy", test_idx)
    (DATA_DIR / f"feature_columns_{suffix}.json").write_text(json.dumps(feature_cols, indent=2))
    (DATA_DIR / f"label_columns_{suffix}.json").write_text(json.dumps(label_cols, indent=2))
    (DATA_DIR / f"processing_info_{suffix}.json").write_text(
        json.dumps(
            {
                "n_samples": int(len(combined)),
                "label_dim": int(len(label_cols)),
                "feature_dim": int(len(feature_cols)),
                "focal_labels": FOCAL_LABELS,
                "group_names": group_names,
                "lookup_csv": LOOKUP_CSV.name,
                "label_transform": "log1p(raw species abundance or summed group abundance per sample)",
            },
            indent=2,
        )
    )

    print(DATA_DIR / f"nekton_seagrass_{suffix}_data.npy")
    print(DATA_DIR / f"label_columns_{suffix}.json")
    print(DATA_DIR / f"feature_columns_{suffix}.json")


if __name__ == "__main__":
    main()
