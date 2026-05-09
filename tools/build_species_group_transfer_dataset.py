from pathlib import Path
import json

import numpy as np
import pandas as pd

from process_nekton_seagrass import (
    DATA_DIR,
    SLED_CSV,
    TRAWL_CSV,
    SEED,
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

TRANSFER_GROUPS = [
    "detritivore__demersal",
    "piscivore__demersal",
]

WORKBOOK = DATA_DIR / "species_trophic_levels_master.xlsx"


def load_workbook_mapping() -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    df = pd.read_excel(WORKBOOK).copy()
    df["species_scientific"] = df["species_scientific"].fillna("").astype(str).str.strip()
    df["feeding_type"] = df["feeding_type"].fillna("").astype(str).str.strip().str.lower()
    df["position"] = df["position"].fillna("").astype(str).str.strip().str.lower()
    df["trophic_group"] = df["feeding_type"] + "__" + df["position"]
    focal_df = df[df["species_scientific"].isin(FOCAL_LABELS)].copy()
    focal_map = dict(zip(focal_df["species_scientific"], focal_df["trophic_group"]))
    group_species = df[df["trophic_group"].isin(TRANSFER_GROUPS)]["species_scientific"].dropna().astype(str).str.strip()
    group_species = group_species[group_species != ""]
    group_map = (
        df[df["species_scientific"].isin(group_species)]
        .drop_duplicates("species_scientific")
        .loc[:, ["species_scientific", "trophic_group", "source"]]
        .rename(columns={"species_scientific": "species"})
        .sort_values(["trophic_group", "species"])
        .reset_index(drop=True)
    )
    return df, focal_map, group_map


def build_transfer_group_labels(raw_df: pd.DataFrame, sample_id_col: str, group_species: set[str]) -> pd.DataFrame:
    df = raw_df.copy()
    df["sample_id"] = df[sample_id_col].astype(str)
    df["species_label"] = choose_species_label(df)
    df["species_abundance"] = pd.to_numeric(df["species_abundance"], errors="coerce").fillna(0.0)
    df = df[df["species_label"].isin(group_species)].copy()
    df = df[df["species_abundance"] > 0].copy()
    df["present"] = 1.0

    workbook = pd.read_excel(WORKBOOK).copy()
    workbook["species_scientific"] = workbook["species_scientific"].fillna("").astype(str).str.strip()
    workbook["feeding_type"] = workbook["feeding_type"].fillna("").astype(str).str.strip().str.lower()
    workbook["position"] = workbook["position"].fillna("").astype(str).str.strip().str.lower()
    workbook["trophic_group"] = workbook["feeding_type"] + "__" + workbook["position"]
    species_to_group = (
        workbook.drop_duplicates("species_scientific")
        .set_index("species_scientific")["trophic_group"]
        .to_dict()
    )
    df["trophic_group"] = df["species_label"].map(species_to_group)
    df = df[df["trophic_group"].isin(TRANSFER_GROUPS)]

    grouped = (
        df.pivot_table(
            index="sample_id",
            columns="trophic_group",
            values="present",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(columns=TRANSFER_GROUPS, fill_value=0.0)
        .reset_index()
    )
    rename = {c: f"transfer_group__{c}" for c in TRANSFER_GROUPS}
    return grouped.rename(columns=rename)


def main() -> None:
    _, focal_map, group_map = load_workbook_mapping()
    if set(focal_map) != set(FOCAL_LABELS):
        missing = sorted(set(FOCAL_LABELS) - set(focal_map))
        raise ValueError(f"Missing focal species in workbook mapping: {missing}")

    sled_raw = load_frame(SLED_CSV)
    trawl_raw = load_frame(TRAWL_CSV)

    sample_table = pd.concat(
        [
            build_sample_table(sled_raw, "pull_id", 1),
            build_sample_table(trawl_raw, "tow_id", 0),
        ],
        ignore_index=True,
    )
    sample_table["sample_id"] = sample_table["sample_id"].astype(str)

    focal_labels = sample_table[["sample_id"] + FOCAL_LABELS].copy()
    group_species = set(group_map["species"])
    group_labels = pd.concat(
        [
            build_transfer_group_labels(sled_raw, "pull_id", group_species),
            build_transfer_group_labels(trawl_raw, "tow_id", group_species),
        ],
        ignore_index=True,
    )
    merged = sample_table.merge(group_labels, on="sample_id", how="left")
    transfer_cols = [f"transfer_group__{group}" for group in TRANSFER_GROUPS]
    label_cols = FOCAL_LABELS + transfer_cols
    for col in transfer_cols:
        merged[col] = merged[col].fillna(0.0)

    drop_cols = ["sample_id"] + label_cols
    feature_matrix, feature_cols = build_feature_matrix(merged, drop_cols=drop_cols)

    data = np.concatenate(
        [
            merged[label_cols].to_numpy(dtype=np.float32),
            feature_matrix.astype(np.float32),
        ],
        axis=1,
    )
    train_idx, val_idx, test_idx = make_splits(len(merged))

    suffix = "_fisheries6_transfergroups_seagrass"
    stem = f"nekton_seagrass{suffix}"
    paths = {
        "data": DATA_DIR / f"{stem}_data.npy",
        "train": DATA_DIR / f"{stem}_train_idx.npy",
        "val": DATA_DIR / f"{stem}_val_idx.npy",
        "test": DATA_DIR / f"{stem}_test_idx.npy",
        "features": DATA_DIR / f"feature_columns{suffix}.json",
        "labels": DATA_DIR / f"label_columns{suffix}.json",
        "info": DATA_DIR / f"processing_info{suffix}.json",
        "group_map": DATA_DIR / f"transfer_group_species_map{suffix}.csv",
        "occupancy": DATA_DIR / f"target_occupancy{suffix}.csv",
    }

    np.save(paths["data"], data)
    np.save(paths["train"], train_idx)
    np.save(paths["val"], val_idx)
    np.save(paths["test"], test_idx)
    paths["features"].write_text(json.dumps(feature_cols, indent=2))
    paths["labels"].write_text(json.dumps(label_cols, indent=2))
    group_map.to_csv(paths["group_map"], index=False)

    occupancy = merged[label_cols].sum(0).rename("occupancy").reset_index().rename(columns={"index": "target"})
    occupancy.to_csv(paths["occupancy"], index=False)

    info = {
        "seed": SEED,
        "n_samples": int(data.shape[0]),
        "feature_dim": int(len(feature_cols)),
        "label_dim": int(len(label_cols)),
        "focal_species": FOCAL_LABELS,
        "transfer_groups": TRANSFER_GROUPS,
        "focal_species_to_group": focal_map,
        "group_definition": "Auxiliary transfer-group labels include all species in the workbook assigned to the same feeding_type__position group, including the focal species.",
        "files": {k: str(v.name) for k, v in paths.items()},
    }
    paths["info"].write_text(json.dumps(info, indent=2))

    print(paths["data"])
    print(paths["labels"])
    print(paths["group_map"])
    print(paths["occupancy"])


if __name__ == "__main__":
    main()
