from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nekton-seagrass"
OUT_DIR = DATA_DIR / "transfer_learning"

FULL_SPECIES_DATA = DATA_DIR / "nekton_seagrass_seagrass_data.npy"
FULL_SPECIES_LABELS = DATA_DIR / "label_columns_seagrass.json"
FULL_SPECIES_FEATURES = DATA_DIR / "feature_columns_seagrass.json"
BASE_TRAIN = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_train_idx.npy"
BASE_VAL = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_val_idx.npy"
BASE_TEST = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_test_idx.npy"
WORKBOOK = DATA_DIR / "species_trophic_levels_master.xlsx"

GROUPS_11 = [
    "omnivore__demersal",
    "detritivore__demersal",
    "carnivore__demersal",
    "carnivore__epifaunal",
    "detritivore__epifaunal",
    "piscivore__demersal",
    "planktivore__epifaunal",
    "invertivore__demersal",
    "carnivore__pelagic",
    "planktivore__pelagic",
    "piscivore__pelagic",
]

FOCAL_SPECIES_8 = [
    "Penaeus aztecus",
    "Penaeus duorarum",
    "Penaeus setiferus",
    "Callinectes sapidus",
    "Sciaenops ocellatus",
    "Cynoscion nebulosus",
    "Lutjanus griseus",
    "Lutjanus synagris",
]

SPECIES_TO_PRETRAIN_GROUP_FISHERIES8 = {
    "Penaeus aztecus": "detritivore__demersal",
    "Penaeus duorarum": "detritivore__demersal",
    "Penaeus setiferus": "detritivore__demersal",
    "Callinectes sapidus": "detritivore__demersal",
    "Sciaenops ocellatus": "omnivore__demersal",
    "Cynoscion nebulosus": "piscivore__demersal",
    "Lutjanus griseus": "piscivore__demersal",
    "Lutjanus synagris": "piscivore__demersal",
}


def load_workbook_group_map() -> dict[str, str]:
    workbook = pd.read_excel(WORKBOOK).copy()
    workbook["species_scientific"] = workbook["species_scientific"].fillna("").astype(str).str.strip()
    workbook["feeding_type"] = workbook["feeding_type"].fillna("").astype(str).str.strip().str.lower()
    workbook["position"] = workbook["position"].fillna("").astype(str).str.strip().str.lower()
    workbook["trophic_group"] = workbook["feeding_type"] + "__" + workbook["position"]
    workbook = workbook[workbook["trophic_group"].isin(GROUPS_11)].copy()
    workbook = workbook[workbook["species_scientific"] != ""].drop_duplicates("species_scientific")
    return workbook.set_index("species_scientific")["trophic_group"].to_dict()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(FULL_SPECIES_DATA)
    train_idx = np.load(BASE_TRAIN)
    val_idx = np.load(BASE_VAL)
    test_idx = np.load(BASE_TEST)
    labels = json.loads(FULL_SPECIES_LABELS.read_text())
    features = json.loads(FULL_SPECIES_FEATURES.read_text())

    species_to_group = load_workbook_group_map()
    group_to_species = {group: [] for group in GROUPS_11}
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for species, group in species_to_group.items():
        if species in label_to_idx:
            group_to_species[group].append(species)

    missing_groups = [group for group, species in group_to_species.items() if not species]
    if missing_groups:
        raise ValueError(f"No species labels found for groups: {missing_groups}")

    group_arrays = []
    species_membership_rows = []
    for group in GROUPS_11:
        idxs = [label_to_idx[species] for species in sorted(group_to_species[group])]
        group_presence = data[:, idxs].max(axis=1, keepdims=True)
        group_arrays.append(group_presence)
        for species in sorted(group_to_species[group]):
            species_membership_rows.append({"species": species, "trophic_group": group})

    group_matrix = np.concatenate(group_arrays, axis=1).astype(np.float32)
    feature_start = len(labels)
    out_data = np.concatenate([group_matrix, data[:, feature_start:]], axis=1).astype(np.float32)

    suffix = "pretrain_feedpos11_allspecies_seagrass"
    stem = f"nekton_seagrass_{suffix}"

    np.save(OUT_DIR / f"{stem}_data.npy", out_data)
    np.save(OUT_DIR / f"{stem}_train_idx.npy", train_idx)
    np.save(OUT_DIR / f"{stem}_val_idx.npy", val_idx)
    np.save(OUT_DIR / f"{stem}_test_idx.npy", test_idx)
    (OUT_DIR / f"label_columns_{suffix}.json").write_text(json.dumps(GROUPS_11, indent=2))
    (OUT_DIR / f"feature_columns_{suffix}.json").write_text(json.dumps(features, indent=2))
    pd.DataFrame({"target": GROUPS_11, "occupancy": group_matrix.sum(axis=0).astype(int)}).to_csv(
        OUT_DIR / f"target_occupancy_{suffix}.csv",
        index=False,
    )
    pd.DataFrame(species_membership_rows).to_csv(
        OUT_DIR / f"group_species_{suffix}.csv",
        index=False,
    )
    (OUT_DIR / "species_to_pretrain_group_fisheries8_feedpos11_allspecies.json").write_text(
        json.dumps(SPECIES_TO_PRETRAIN_GROUP_FISHERIES8, indent=2)
    )
    (OUT_DIR / f"processing_info_{suffix}.json").write_text(
        json.dumps(
            {
                "source_dataset": FULL_SPECIES_DATA.name,
                "source_labels": FULL_SPECIES_LABELS.name,
                "group_workbook": WORKBOOK.name,
                "group_definition": "Workbook-based feeding_type__position presence groups aggregated across all sampled species, including focal species.",
                "n_samples": int(out_data.shape[0]),
                "label_dim": int(len(GROUPS_11)),
                "feature_dim": int(len(features)),
                "labels": GROUPS_11,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
                "focal_species_8": FOCAL_SPECIES_8,
            },
            indent=2,
        )
    )

    print(OUT_DIR / f"{stem}_data.npy")
    print(OUT_DIR / f"label_columns_{suffix}.json")
    print(OUT_DIR / f"target_occupancy_{suffix}.csv")
    print(OUT_DIR / f"group_species_{suffix}.csv")
    print(OUT_DIR / "species_to_pretrain_group_fisheries8_feedpos11_allspecies.json")


if __name__ == "__main__":
    main()
