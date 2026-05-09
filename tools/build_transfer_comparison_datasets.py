from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nekton-seagrass"
OUT_DIR = DATA_DIR / "transfer_learning"

BASE_DATA = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_data.npy"
BASE_TRAIN = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_train_idx.npy"
BASE_VAL = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_val_idx.npy"
BASE_TEST = DATA_DIR / "nekton_seagrass_fisheries6_plus_feedposlabels_seagrass_test_idx.npy"
BASE_LABELS = DATA_DIR / "label_columns_fisheries6_plus_feedposlabels_seagrass.json"
BASE_FEATURES = DATA_DIR / "feature_columns_fisheries6_plus_feedposlabels_seagrass.json"

FULL_SPECIES_DATA = DATA_DIR / "nekton_seagrass_seagrass_data.npy"
FULL_SPECIES_LABELS = DATA_DIR / "label_columns_seagrass.json"
FULL_SPECIES_FEATURES = DATA_DIR / "feature_columns_seagrass.json"

FOCAL_SPECIES = [
    "Penaeus aztecus",
    "Penaeus duorarum",
    "Callinectes sapidus",
    "Cynoscion nebulosus",
    "Lutjanus griseus",
    "Lutjanus synagris",
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

GROUPS_10 = [
    "omnivore__demersal",
    "carnivore__demersal",
    "carnivore__epifaunal",
    "detritivore__epifaunal",
    "detritivore__demersal",
    "planktivore__epifaunal",
    "invertivore__demersal",
    "carnivore__pelagic",
    "planktivore__pelagic",
    "piscivore__pelagic",
]

GROUPS_11 = GROUPS_10 + [
    "piscivore__demersal",
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

SPECIES_TO_PRETRAIN_GROUP_FISHERIES8_GROUPS10 = {
    species: (group if group in GROUPS_10 else None)
    for species, group in SPECIES_TO_PRETRAIN_GROUP_FISHERIES8.items()
}

SPECIES_TO_PRETRAIN_GROUP_FISHERIES8_GROUPS11 = {
    species: (group if group in GROUPS_11 else None)
    for species, group in SPECIES_TO_PRETRAIN_GROUP_FISHERIES8.items()
}


def save_dataset(
    suffix: str,
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
    features: list[str],
    source_dataset: str,
) -> None:
    stem = f"nekton_seagrass_{suffix}"
    np.save(OUT_DIR / f"{stem}_data.npy", data.astype(np.float32))
    np.save(OUT_DIR / f"{stem}_train_idx.npy", train_idx)
    np.save(OUT_DIR / f"{stem}_val_idx.npy", val_idx)
    np.save(OUT_DIR / f"{stem}_test_idx.npy", test_idx)
    (OUT_DIR / f"label_columns_{suffix}.json").write_text(json.dumps(labels, indent=2))
    (OUT_DIR / f"feature_columns_{suffix}.json").write_text(json.dumps(features, indent=2))
    pd.DataFrame({"target": labels, "occupancy": data[:, : len(labels)].sum(axis=0)}).to_csv(
        OUT_DIR / f"target_occupancy_{suffix}.csv",
        index=False,
    )
    (OUT_DIR / f"processing_info_{suffix}.json").write_text(
        json.dumps(
            {
                "source_dataset": source_dataset,
                "n_samples": int(data.shape[0]),
                "label_dim": int(len(labels)),
                "feature_dim": int(len(features)),
                "labels": labels,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
            },
            indent=2,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_data = np.load(BASE_DATA)
    base_train = np.load(BASE_TRAIN)
    base_val = np.load(BASE_VAL)
    base_test = np.load(BASE_TEST)
    base_labels = json.loads(BASE_LABELS.read_text())
    base_features = json.loads(BASE_FEATURES.read_text())
    full_species_data = np.load(FULL_SPECIES_DATA)
    full_species_labels = json.loads(FULL_SPECIES_LABELS.read_text())
    full_species_features = json.loads(FULL_SPECIES_FEATURES.read_text())

    label_to_idx = {label: i for i, label in enumerate(base_labels)}
    full_species_label_to_idx = {label: i for i, label in enumerate(full_species_labels)}
    feature_start = len(base_labels)
    full_species_feature_start = len(full_species_labels)

    joint_16_raw = FOCAL_SPECIES + GROUPS_10
    joint_16_base = FOCAL_SPECIES + [f"niche_label__{g}" for g in GROUPS_10]
    pretrain_10_base = [f"niche_label__{g}" for g in GROUPS_10]
    pretrain_11_base = [f"niche_label__{g}" for g in GROUPS_11]

    datasets = [
        ("joint16_targets_seagrass", joint_16_base, joint_16_raw),
        ("pretrain_groups10_seagrass", pretrain_10_base, GROUPS_10),
        ("pretrain_groups11_seagrass", pretrain_11_base, GROUPS_11),
        ("finetune_fisheries6_seagrass", FOCAL_SPECIES, FOCAL_SPECIES),
        ("finetune_fisheries8_seagrass", FOCAL_SPECIES_8, FOCAL_SPECIES_8),
    ]

    for suffix, labels_in_base, labels_out in datasets:
        if suffix == "finetune_fisheries8_seagrass":
            idxs = [full_species_label_to_idx[label] for label in labels_in_base]
            data = np.concatenate(
                [full_species_data[:, idxs], full_species_data[:, full_species_feature_start:]],
                axis=1,
            )
            features = full_species_features
            source_dataset = FULL_SPECIES_DATA.name
        else:
            idxs = [label_to_idx[label] for label in labels_in_base]
            data = np.concatenate([base_data[:, idxs], base_data[:, feature_start:]], axis=1)
            features = base_features
            source_dataset = BASE_DATA.name
        save_dataset(
            suffix=suffix,
            data=data,
            train_idx=base_train,
            val_idx=base_val,
            test_idx=base_test,
            labels=labels_out,
            features=features,
            source_dataset=source_dataset,
        )
        print(OUT_DIR / f"nekton_seagrass_{suffix}_data.npy")
        print(OUT_DIR / f"label_columns_{suffix}.json")
        print(OUT_DIR / f"target_occupancy_{suffix}.csv")

    (OUT_DIR / "species_to_pretrain_group_fisheries8.json").write_text(
        json.dumps(SPECIES_TO_PRETRAIN_GROUP_FISHERIES8_GROUPS10, indent=2)
    )
    (OUT_DIR / "species_to_pretrain_group_fisheries8_groups11.json").write_text(
        json.dumps(SPECIES_TO_PRETRAIN_GROUP_FISHERIES8_GROUPS11, indent=2)
    )
    print(OUT_DIR / "species_to_pretrain_group_fisheries8.json")
    print(OUT_DIR / "species_to_pretrain_group_fisheries8_groups11.json")


if __name__ == "__main__":
    main()
