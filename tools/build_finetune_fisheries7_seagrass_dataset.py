from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "nekton-seagrass" / "transfer_learning"

SOURCE_STEM = "finetune_fisheries8_seagrass"
TARGET_STEM = "finetune_fisheries7_seagrass"
DROP_SPECIES = {"Penaeus setiferus"}


def main() -> None:
    data = np.load(BASE_DIR / f"nekton_seagrass_{SOURCE_STEM}_data.npy")
    train_idx = np.load(BASE_DIR / f"nekton_seagrass_{SOURCE_STEM}_train_idx.npy")
    val_idx = np.load(BASE_DIR / f"nekton_seagrass_{SOURCE_STEM}_val_idx.npy")
    test_idx = np.load(BASE_DIR / f"nekton_seagrass_{SOURCE_STEM}_test_idx.npy")
    labels = json.loads((BASE_DIR / f"label_columns_{SOURCE_STEM}.json").read_text())
    features = json.loads((BASE_DIR / f"feature_columns_{SOURCE_STEM}.json").read_text())
    species_to_group = json.loads((BASE_DIR / "species_to_pretrain_group_fisheries8_feedpos11_allspecies.json").read_text())

    keep_label_indices = [i for i, name in enumerate(labels) if name not in DROP_SPECIES]
    kept_labels = [labels[i] for i in keep_label_indices]
    reduced = np.concatenate([data[:, keep_label_indices], data[:, len(labels):]], axis=1).astype(np.float32)
    reduced_mapping = {label: species_to_group[label] for label in kept_labels if label in species_to_group}

    np.save(BASE_DIR / f"nekton_seagrass_{TARGET_STEM}_data.npy", reduced)
    np.save(BASE_DIR / f"nekton_seagrass_{TARGET_STEM}_train_idx.npy", train_idx)
    np.save(BASE_DIR / f"nekton_seagrass_{TARGET_STEM}_val_idx.npy", val_idx)
    np.save(BASE_DIR / f"nekton_seagrass_{TARGET_STEM}_test_idx.npy", test_idx)
    (BASE_DIR / f"label_columns_{TARGET_STEM}.json").write_text(json.dumps(kept_labels, indent=2))
    (BASE_DIR / f"feature_columns_{TARGET_STEM}.json").write_text(json.dumps(features, indent=2))
    (BASE_DIR / "species_to_pretrain_group_fisheries7_feedpos11_allspecies.json").write_text(
        json.dumps(reduced_mapping, indent=2)
    )
    pd.DataFrame({"target": kept_labels, "occupancy": reduced[:, : len(kept_labels)].sum(axis=0)}).to_csv(
        BASE_DIR / f"target_occupancy_{TARGET_STEM}.csv",
        index=False,
    )
    (BASE_DIR / f"processing_info_{TARGET_STEM}.json").write_text(
        json.dumps(
            {
                "source_dataset": f"nekton_seagrass_{SOURCE_STEM}_data.npy",
                "removed_species": sorted(DROP_SPECIES),
                "n_samples": int(len(reduced)),
                "label_dim": int(len(kept_labels)),
                "feature_dim": int(len(features)),
                "labels": kept_labels,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
            },
            indent=2,
        )
    )

    print(BASE_DIR / f"nekton_seagrass_{TARGET_STEM}_data.npy")
    print(BASE_DIR / f"label_columns_{TARGET_STEM}.json")
    print(BASE_DIR / "species_to_pretrain_group_fisheries7_feedpos11_allspecies.json")


if __name__ == "__main__":
    main()
