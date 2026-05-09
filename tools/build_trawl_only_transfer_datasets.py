from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "nekton-seagrass" / "transfer_learning"

DATASETS = [
    "pretrain_groups11_seagrass",
    "finetune_fisheries6_seagrass",
    "finetune_fisheries8_seagrass",
]


def remap_subset_indices(selected_rows: np.ndarray, original_idx: np.ndarray) -> np.ndarray:
    selected_lookup = {int(old_idx): new_idx for new_idx, old_idx in enumerate(selected_rows.tolist())}
    subset = [selected_lookup[int(idx)] for idx in original_idx.tolist() if int(idx) in selected_lookup]
    return np.array(subset, dtype=np.int64)


def main() -> None:
    for stem in DATASETS:
        data_path = BASE_DIR / f"nekton_seagrass_{stem}_data.npy"
        train_path = BASE_DIR / f"nekton_seagrass_{stem}_train_idx.npy"
        val_path = BASE_DIR / f"nekton_seagrass_{stem}_val_idx.npy"
        test_path = BASE_DIR / f"nekton_seagrass_{stem}_test_idx.npy"
        label_path = BASE_DIR / f"label_columns_{stem}.json"
        feature_path = BASE_DIR / f"feature_columns_{stem}.json"

        data = np.load(data_path)
        train_idx = np.load(train_path)
        val_idx = np.load(val_path)
        test_idx = np.load(test_path)
        labels = json.loads(label_path.read_text())
        features = json.loads(feature_path.read_text())

        gear_feature_idx = features.index("gear_is_sled")
        gear_values = data[:, len(labels) + gear_feature_idx]
        gear_threshold = 0.5 * (gear_values.min() + gear_values.max())
        trawl_rows = np.where(gear_values <= gear_threshold)[0]

        subset_data = data[trawl_rows].astype(np.float32)
        subset_train = remap_subset_indices(trawl_rows, train_idx)
        subset_val = remap_subset_indices(trawl_rows, val_idx)
        subset_test = remap_subset_indices(trawl_rows, test_idx)

        out_stem = stem.replace("_seagrass", "_trawl_seagrass")
        np.save(BASE_DIR / f"nekton_seagrass_{out_stem}_data.npy", subset_data)
        np.save(BASE_DIR / f"nekton_seagrass_{out_stem}_train_idx.npy", subset_train)
        np.save(BASE_DIR / f"nekton_seagrass_{out_stem}_val_idx.npy", subset_val)
        np.save(BASE_DIR / f"nekton_seagrass_{out_stem}_test_idx.npy", subset_test)
        (BASE_DIR / f"label_columns_{out_stem}.json").write_text(json.dumps(labels, indent=2))
        (BASE_DIR / f"feature_columns_{out_stem}.json").write_text(json.dumps(features, indent=2))
        pd.DataFrame({"target": labels, "occupancy": subset_data[:, : len(labels)].sum(axis=0)}).to_csv(
            BASE_DIR / f"target_occupancy_{out_stem}.csv",
            index=False,
        )
        (BASE_DIR / f"processing_info_{out_stem}.json").write_text(
            json.dumps(
                {
                    "source_dataset": data_path.name,
                    "selection": "trawl_only",
                    "n_samples": int(len(subset_data)),
                    "label_dim": int(len(labels)),
                    "feature_dim": int(len(features)),
                    "labels": labels,
                    "train_size": int(len(subset_train)),
                    "val_size": int(len(subset_val)),
                    "test_size": int(len(subset_test)),
                    "gear_feature_index": int(gear_feature_idx),
                },
                indent=2,
            )
        )

        print(BASE_DIR / f"nekton_seagrass_{out_stem}_data.npy")
        print(
            json.dumps(
                {
                    "train_size": int(len(subset_train)),
                    "val_size": int(len(subset_val)),
                    "test_size": int(len(subset_test)),
                }
            )
        )


if __name__ == "__main__":
    main()
