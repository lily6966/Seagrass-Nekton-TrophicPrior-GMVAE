from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "nekton-seagrass" / "transfer_learning"

DATASETS = [
    ("pretrain_groups11_seagrass", "pretrain_groups11"),
    ("finetune_fisheries8_seagrass", "finetune_fisheries8"),
]

DROP_FEATURES = {"gear_is_sled", "gear_is_sled__missing"}


def remap_subset_indices(selected_rows: np.ndarray, original_idx: np.ndarray) -> np.ndarray:
    selected_lookup = {int(old_idx): new_idx for new_idx, old_idx in enumerate(selected_rows.tolist())}
    subset = [selected_lookup[int(idx)] for idx in original_idx.tolist() if int(idx) in selected_lookup]
    return np.array(subset, dtype=np.int64)


def load_dataset(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    data = np.load(BASE_DIR / f"nekton_seagrass_{stem}_data.npy")
    train_idx = np.load(BASE_DIR / f"nekton_seagrass_{stem}_train_idx.npy")
    val_idx = np.load(BASE_DIR / f"nekton_seagrass_{stem}_val_idx.npy")
    test_idx = np.load(BASE_DIR / f"nekton_seagrass_{stem}_test_idx.npy")
    labels = json.loads((BASE_DIR / f"label_columns_{stem}.json").read_text())
    features = json.loads((BASE_DIR / f"feature_columns_{stem}.json").read_text())
    return data, train_idx, val_idx, test_idx, labels, features


def drop_features(data: np.ndarray, label_dim: int, features: list[str]) -> tuple[np.ndarray, list[str]]:
    keep_feature_indices = [i for i, name in enumerate(features) if name not in DROP_FEATURES]
    kept_features = [features[i] for i in keep_feature_indices]
    feature_block = data[:, label_dim:]
    reduced = np.concatenate([data[:, :label_dim], feature_block[:, keep_feature_indices]], axis=1).astype(np.float32)
    return reduced, kept_features


def save_dataset(
    stem: str,
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
    features: list[str],
    source_stem: str,
    selection: str,
) -> None:
    np.save(BASE_DIR / f"nekton_seagrass_{stem}_data.npy", data)
    np.save(BASE_DIR / f"nekton_seagrass_{stem}_train_idx.npy", train_idx)
    np.save(BASE_DIR / f"nekton_seagrass_{stem}_val_idx.npy", val_idx)
    np.save(BASE_DIR / f"nekton_seagrass_{stem}_test_idx.npy", test_idx)
    (BASE_DIR / f"label_columns_{stem}.json").write_text(json.dumps(labels, indent=2))
    (BASE_DIR / f"feature_columns_{stem}.json").write_text(json.dumps(features, indent=2))
    pd.DataFrame({"target": labels, "occupancy": data[:, : len(labels)].sum(axis=0)}).to_csv(
        BASE_DIR / f"target_occupancy_{stem}.csv",
        index=False,
    )
    (BASE_DIR / f"processing_info_{stem}.json").write_text(
        json.dumps(
            {
                "source_dataset": f"nekton_seagrass_{source_stem}_data.npy",
                "selection": selection,
                "removed_features": sorted(DROP_FEATURES),
                "n_samples": int(len(data)),
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
    for source_stem, short_name in DATASETS:
        data, train_idx, val_idx, test_idx, labels, features = load_dataset(source_stem)
        gear_feature_idx = features.index("gear_is_sled")
        gear_values = data[:, len(labels) + gear_feature_idx]
        gear_threshold = 0.5 * (gear_values.min() + gear_values.max())

        selections = {
            "trawl_only_without_gear_feature": np.where(gear_values <= gear_threshold)[0],
            "sled_only_without_gear_feature": np.where(gear_values > gear_threshold)[0],
        }
        for selection_name, rows in selections.items():
            subset_data = data[rows].astype(np.float32)
            subset_data, subset_features = drop_features(subset_data, len(labels), features)
            subset_train = remap_subset_indices(rows, train_idx)
            subset_val = remap_subset_indices(rows, val_idx)
            subset_test = remap_subset_indices(rows, test_idx)
            gear_prefix = "trawl" if "trawl" in selection_name else "sled"
            out_stem = f"{short_name}_{gear_prefix}_nogear_matchvae261_seagrass"
            save_dataset(
                out_stem,
                subset_data,
                subset_train,
                subset_val,
                subset_test,
                labels,
                subset_features,
                source_stem,
                selection_name,
            )
            print(BASE_DIR / f"nekton_seagrass_{out_stem}_data.npy")
            print(
                json.dumps(
                    {
                        "train_size": int(len(subset_train)),
                        "val_size": int(len(subset_val)),
                        "test_size": int(len(subset_test)),
                        "feature_dim": int(len(subset_features)),
                    }
                )
            )


if __name__ == "__main__":
    main()
