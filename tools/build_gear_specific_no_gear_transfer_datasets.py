from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "nekton-seagrass" / "transfer_learning"

PRETRAIN_STEM = "pretrain_feedpos11_allspecies_seagrass"
FINETUNE8_STEM = "finetune_fisheries8_seagrass"

DROP_SPECIES = {"Penaeus setiferus"}
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


def subset_by_gear(
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    label_dim: int,
    features: list[str],
    keep_sled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gear_idx = features.index("gear_is_sled")
    gear_values = data[:, label_dim + gear_idx]
    gear_threshold = 0.5 * (gear_values.min() + gear_values.max())
    if keep_sled:
        rows = np.where(gear_values > gear_threshold)[0]
    else:
        rows = np.where(gear_values <= gear_threshold)[0]

    subset_data = data[rows].astype(np.float32)
    subset_train = remap_subset_indices(rows, train_idx)
    subset_val = remap_subset_indices(rows, val_idx)
    subset_test = remap_subset_indices(rows, test_idx)
    return subset_data, subset_train, subset_val, subset_test


def drop_features(data: np.ndarray, label_dim: int, features: list[str]) -> tuple[np.ndarray, list[str]]:
    keep_feature_indices = [i for i, name in enumerate(features) if name not in DROP_FEATURES]
    kept_features = [features[i] for i in keep_feature_indices]
    feature_block = data[:, label_dim:]
    reduced = np.concatenate([data[:, :label_dim], feature_block[:, keep_feature_indices]], axis=1).astype(np.float32)
    return reduced, kept_features


def drop_finetune_labels(data: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    keep_label_indices = [i for i, name in enumerate(labels) if name not in DROP_SPECIES]
    kept_labels = [labels[i] for i in keep_label_indices]
    reduced = np.concatenate([data[:, keep_label_indices], data[:, len(labels):]], axis=1).astype(np.float32)
    return reduced, kept_labels


def save_dataset(
    stem: str,
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
    features: list[str],
    selection: str,
    source_stem: str,
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
    pretrain_data, pretrain_train, pretrain_val, pretrain_test, pretrain_labels, pretrain_features = load_dataset(PRETRAIN_STEM)
    finetune_data, finetune_train, finetune_val, finetune_test, finetune_labels, finetune_features = load_dataset(FINETUNE8_STEM)

    for gear_name, keep_sled in [("trawl", False), ("sled", True)]:
        pre_data, pre_train, pre_val, pre_test = subset_by_gear(
            pretrain_data, pretrain_train, pretrain_val, pretrain_test, len(pretrain_labels), pretrain_features, keep_sled
        )
        pre_data, pre_features = drop_features(pre_data, len(pretrain_labels), pretrain_features)
        pre_stem = f"pretrain_feedpos11_allspecies_{gear_name}_nogear_seagrass"
        save_dataset(
            pre_stem,
            pre_data,
            pre_train,
            pre_val,
            pre_test,
            pretrain_labels,
            pre_features,
            f"{gear_name}_only_no_gear_feature",
            PRETRAIN_STEM,
        )

        fin_data, fin_train, fin_val, fin_test = subset_by_gear(
            finetune_data, finetune_train, finetune_val, finetune_test, len(finetune_labels), finetune_features, keep_sled
        )
        fin_data, fin_labels_7 = drop_finetune_labels(fin_data, finetune_labels)
        fin_data, fin_features = drop_features(fin_data, len(fin_labels_7), finetune_features)
        fin_stem = f"finetune_fisheries7_{gear_name}_nogear_seagrass"
        save_dataset(
            fin_stem,
            fin_data,
            fin_train,
            fin_val,
            fin_test,
            fin_labels_7,
            fin_features,
            f"{gear_name}_only_no_gear_feature",
            FINETUNE8_STEM,
        )

        print(BASE_DIR / f"nekton_seagrass_{pre_stem}_data.npy")
        print(json.dumps({"train_size": int(len(pre_train)), "val_size": int(len(pre_val)), "test_size": int(len(pre_test))}))
        print(BASE_DIR / f"nekton_seagrass_{fin_stem}_data.npy")
        print(json.dumps({"train_size": int(len(fin_train)), "val_size": int(len(fin_val)), "test_size": int(len(fin_test))}))


if __name__ == "__main__":
    main()
