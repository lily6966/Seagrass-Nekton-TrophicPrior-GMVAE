from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "nekton-seagrass" / "transfer_learning"

PRETRAIN_STEM = "pretrain_groups11_seagrass"
FINETUNE_STEM = "finetune_fisheries8_seagrass"
DROP_FEATURES = {"gear_is_sled", "gear_is_sled__missing"}


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
                "selection": "mixed_gear_without_gear_feature",
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
    pretrain_data, pretrain_features = drop_features(pretrain_data, len(pretrain_labels), pretrain_features)
    save_dataset(
        "pretrain_groups11_nogear_seagrass",
        pretrain_data,
        pretrain_train,
        pretrain_val,
        pretrain_test,
        pretrain_labels,
        pretrain_features,
        PRETRAIN_STEM,
    )

    finetune_data, finetune_train, finetune_val, finetune_test, finetune_labels, finetune_features = load_dataset(FINETUNE_STEM)
    finetune_data, finetune_features = drop_features(finetune_data, len(finetune_labels), finetune_features)
    save_dataset(
        "finetune_fisheries8_nogear_seagrass",
        finetune_data,
        finetune_train,
        finetune_val,
        finetune_test,
        finetune_labels,
        finetune_features,
        FINETUNE_STEM,
    )

    print(BASE_DIR / "nekton_seagrass_pretrain_groups11_nogear_seagrass_data.npy")
    print(BASE_DIR / "nekton_seagrass_finetune_fisheries8_nogear_seagrass_data.npy")


if __name__ == "__main__":
    main()
