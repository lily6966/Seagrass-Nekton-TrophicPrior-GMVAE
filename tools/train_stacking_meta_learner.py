import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import evals
from model import VAE
from utils import THRESHOLDS, get_feat, get_label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def infer_feature_columns_path(data_dir: str) -> Path | None:
    data_path = Path(data_dir).resolve()
    if not data_path.name.endswith("_data.npy"):
        return None
    base_name = data_path.name
    if base_name == "nekton_seagrass_data.npy":
        feature_name = "feature_columns.json"
    elif base_name.startswith("nekton_seagrass_"):
        suffix = base_name[len("nekton_seagrass_"):-len("_data.npy")]
        feature_name = f"feature_columns_{suffix}.json"
    else:
        return None
    feature_path = data_path.parent / feature_name
    return feature_path if feature_path.exists() else None


def get_gear_values(data_dir: str, label_dim: int) -> np.ndarray:
    data = np.load(data_dir)
    feature_path = infer_feature_columns_path(data_dir)
    if feature_path is None:
        raise ValueError(f"Could not infer feature columns for {data_dir}")
    features = json.loads(feature_path.read_text())
    gear_feature_idx = features.index("gear_is_sled")
    return data[:, label_dim + gear_feature_idx]


def load_model(checkpoint_path: str, label_dim: int, feature_dim: int, z_dim: int, latent_dim: int, emb_size: int):
    args = argparse.Namespace(
        label_dim=label_dim,
        feature_dim=feature_dim,
        z_dim=z_dim,
        latent_dim=latent_dim,
        emb_size=emb_size,
        keep_prob=0.5,
        reg="gmvae",
        gear_residualize=False,
        gear_interaction_design=False,
        gear_feature_idx=None,
        meta_offset=0,
    )
    model = VAE(args).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_probabilities(model, data, indices, label_dim, feature_dim, batch_size):
    outputs = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        feats = torch.from_numpy(get_feat(data, batch_idx, 0, label_dim, feature_dim)).float().to(device)
        labels = torch.from_numpy(get_label(data, batch_idx, 0, label_dim)).float().to(device)
        dummy = torch.zeros(label_dim, device=device)
        dummy[0] = 1.0
        mask = labels.sum(dim=1) > 0
        if not mask.all():
            labels = labels.clone()
            labels[~mask] = dummy
        with torch.no_grad():
            pred = model(labels, feats)["feat_out"]
        outputs.append(torch.sigmoid(pred).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, label_dim), dtype=np.float32)


def fit_species_meta(train_x, train_y):
    if np.unique(train_y).size < 2:
        return None
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(train_x, train_y.astype(int))
    return model


def select_best_threshold(pred, true):
    best_threshold = THRESHOLDS[0]
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        pred_label = (pred >= threshold).astype(np.int32)
        true_label = true.astype(np.int32)
        score = f1_score(true_label, pred_label, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1


def compute_best_metrics(predictions, labels):
    metric_names = ["ACC", "HA", "ebF1", "miF1", "maF1", "p_at_1", "p_at_3", "p_at_5"]
    best = None
    best_threshold = None
    for threshold in THRESHOLDS:
        metrics = evals.compute_metrics(predictions, labels, threshold, all_metrics=True)
        if best is None or metrics["maF1"] > best["maF1"]:
            best = {name: metrics[name] for name in metric_names}
            best_threshold = threshold
    return best_threshold, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gear_source_data_dir", required=True)
    parser.add_argument("--train_idx", required=True)
    parser.add_argument("--valid_idx", required=True)
    parser.add_argument("--test_idx", required=True)
    parser.add_argument("--label_dim", type=int, default=8)
    parser.add_argument("--feature_dim", type=int, default=509)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sled_checkpoint", required=True)
    parser.add_argument("--trawl_checkpoint", required=True)
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data = np.load(args.data_dir)
    train_idx = np.load(args.train_idx)
    valid_idx = np.load(args.valid_idx)
    test_idx = np.load(args.test_idx)
    label_names = json.loads(Path(args.label_json).read_text())

    gear_values = get_gear_values(args.gear_source_data_dir, args.label_dim)
    gear_threshold = 0.5 * (gear_values.min() + gear_values.max())

    sled_model = load_model(args.sled_checkpoint, args.label_dim, args.feature_dim, args.z_dim, args.latent_dim, args.emb_size)
    trawl_model = load_model(args.trawl_checkpoint, args.label_dim, args.feature_dim, args.z_dim, args.latent_dim, args.emb_size)

    valid_sled_prob = predict_probabilities(sled_model, data, valid_idx, args.label_dim, args.feature_dim, args.batch_size)
    valid_trawl_prob = predict_probabilities(trawl_model, data, valid_idx, args.label_dim, args.feature_dim, args.batch_size)
    test_sled_prob = predict_probabilities(sled_model, data, test_idx, args.label_dim, args.feature_dim, args.batch_size)
    test_trawl_prob = predict_probabilities(trawl_model, data, test_idx, args.label_dim, args.feature_dim, args.batch_size)

    valid_y = get_label(data, valid_idx, 0, args.label_dim)
    test_y = get_label(data, test_idx, 0, args.label_dim)
    valid_gear = (gear_values[valid_idx] > gear_threshold).astype(np.float32)
    test_gear = (gear_values[test_idx] > gear_threshold).astype(np.float32)

    valid_meta = np.stack([valid_sled_prob, valid_trawl_prob, np.repeat(valid_gear[:, None], args.label_dim, axis=1)], axis=-1)
    test_meta = np.stack([test_sled_prob, test_trawl_prob, np.repeat(test_gear[:, None], args.label_dim, axis=1)], axis=-1)

    ensemble_valid = np.zeros_like(valid_y, dtype=np.float32)
    ensemble_test = np.zeros_like(test_y, dtype=np.float32)
    species_rows = []
    for label_idx, label_name in enumerate(label_names):
        x_valid = valid_meta[:, label_idx, :]
        x_test = test_meta[:, label_idx, :]
        y_valid = valid_y[:, label_idx]
        y_test = test_y[:, label_idx]
        meta_model = fit_species_meta(x_valid, y_valid)
        if meta_model is None:
            if y_valid.sum() == 0:
                ensemble_valid[:, label_idx] = 0.0
                ensemble_test[:, label_idx] = 0.0
            else:
                ensemble_valid[:, label_idx] = valid_sled_prob[:, label_idx]
                ensemble_test[:, label_idx] = test_sled_prob[:, label_idx]
            threshold, best_valid_f1 = select_best_threshold(ensemble_valid[:, label_idx], y_valid)
            test_metrics = {
                "maF1": f1_score(y_test.astype(np.int32), (ensemble_test[:, label_idx] >= threshold).astype(np.int32), zero_division=0)
            }
            species_rows.append({
                "label": label_name,
                "meta_model": "fallback",
                "selected_threshold": threshold,
                "valid_best_f1": best_valid_f1,
                "test_f1": test_metrics["maF1"],
            })
            continue

        ensemble_valid[:, label_idx] = meta_model.predict_proba(x_valid)[:, 1]
        ensemble_test[:, label_idx] = meta_model.predict_proba(x_test)[:, 1]
        threshold, best_valid_f1 = select_best_threshold(ensemble_valid[:, label_idx], y_valid)
        test_metrics = {
            "maF1": f1_score(y_test.astype(np.int32), (ensemble_test[:, label_idx] >= threshold).astype(np.int32), zero_division=0)
        }
        species_rows.append({
            "label": label_name,
            "meta_model": "logistic_regression",
            "selected_threshold": threshold,
            "valid_best_f1": best_valid_f1,
            "test_f1": test_metrics["maF1"],
            "intercept": float(meta_model.intercept_[0]),
            "coef_sled_model": float(meta_model.coef_[0][0]),
            "coef_trawl_model": float(meta_model.coef_[0][1]),
            "coef_gear_is_sled": float(meta_model.coef_[0][2]),
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = []
    for model_name, valid_prob, test_prob in [
        ("sled_base", valid_sled_prob, test_sled_prob),
        ("trawl_base", valid_trawl_prob, test_trawl_prob),
        ("stacked_ensemble", ensemble_valid, ensemble_test),
    ]:
        valid_threshold, valid_metrics = compute_best_metrics(valid_prob, valid_y)
        test_threshold, test_metrics = compute_best_metrics(test_prob, test_y)
        base_rows.append({
            "model": model_name,
            "valid_threshold": valid_threshold,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "test_threshold": test_threshold,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

    pd.DataFrame(base_rows).to_csv(output_dir / "stacking_summary_metrics.csv", index=False)
    pd.DataFrame(species_rows).to_csv(output_dir / "stacking_species_thresholds.csv", index=False)
    np.save(output_dir / "ensemble_valid_prob.npy", ensemble_valid)
    np.save(output_dir / "ensemble_test_prob.npy", ensemble_test)
    print(output_dir / "stacking_summary_metrics.csv")
    print(output_dir / "stacking_species_thresholds.csv")


if __name__ == "__main__":
    main()
