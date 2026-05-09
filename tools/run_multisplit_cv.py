import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evals
from model import VAE
from utils import THRESHOLDS, get_feat, get_label


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated multi-split CV for GMVAE.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--latent-dim", default=32, type=int)
    parser.add_argument("--z-dim", default=32, type=int)
    parser.add_argument("--emb-size", default=256, type=int)
    parser.add_argument("--keep-prob", default=0.5, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n-splits", default=3, type=int)
    parser.add_argument("--n-repeats", default=1, type=int)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--learning-rate", default=0.0005, type=float)
    parser.add_argument("--max-epoch", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--check-freq", default=15, type=int)
    parser.add_argument("--use-balanced-batches", action="store_true")
    return parser.parse_args()


def make_repeat_splits(n_samples: int, n_splits: int, n_repeats: int, val_ratio: float, seed: int):
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
            rng = np.random.default_rng(seed + repeat * 100 + fold)
            trainval_idx = np.array(trainval_idx)
            rng.shuffle(trainval_idx)
            n_val = max(1, int(round(len(trainval_idx) * val_ratio)))
            val_idx = np.sort(trainval_idx[:n_val])
            train_idx = np.sort(trainval_idx[n_val:])
            test_idx = np.sort(np.array(test_idx))
            yield repeat, fold, train_idx, val_idx, test_idx


def save_split(split_dir: Path, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": split_dir / "train_idx.npy",
        "val": split_dir / "val_idx.npy",
        "test": split_dir / "test_idx.npy",
    }
    np.save(paths["train"], train_idx)
    np.save(paths["val"], val_idx)
    np.save(paths["test"], test_idx)
    return paths


def param_setting(args: argparse.Namespace) -> str:
    return "lr-{}_lr-decay_{:.2f}_lr-times_{:.1f}_nll-{:.2f}_l2-{:.2f}_c-{:.2f}".format(
        args.learning_rate, 0.5, 4.0, 0.5, 1.0, 0.0
    )


def run_training(args: argparse.Namespace, dataset_name: str, split_paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "--data_dir", args.data,
        "--train_idx", str(split_paths["train"]),
        "--valid_idx", str(split_paths["val"]),
        "--test_idx", str(split_paths["test"]),
        "--learning_rate", str(args.learning_rate),
        "--max_epoch", str(args.max_epoch),
        "--label_dim", str(args.label_dim),
        "--z_dim", str(args.z_dim),
        "--latent_dim", str(args.latent_dim),
        "--feature_dim", str(args.feature_dim),
        "--nll_coeff", "0.5",
        "--l2_coeff", "1.0",
        "--c_coeff", "0.0",
        "--batch_size", str(args.batch_size),
        "--dataset", dataset_name,
        "--lr_decay_ratio", "0.5",
        "--lr_decay_times", "4.0",
        "--check_freq", str(args.check_freq),
        "--keep_prob", str(args.keep_prob),
        "--mode", "train",
        "--emb_size", str(args.emb_size),
        "--reg", "gmvae",
        "--seed", str(args.seed),
        "--train_ratio", "1.0",
        "--T0", "25",
        "--eta_min", "2e-4",
        "--T_mult", "2",
    ]
    if args.use_balanced_batches:
        cmd.append("--balanced_batches")
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_model(feature_dim: int, label_dim: int, latent_dim: int, z_dim: int, emb_size: int, keep_prob: float, checkpoint: Path) -> VAE:
    model_args = SimpleNamespace(
        feature_dim=feature_dim,
        label_dim=label_dim,
        latent_dim=latent_dim,
        z_dim=z_dim,
        emb_size=emb_size,
        keep_prob=keep_prob,
    )
    model = VAE(model_args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def predict(model: VAE, x: np.ndarray, label_dim: int, batch_size: int = 64) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        dummy_label = torch.zeros((batch.shape[0], label_dim), device=device)
        dummy_label[:, 0] = 1.0
        with torch.no_grad():
            output = model(dummy_label, batch)
            prob = output["feat_out"].cpu().numpy()
        preds.append(prob)
    return np.concatenate(preds, axis=0)


def best_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    best = None
    for threshold in THRESHOLDS:
        metrics = evals.compute_metrics(y_score, y_true, threshold, all_metrics=True)
        if best is None:
            best = metrics
        else:
            for metric in ["ACC", "HA", "ebF1", "miF1", "maF1", "p_at_1"]:
                best[metric] = max(best[metric], metrics[metric])
    return best


def score_checkpoint(data: np.ndarray, idx: np.ndarray, checkpoint: Path, args: argparse.Namespace) -> dict[str, float]:
    y_true = get_label(data, idx, 0, args.label_dim)
    x = get_feat(data, idx, 0, args.label_dim, args.feature_dim)
    model = build_model(args.feature_dim, args.label_dim, args.latent_dim, args.z_dim, args.emb_size, args.keep_prob, checkpoint)
    y_score = predict(model, x, args.label_dim)
    return best_metrics(y_true, y_score)


def choose_best_checkpoint(model_dir: Path, data: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[Path, dict[str, float]]:
    checkpoints = sorted(model_dir.glob("vae-*"), key=lambda p: int(p.name.split("-")[-1]))
    scored = []
    for ckpt in checkpoints:
        metrics = score_checkpoint(data, val_idx, ckpt, args)
        scored.append((ckpt, metrics))
    scored.sort(key=lambda item: (item[1]["maF1"], item[1]["miF1"]), reverse=True)
    return scored[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    labels = json.loads(Path(args.labels_json).read_text())

    rows = []
    split_root = out_dir / "splits"

    for repeat, fold, train_idx, val_idx, test_idx in make_repeat_splits(len(data), args.n_splits, args.n_repeats, args.val_ratio, args.seed):
        dataset_name = f"{args.dataset_prefix}_r{repeat+1}_f{fold+1}"
        split_dir = split_root / f"repeat_{repeat+1}" / f"fold_{fold+1}"
        split_paths = save_split(split_dir, train_idx, val_idx, test_idx)

        run_training(args, dataset_name, split_paths)

        model_dir = ROOT / "model" / f"model_{dataset_name}" / param_setting(args)
        best_ckpt, val_metrics = choose_best_checkpoint(model_dir, data, val_idx, args)
        test_metrics = score_checkpoint(data, test_idx, best_ckpt, args)

        rows.append(
            {
                "repeat": repeat + 1,
                "fold": fold + 1,
                "dataset_name": dataset_name,
                "best_checkpoint": str(best_ckpt.relative_to(ROOT)),
                "val_maF1": val_metrics["maF1"],
                "val_miF1": val_metrics["miF1"],
                "test_HA": test_metrics["HA"],
                "test_ebF1": test_metrics["ebF1"],
                "test_miF1": test_metrics["miF1"],
                "test_maF1": test_metrics["maF1"],
                "test_p_at_1": test_metrics["p_at_1"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        )

        pd.DataFrame(rows).to_csv(out_dir / "new_multisplit_cv_fold_metrics.csv", index=False)

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df[["test_HA", "test_ebF1", "test_miF1", "test_maF1", "test_p_at_1"]]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    fold_df.to_csv(out_dir / "new_multisplit_cv_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "new_multisplit_cv_summary.csv", index=False)
    (out_dir / "new_multisplit_cv_config.json").write_text(
        json.dumps(
            {
                "data": args.data,
                "label_dim": args.label_dim,
                "feature_dim": args.feature_dim,
                "labels": labels,
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "max_epoch": args.max_epoch,
                "balanced_batches": args.use_balanced_batches,
            },
            indent=2,
        )
    )
    print(out_dir / "new_multisplit_cv_fold_metrics.csv")
    print(out_dir / "new_multisplit_cv_summary.csv")


if __name__ == "__main__":
    main()
