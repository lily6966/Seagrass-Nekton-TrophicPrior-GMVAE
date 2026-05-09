import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import VAE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clean_label(label: str) -> str:
    cleaned = label.replace("niche_label__", "")
    friendly = {
        "Penaeus aztecus": "Brown shrimp",
        "Penaeus duorarum": "Pink shrimp",
        "Penaeus setiferus": "White shrimp",
        "Callinectes sapidus": "Blue crab",
        "Sciaenops ocellatus": "Red drum",
        "Cynoscion nebulosus": "Spotted seatrout",
        "Lutjanus griseus": "Gray snapper",
        "Lutjanus synagris": "Lane snapper",
        "planktivore": "Planktivore",
        "benthic_invertivore": "Benthic invertivore",
        "piscivore": "Piscivore",
        "omnivore": "Omnivore",
        "herbivore_detritivore": "Herbivore/detritivore",
        "shrimp_crab_invertivore": "Shrimp/crab invertivore",
        "unknown": "Unknown",
    }
    return friendly.get(cleaned, cleaned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated gradients attribution for GMVAE feature-based predictions.")
    parser.add_argument("--data", required=True, help="Path to .npy data matrix.")
    parser.add_argument("--features", required=True, help="Path to feature_columns.json.")
    parser.add_argument("--labels", required=True, help="Path to label_columns.json.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for attribution outputs.")
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--label-dim", required=True, type=int)
    parser.add_argument("--latent-dim", required=True, type=int)
    parser.add_argument("--z-dim", required=True, type=int)
    parser.add_argument("--emb-size", required=True, type=int)
    parser.add_argument("--keep-prob", default=0.5, type=float)
    parser.add_argument("--steps", default=32, type=int, help="Integrated gradients interpolation steps.")
    parser.add_argument("--max-samples", default=0, type=int, help="Optional cap on number of samples; 0 means all.")
    parser.add_argument("--region-prefix", default="site_", help="One-hot feature prefix to use as region grouping.")
    parser.add_argument("--target-label", action="append", default=[], help="Specific label(s) to attribute; default is all labels.")
    parser.add_argument("--baseline", choices=["zero", "mean"], default="zero")
    parser.add_argument("--sample-filter-feature", default="", help="Optional feature name to filter samples by.")
    parser.add_argument(
        "--sample-filter-mode",
        choices=["positive", "nonpositive"],
        default="positive",
        help="How to filter on sample-filter-feature.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> VAE:
    model_args = SimpleNamespace(
        feature_dim=args.feature_dim,
        label_dim=args.label_dim,
        latent_dim=args.latent_dim,
        z_dim=args.z_dim,
        emb_size=args.emb_size,
        keep_prob=args.keep_prob,
    )
    model = VAE(model_args).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_regions(x: np.ndarray, feature_names: list[str], region_prefix: str) -> np.ndarray:
    region_idx = [i for i, name in enumerate(feature_names) if name.startswith(region_prefix)]
    if not region_idx:
        raise ValueError(f"No region one-hot columns found with prefix {region_prefix!r}.")
    region_names = [feature_names[i][len(region_prefix):] for i in region_idx]
    region_block = x[:, region_idx]
    chosen = np.argmax(region_block, axis=1)
    return np.array([region_names[i] for i in chosen])


def get_feature_groups(feature_names: list[str]) -> list[str]:
    groups = []
    for name in feature_names:
        if name.startswith("species_feature__"):
            groups.append("cooccurrence")
        else:
            groups.append("environment")
    return groups


def integrated_gradients(
    model: VAE,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_idx: int,
    label_dim: int,
    steps: int,
) -> torch.Tensor:
    delta = inputs - baseline
    total_grads = torch.zeros_like(inputs)

    for step in range(1, steps + 1):
        alpha = step / steps
        current = baseline + alpha * delta
        current.requires_grad_(True)

        dummy_label = torch.zeros((current.shape[0], label_dim), device=current.device)
        dummy_label[:, 0] = 1.0

        output = model(dummy_label, current)
        scores = output["feat_out"][:, target_idx].sum()
        grads = torch.autograd.grad(scores, current)[0]
        total_grads += grads.detach()

    avg_grads = total_grads / steps
    return delta * avg_grads


def summarize(
    attributions: np.ndarray,
    x: np.ndarray,
    regions: np.ndarray,
    feature_names: list[str],
    feature_groups: list[str],
    target_label: str,
) -> pd.DataFrame:
    rows = []
    display_label = clean_label(target_label)
    for region in np.unique(regions):
        mask = regions == region
        region_attr = attributions[mask]
        region_x = x[mask]
        for j, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "region": region,
                    "target_label": display_label,
                    "feature_name": feature_name,
                    "feature_group": feature_groups[j],
                    "mean_signed_attribution": float(region_attr[:, j].mean()),
                    "mean_abs_attribution": float(np.abs(region_attr[:, j]).mean()),
                    "mean_feature_value": float(region_x[:, j].mean()),
                    "n_samples": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = json.loads(Path(args.features).read_text())
    label_names = json.loads(Path(args.labels).read_text())
    data = np.load(args.data)

    x = data[:, args.label_dim:]
    sample_mask = np.ones(x.shape[0], dtype=bool)
    if args.sample_filter_feature:
        if args.sample_filter_feature not in feature_names:
            raise ValueError(f"Unknown sample filter feature: {args.sample_filter_feature}")
        feature_idx = feature_names.index(args.sample_filter_feature)
        values = x[:, feature_idx]
        if args.sample_filter_mode == "positive":
            sample_mask = values > 0
        else:
            sample_mask = values <= 0
        x = x[sample_mask]
    if args.max_samples > 0:
        x = x[: args.max_samples]

    regions = get_regions(x, feature_names, args.region_prefix)
    feature_groups = get_feature_groups(feature_names)

    model = build_model(args)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    if args.baseline == "zero":
        baseline = torch.zeros_like(x_tensor)
    else:
        baseline = x_tensor.mean(dim=0, keepdim=True).repeat(x_tensor.shape[0], 1)

    if args.target_label:
        target_labels = args.target_label
    else:
        target_labels = label_names

    all_summaries = []
    group_totals = []

    for target_label in target_labels:
        if target_label not in label_names:
            raise ValueError(f"Unknown target label: {target_label}")
        target_idx = label_names.index(target_label)
        attr = integrated_gradients(model, x_tensor, baseline, target_idx, args.label_dim, args.steps)
        attr_np = attr.detach().cpu().numpy()

        summary = summarize(attr_np, x, regions, feature_names, feature_groups, target_label)
        all_summaries.append(summary)

        totals = (
            summary.groupby(["region", "target_label", "feature_group"], as_index=False)["mean_abs_attribution"]
            .sum()
            .pivot(index=["region", "target_label"], columns="feature_group", values="mean_abs_attribution")
            .reset_index()
            .fillna(0.0)
        )
        group_totals.append(totals)

    full_summary = pd.concat(all_summaries, ignore_index=True)
    full_totals = pd.concat(group_totals, ignore_index=True)

    summary_path = output_dir / "integrated_gradients_by_region.csv"
    totals_path = output_dir / "integrated_gradients_group_totals.csv"
    meta_path = output_dir / "integrated_gradients_metadata.json"

    full_summary.to_csv(summary_path, index=False)
    full_totals.to_csv(totals_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "data": args.data,
                "features": args.features,
                "labels": args.labels,
                "steps": args.steps,
                "baseline": args.baseline,
                "region_prefix": args.region_prefix,
                "n_samples": int(x.shape[0]),
                "sample_filter_feature": args.sample_filter_feature,
                "sample_filter_mode": args.sample_filter_mode if args.sample_filter_feature else "",
                "targets": target_labels,
                "display_targets": [clean_label(label) for label in target_labels],
            },
            indent=2,
        )
    )

    print(summary_path)
    print(totals_path)
    print(meta_path)


if __name__ == "__main__":
    main()
