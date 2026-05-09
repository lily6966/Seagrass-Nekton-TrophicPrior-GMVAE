from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT_PDF = OUTPUT_DIR / "multi_taxa_gm_vae_app_summary.pdf"


TITLE = "Multi-taxa-GM-VAE"
SUBTITLE = "Repo-based one-page app summary"

WHAT_IT_IS = (
    "A PyTorch research codebase implementing C-GMVAE, a Gaussian Mixture Variational "
    "Autoencoder with contrastive learning for multi-label classification. It includes training, "
    "validation, testing, checkpoints, and prepared dataset folders."
)

WHO_ITS_FOR = (
    "ML researchers or engineers running multi-label classification experiments in Python/PyTorch; "
    "a named end-user persona is Not found in repo."
)

FEATURES = [
    "Trains dataset-specific C-GMVAE models from CLI arguments.",
    "Supports multiple prepared datasets: mirflickr, fish, GBE_fish, and eBird_entire.",
    "Builds separate feature and label encoders plus a shared decoder.",
    "Combines reconstruction-style losses, KL divergence, and supervised contrastive loss.",
    "Evaluates predictions across thresholds with ACC, F1, AUC, AUPR, FDR, and precision@k.",
    "Writes TensorBoard logs under `summary/` and checkpoints under `model/model_<dataset>/`.",
    "Can auto-write dataset-specific test commands under `script/`.",
]

HOW_IT_WORKS = [
    "CLI: `main.py` parses dataset paths and hyperparameters, then dispatches to train or test mode.",
    "Data: `.npy` arrays plus train/val/test index files are loaded from `data/`; `utils.py` slices labels and features.",
    "Model: `model.py` encodes labels and features, decodes embeddings, and outputs label scores.",
    "Training: `train.py` batches data, computes losses and metrics, validates, saves checkpoints, and logs with TensorBoard.",
    "Evaluation: `test.py` reloads a checkpoint and `evals.py` scans thresholds to report multi-label metrics.",
]

RUN_STEPS = [
    "Install dependencies listed in `README.md`: Python 3.7+, PyTorch 1.7.0, numpy 1.17.3, sklearn 0.22.1.",
    "Use the sample dataset already in the repo at `data/mirflickr/`.",
    "Train: `bash script/run_train_mirflickr.sh`.",
    "Test: `bash script/run_test_mirflickr.sh`.",
    "Package manager, lockfile, and environment file: Not found in repo.",
]


def draw_section_box(ax, x, y, w, h, title, lines, bullet=False, body_fs=9.0, line_gap=0.030):
    ax.add_patch(
        Rectangle(
            (x, y - h),
            w,
            h,
            facecolor="#FAFBFC",
            edgecolor="#D8DEE4",
            linewidth=0.9,
        )
    )
    ax.text(
        x + 0.018,
        y - 0.028,
        title,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
    )
    cursor = y - 0.070
    for line in lines:
        prefix = u"\u2022 " if bullet else ""
        ax.text(
            x + 0.022,
            cursor,
            prefix + line,
            ha="left",
            va="top",
            fontsize=body_fs,
            color="#1F2937",
            wrap=True,
        )
        cursor -= line_gap


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.055, 0.93), 0.89, 0.012, color="#0F766E"))
    ax.text(0.06, 0.905, TITLE, fontsize=22, fontweight="bold", color="#0F172A", va="top")
    ax.text(0.06, 0.878, SUBTITLE, fontsize=10.5, color="#475569", va="top")

    draw_section_box(ax, 0.06, 0.845, 0.88, 0.105, "What It Is", [WHAT_IT_IS], body_fs=9.6, line_gap=0.033)
    draw_section_box(ax, 0.06, 0.725, 0.88, 0.082, "Who It's For", [WHO_ITS_FOR], body_fs=9.4, line_gap=0.032)
    draw_section_box(ax, 0.06, 0.628, 0.88, 0.238, "What It Does", FEATURES, bullet=True, body_fs=8.95, line_gap=0.027)
    draw_section_box(ax, 0.06, 0.372, 0.88, 0.176, "How It Works", HOW_IT_WORKS, bullet=True, body_fs=8.6, line_gap=0.0245)
    draw_section_box(ax, 0.06, 0.180, 0.88, 0.108, "How To Run", RUN_STEPS, bullet=True, body_fs=8.55, line_gap=0.0205)

    fig.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches=None)
    plt.close(fig)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
