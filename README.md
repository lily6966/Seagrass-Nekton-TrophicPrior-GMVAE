# Seagrass Nekton Trophic Prior GMVAE

This repository adapts the original Contrastive Gaussian Mixture Variational Autoencoder (C-GMVAE) codebase to model multi-species nekton occurrence in seagrass systems, with additional data-processing, transfer-learning, abundance, attribution, and partial-dependence analysis workflows.

The current repo is centered on the `data/nekton-seagrass/` . In practice, the main workflow is:

1. Build processed `.npy` datasets from raw nekton survey tables.
2. Train a GMVAE or transfer-learning variant with one of the shell scripts in `script/`.
3. Optionally run interpretation and plotting utilities from `tools/` and `pdp analysis py/`.

## Citation Required

If you use the model code in this repository, please cite the original C-GMVAE paper:

```bibtex
@inproceedings{bai2022gaussian,
  title={Gaussian Mixture Variational Autoencoder with Contrastive Learning for Multi-Label Classification},
  author={Bai, Junwen and Kong, Shufeng and Gomes, Carla P},
  booktitle={International Conference on Machine Learning},
  pages={1383--1398},
  year={2022},
  organization={PMLR}
}
```


## What This Repo Contains

- A PyTorch implementation of a GMVAE-style multilabel model.
- Processed-data builders for seagrass nekton presence/absence tasks.
- Transfer-learning dataset builders for trophic-group and fisheries-focused experiments.
- Training and evaluation scripts for transfer-learning models.
- Analysis helpers for attribution, plotting, and partial dependence summaries.

## Environment Setup

The repository includes a Conda environment file at `environment.yml`.

### Create the environment

```bash
conda env create -f environment.yml
conda activate multi-taxa-gmvae
```

### Core dependencies

The current environment file includes:

- Python 3.11
- PyTorch
- NumPy
- pandas
- matplotlib
- scikit-learn
- tensorboard
- joblib
- pillow
- openpyxl
- jupyterlab

### Notes

- Training will use GPU automatically if CUDA is available; otherwise it falls back to CPU.
- The codebase was originally derived from an older upstream C-GMVAE project, but this repo's checked-in environment is modernized around Python 3.11.
- If Conda solve time is slow on macOS, `mamba` is a good drop-in replacement for `conda`.

## File Structure

```text
.
├── main.py                     # CLI entry point for GMVAE train/test runs
├── train.py                    # Training loop and checkpoint/logging logic
├── test.py                     # Checkpoint evaluation for multilabel models
├── model.py                    # VAE / GMVAE architecture and loss wiring
├── data_loader.py              # Legacy dataset utilities
├── utils.py                    # Shared helpers
├── evals.py                    # Evaluation metrics
├── environment.yml             # Conda environment
├── data/
│   └── nekton-seagrass/        # Raw inputs and processed .npy datasets
├── losses/                     # Loss functions, including distribution-balanced loss
├── script/                     # Ready-to-run train/test shell scripts
├── tools/                      # Dataset builders, plotting, attribution, abundance workflows
├── pdp analysis py/            # Partial dependence analysis scripts
├── model/                      # Saved checkpoints
├── summary/                    # TensorBoard event logs
├── output/                     # Outputs from auxiliary tools
└── figs/                       # Generated figures and PDFs
```

## Data Layout

The raw ecological inputs currently live in `data/nekton-seagrass/`, including:

- `combined_sled_species_sp.csv`
- `combined_trawl_species.csv`
- `species_trophic_levels_master.xlsx`
- `nekton_grouping_lookup.csv`

Processed training datasets are written back into the same area as files such as:

- `nekton_seagrass_data.npy`
- `nekton_seagrass_train_idx.npy`
- `nekton_seagrass_val_idx.npy`
- `nekton_seagrass_test_idx.npy`
- `feature_columns*.json`
- `label_columns*.json`
- `processing_info*.json`

These processed files are what `main.py` and most training scripts consume.

## Easy Implementation

The easiest way to use this repository is to follow the workflow below.

### 1. Build the default processed dataset

```bash
python tools/process_nekton_seagrass.py
```

This reads the raw sled and trawl survey tables, builds feature matrices and multilabel targets, and writes:

- `data/nekton-seagrass/nekton_seagrass_data.npy`
- `data/nekton-seagrass/nekton_seagrass_train_idx.npy`
- `data/nekton-seagrass/nekton_seagrass_val_idx.npy`
- `data/nekton-seagrass/nekton_seagrass_test_idx.npy`
- `data/nekton-seagrass/feature_columns.json`
- `data/nekton-seagrass/label_columns.json`



### 2. Train a baseline multilabel model

Use the ready-made script:

```bash
bash script/run_train_nekton_seagrass.sh
```

That script trains with:

- dataset: `nekton_seagrass`
- labels: `199`
- features: `183`
- latent dimension: `32`
- epochs: `60`

It writes checkpoints under `model/` and TensorBoard logs under `summary/`.

### 3. Test a trained checkpoint

Use the paired evaluation script:

```bash
bash script/run_test_nekton_seagrass.sh
```

Or call the CLI directly:

```bash
python main.py \
  --data_dir data/nekton-seagrass/nekton_seagrass_data.npy \
  --test_idx data/nekton-seagrass/nekton_seagrass_test_idx.npy \
  --label_dim 199 \
  --feature_dim 183 \
  --z_dim 32 \
  --latent_dim 32 \
  --emb_size 256 \
  --nll_coeff 0.5 \
  --c_coeff 0.0 \
  --batch_size 64 \
  --reg gmvae \
  --mode test \
  --checkpoint_path model/model_nekton_seagrass/<run-dir>/<checkpoint>
```

### 4. Monitor training

```bash
tensorboard --logdir summary
```

## Main Workflows

### Baseline presence/absence modeling

- Dataset build: `python tools/process_nekton_seagrass.py`
- Train: `bash script/run_train_nekton_seagrass.sh`
- Test: `bash script/run_test_nekton_seagrass.sh`

### Transfer learning

This repo contains many prepared transfer-learning experiments in `script/`, including pretraining and fine-tuning variants for trophic groups, gear handling, and fisheries subsets.

Representative examples:


- `script/run_train_transfer_learning_pretrain_feedpos11_allspecies_seagrass.sh`
- `script/run_train_transfer_learning_finetune_fisheries8_fullgroupconsistency_nogear_v1.sh`

Related dataset builders live in `tools/`, for example:

- `tools/build_species_group_transfer_dataset.py`
- `tools/build_fisheries_trophic_dataset.py`


### Interpretation and figures

For post hoc interpretation and visualization, see:

- `tools/attribute_integrated_gradients.py`
- `tools/run_partial_dependence.py`
- `tools/plot_attribution_csvs.py`
- `tools/plot_species_performance.py`
- `pdp analysis py/`

## Command-Line Entry Point

`main.py` is the main entry point for GMVAE train/test runs.

### Train

```bash
python main.py --mode train ...
```

### Test

```bash
python main.py --mode test ...
```

Important arguments include:

- `--data_dir`: processed `.npy` matrix containing labels followed by features
- `--train_idx`, `--valid_idx`, `--test_idx`: split indices
- `--label_dim`: number of output labels
- `--feature_dim`: number of predictor columns
- `--latent_dim`, `--z_dim`, `--emb_size`: model size controls
- `--dataset`: run name used in output paths
- `--checkpoint_path`: checkpoint to evaluate or fine-tune

This repo also includes additional research flags for:

- rare-label oversampling
- distribution-balanced loss
- group consistency supervision
- gear residualization / gear interaction modeling
- gear flip regularization

For concrete settings, the easiest reference is still the corresponding shell script in `script/`.

## Outputs

During training and evaluation, you should expect:

- Checkpoints in `model/model_<dataset>/...`
- TensorBoard logs in `summary/<dataset>/...`
- Analysis outputs in `output/` or `figs/`

## License

This repository is distributed under the license in [LICENSE](LICENSE).

## Acknowledgment

This repository builds on the original C-GMVAE implementation and extends it for seagrass nekton ecological modeling, transfer learning, and interpretation workflows.
