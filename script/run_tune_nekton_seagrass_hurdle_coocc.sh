OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HOME=$PWD/tmp/home XDG_CACHE_HOME=$PWD/tmp/cache MPLCONFIGDIR=$PWD/tmp/mpl python3 tools/tune_hurdle_abundance.py \
  --data data/nekton-seagrass/nekton_seagrass_abundance_top20_coocc_abundance_seagrass_data.npy \
  --train-idx data/nekton-seagrass/nekton_seagrass_abundance_top20_coocc_abundance_seagrass_train_idx.npy \
  --val-idx data/nekton-seagrass/nekton_seagrass_abundance_top20_coocc_abundance_seagrass_val_idx.npy \
  --labels-json data/nekton-seagrass/label_columns_abundance_top20_coocc_abundance_seagrass.json \
  --label-dim 20 \
  --output-dir output/hurdle_abundance_tuning/nekton_seagrass_top20_coocc_abundance_seagrass
