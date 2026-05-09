python3 tools/train_abundance_regression.py \
  --data data/nekton-seagrass/nekton_seagrass_abundance_top20_env_seagrass_data.npy \
  --train-idx data/nekton-seagrass/nekton_seagrass_abundance_top20_env_seagrass_train_idx.npy \
  --val-idx data/nekton-seagrass/nekton_seagrass_abundance_top20_env_seagrass_val_idx.npy \
  --test-idx data/nekton-seagrass/nekton_seagrass_abundance_top20_env_seagrass_test_idx.npy \
  --labels-json data/nekton-seagrass/label_columns_abundance_top20_env_seagrass.json \
  --features-json data/nekton-seagrass/feature_columns_abundance_top20_env_seagrass.json \
  --label-dim 20 \
  --output-dir output/abundance/nekton_seagrass_top20_env_seagrass
