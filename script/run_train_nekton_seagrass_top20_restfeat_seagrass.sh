python main.py --data_dir data/nekton-seagrass/nekton_seagrass_top20_restfeat_seagrass_data.npy --train_idx data/nekton-seagrass/nekton_seagrass_top20_restfeat_seagrass_train_idx.npy --valid_idx data/nekton-seagrass/nekton_seagrass_top20_restfeat_seagrass_val_idx.npy --test_idx data/nekton-seagrass/nekton_seagrass_top20_restfeat_seagrass_test_idx.npy --learning_rate 0.0005 \
    --max_epoch 60 --label_dim 20 --z_dim 32 --latent_dim 32 --feature_dim 690 --nll_coeff 0.5 --l2_coeff 1.0 --c_coeff 0.0 --batch_size 32 --write_to_test_sh --dataset nekton_seagrass_top20_restfeat_seagrass \
    --lr_decay_ratio 0.5 --lr_decay_times 4. --check_freq 20 --keep_prob 0.5 --mode train --emb_size 256 --reg gmvae --seed 1 \
    --train_ratio 1.0 --T0 25 --eta_min 2e-4 --T_mult 2
