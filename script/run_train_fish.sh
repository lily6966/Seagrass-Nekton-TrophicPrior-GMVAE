python main.py --data_dir data/fish/fish_data.npy --train_idx data/fish/fish_train_idx.npy --valid_idx data/fish/fish_val_idx.npy --test_idx data/fish/fish_test_idx.npy --learning_rate 0.002 \
    --max_epoch 90 --label_dim 12 --z_dim 38 --feature_dim 46 --nll_coeff 0.5 --l2_coeff 1.0 --c_coeff 0. --batch_size 32 --write_to_test_sh --dataset fish \
    --lr_decay_ratio 0.5 --lr_decay_times 4. --check_freq 40 --keep_prob 0.5 --mode train --emb_size 2048 --reg gmvae --seed 1 \
    --train_ratio 1.0 --T0 50 --eta_min 2e-4 --T_mult 2
