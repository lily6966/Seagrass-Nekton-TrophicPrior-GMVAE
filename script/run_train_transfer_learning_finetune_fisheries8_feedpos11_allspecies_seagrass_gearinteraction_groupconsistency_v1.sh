python main.py --data_dir data/nekton-seagrass/transfer_learning/nekton_seagrass_finetune_fisheries8_seagrass_data.npy --train_idx data/nekton-seagrass/transfer_learning/nekton_seagrass_finetune_fisheries8_seagrass_train_idx.npy --valid_idx data/nekton-seagrass/transfer_learning/nekton_seagrass_finetune_fisheries8_seagrass_val_idx.npy --test_idx data/nekton-seagrass/transfer_learning/nekton_seagrass_finetune_fisheries8_seagrass_test_idx.npy --learning_rate 0.0005 \
    --max_epoch 60 --label_dim 8 --z_dim 32 --latent_dim 32 --feature_dim 511 --nll_coeff 0.5 --l2_coeff 1.0 --c_coeff 0.0 --batch_size 32 --write_to_test_sh --dataset transfer_learning_finetune_fisheries8_feedpos11_allspecies_seagrass_gearinteraction_groupconsistency_v1 \
    --lr_decay_ratio 0.5 --lr_decay_times 3. --check_freq 20 --keep_prob 0.5 --mode train --emb_size 256 --reg gmvae --seed 1 \
    --train_ratio 1.0 --T0 25 --eta_min 2e-4 --T_mult 2 --finetune -pp REPLACE_WITH_PRETRAIN_CHECKPOINT \
    --group_consistency --group_consistency_weight 0.25 \
    --group_teacher_checkpoint REPLACE_WITH_PRETRAIN_CHECKPOINT \
    --group_teacher_labels_json data/nekton-seagrass/transfer_learning/label_columns_pretrain_feedpos11_allspecies_seagrass.json \
    --current_labels_json data/nekton-seagrass/transfer_learning/label_columns_finetune_fisheries8_seagrass.json \
    --species_group_map_json data/nekton-seagrass/transfer_learning/species_to_pretrain_group_fisheries8_feedpos11_allspecies.json \
    --gear_interaction_design --gear_interaction_hidden_dim 16 --gear_interaction_scale 0.25 --gear_flip_penalty_weight 0.05
