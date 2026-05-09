import math
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os
import datetime
from copy import copy, deepcopy
import evals
from utils import build_path, get_label, get_feat, THRESHOLDS
from model import VAE, compute_loss
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']


def infer_feature_columns_path(data_dir):
    data_path = os.path.abspath(data_dir)
    if not data_path.endswith("_data.npy"):
        return None
    base_name = os.path.basename(data_path)
    if base_name == "nekton_seagrass_data.npy":
        feature_name = "feature_columns.json"
    elif base_name.startswith("nekton_seagrass_"):
        suffix = base_name[len("nekton_seagrass_"):-len("_data.npy")]
        feature_name = f"feature_columns_{suffix}.json"
    else:
        return None
    feature_path = os.path.join(os.path.dirname(data_path), feature_name)
    return feature_path if os.path.exists(feature_path) else None


def get_gear_feature_index(args):
    feature_path = infer_feature_columns_path(args.data_dir)
    if feature_path is None:
        return None
    try:
        feature_columns = json.load(open(feature_path, "r"))
    except Exception:
        return None
    try:
        return feature_columns.index("gear_is_sled")
    except ValueError:
        return None


def compute_best_metrics(predictions, labels):
    best_val_metrics = None
    for threshold in THRESHOLDS:
        val_metrics = evals.compute_metrics(predictions, labels, threshold, all_metrics=True)

        if best_val_metrics is None:
            best_val_metrics = {}
            for metric in METRICS:
                best_val_metrics[metric] = val_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_val_metrics[metric] = min(best_val_metrics[metric], val_metrics[metric])
                else:
                    best_val_metrics[metric] = max(best_val_metrics[metric], val_metrics[metric])
    return best_val_metrics


def print_metric_summary(prefix, metrics_dict, n_samples):
    print(
        "%s (n=%d): acc=%.6f\tha=%.6f\texam_f1=%.6f, macro_f1=%.6f, micro_f1=%.6f" % (
            prefix,
            n_samples,
            metrics_dict['ACC'],
            metrics_dict['HA'],
            metrics_dict['ebF1'],
            metrics_dict['maF1'],
            metrics_dict['miF1'],
        )
    )


def configure_gear_context(args, data=None):
    gear_feature_idx = get_gear_feature_index(args)
    args.gear_feature_idx = gear_feature_idx
    args.gear_flip_low = None
    args.gear_flip_high = None
    if gear_feature_idx is None or data is None:
        return
    feature_block = data[:, args.meta_offset + args.label_dim: args.meta_offset + args.label_dim + args.feature_dim]
    gear_values = feature_block[:, gear_feature_idx].astype(np.float32)
    if len(gear_values) == 0:
        return
    args.gear_flip_low = float(np.min(gear_values))
    args.gear_flip_high = float(np.max(gear_values))


def flip_gear_feature(input_feat, args):
    gear_idx = getattr(args, "gear_feature_idx", None)
    low = getattr(args, "gear_flip_low", None)
    high = getattr(args, "gear_flip_high", None)
    if gear_idx is None or low is None or high is None or abs(high - low) < 1e-8:
        return None
    flipped = input_feat.clone()
    current = flipped[:, gear_idx]
    midpoint = 0.5 * (low + high)
    flipped[:, gear_idx] = torch.where(
        current > midpoint,
        torch.full_like(current, low),
        torch.full_like(current, high),
    )
    return flipped


def compute_gear_flip_penalty(vae, input_label, input_feat, output, args):
    weight = float(getattr(args, "gear_flip_penalty_weight", 0.0))
    if weight <= 0 or getattr(args, "gear_feature_idx", None) is None:
        return torch.tensor(0.0, device=input_feat.device)
    flipped_feat = flip_gear_feature(input_feat, args)
    if flipped_feat is None:
        return torch.tensor(0.0, device=input_feat.device)
    flipped_output = vae(input_label, flipped_feat)
    base_prob = torch.sigmoid(output["feat_out"])
    flipped_prob = torch.sigmoid(flipped_output["feat_out"])
    return (base_prob - flipped_prob).pow(2).mean()

class DataLoader:
    def __init__(self, indices, labels, rare_positive_oversample=False, rare_positive_threshold=20, rare_positive_weight_power=1.0, rare_positive_max_weight=5.0):
        self.indices = indices
        self.n_label = labels.shape[1]
        self.label_sets = [[] for _ in range(self.n_label)]
        self.lengths = [0] * self.n_label
        self.label_counts = labels.sum(0).astype(float)
        for i, label in zip(indices, labels):
            for j in range(self.n_label):
                if label[j] == 1:
                    self.label_sets[j].append(i)
        self.max_len = 0
        self.tot_len = 0
        for i in range(self.n_label):
            self.lengths[i] = len(self.label_sets[i])
            self.tot_len += self.lengths[i]
            self.max_len = max(self.max_len, self.lengths[i])
        self.labels = list(range(self.n_label))
        self.rare_positive_oversample = rare_positive_oversample
        rare_mask = (self.label_counts > 0) & (self.label_counts <= rare_positive_threshold)
        self.rare_labels = np.where(rare_mask)[0].tolist()
        self.label_sampling_weights = np.ones(self.n_label, dtype=float)
        if self.rare_labels:
            rare_counts = np.maximum(self.label_counts[self.rare_labels], 1.0)
            self.label_sampling_weights[self.rare_labels] = np.clip(
                (rare_positive_threshold / rare_counts) ** rare_positive_weight_power,
                1.0,
                rare_positive_max_weight,
            )
        self.sample_weights = np.ones(len(indices), dtype=float)
        if rare_positive_oversample and self.rare_labels:
            rare_counts = np.maximum(self.label_counts[self.rare_labels], 1.0)
            rare_weight_lookup = {
                label_id: np.clip((rare_positive_threshold / count) ** rare_positive_weight_power, 1.0, rare_positive_max_weight)
                for label_id, count in zip(self.rare_labels, rare_counts)
            }
            for sample_pos, label in enumerate(labels):
                active = [rare_weight_lookup[idx] for idx in self.rare_labels if label[idx] == 1]
                if active:
                    self.sample_weights[sample_pos] = min(rare_positive_max_weight, max(active))
        self.sample_weights = (self.sample_weights / self.sample_weights.sum()).tolist()

    def sample_idxs(self, bs):
        if bs < self.n_label:
            label_ids = random.choices(self.labels, weights=self.label_sampling_weights, k=bs)
            ids = []
            for idx in label_ids:
                if self.label_sets[idx]:
                    ids.append(random.choice(self.label_sets[idx]))
                else:
                    ids.append(random.choices(self.indices, weights=self.sample_weights, k=1)[0])
            return ids
        else:
            num = math.floor(bs/self.n_label)
            ids = []
            for idx in self.labels:
                if self.label_sets[idx]:
                    ids.extend(random.choices(self.label_sets[idx], k=num))
            ids = np.array(ids, dtype=int) if len(ids) else np.array([], dtype=int)
            ids = np.concatenate([ids, np.array(random.choices(self.indices, weights=self.sample_weights, k=bs-len(ids)), dtype=int)])
            return ids


def apply_feature_mask(input_feat, mask_rate):
    if mask_rate <= 0:
        return input_feat
    keep_mask = torch.rand_like(input_feat) > mask_rate
    return input_feat * keep_mask.float()

def expand(idxs, labels):
    label_cnts = labels.sum(0)
    tot_cnt = len(labels)
    new_idxs = []
    for idx, label in zip(idxs, labels):
        min_cnt = tot_cnt
        for i, l in enumerate(label):
            if l == 1:
                min_cnt = min(min_cnt, label_cnts[i])
        multi = int(round(500./min_cnt))
        if multi <= 1:
            new_idxs.append(idx)
        else:
            for _ in range(multi):
                new_idxs.append(idx)
    random.shuffle(new_idxs)
    return new_idxs


def load_matching_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    model_state.update(matched)
    model.load_state_dict(model_state)
    print("loaded compatible pretrained weights from:", checkpoint_path)
    print("matched tensors:", len(matched), "skipped tensors:", len(skipped))
    if skipped:
        print("skipped keys:", skipped)
    return matched, skipped


def build_group_consistency_context(args):
    if not getattr(args, "group_consistency", False):
        return None
    if not args.group_teacher_checkpoint or not args.group_teacher_labels_json or not args.current_labels_json or not args.species_group_map_json:
        raise ValueError("group consistency requires teacher checkpoint, teacher labels json, current labels json, and species-group map json")

    teacher_labels = json.load(open(args.group_teacher_labels_json, "r"))
    current_labels = json.load(open(args.current_labels_json, "r"))
    species_group_map = json.load(open(args.species_group_map_json, "r"))
    teacher_label_to_idx = {label: i for i, label in enumerate(teacher_labels)}

    species_to_group_idx = []
    for label in current_labels:
        group_name = species_group_map.get(label)
        if group_name is None or group_name not in teacher_label_to_idx:
            species_to_group_idx.append(-1)
        else:
            species_to_group_idx.append(teacher_label_to_idx[group_name])

    teacher_state = torch.load(args.group_teacher_checkpoint, map_location=device)
    teacher_args = deepcopy(args)
    teacher_args.label_dim = len(teacher_labels)
    teacher_args.emb_size = int(teacher_state["label_lookup.weight"].shape[0])
    teacher_args.latent_dim = int(teacher_state["fx_mu.bias"].shape[0])
    teacher = VAE(teacher_args).to(device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return {
        "teacher": teacher,
        "teacher_label_dim": len(teacher_labels),
        "species_to_group_idx": torch.tensor(species_to_group_idx, dtype=torch.long, device=device),
        "weight": float(getattr(args, "group_consistency_weight", 0.25)),
    }


def compute_group_consistency_loss(current_feat_logits, input_feat, ctx):
    if ctx is None:
        return torch.tensor(0.0, device=input_feat.device)
    mapping = ctx["species_to_group_idx"]
    valid_mask = mapping >= 0
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=input_feat.device)

    teacher = ctx["teacher"]
    teacher_label_dim = ctx["teacher_label_dim"]
    teacher_dummy = torch.zeros((input_feat.shape[0], teacher_label_dim), device=input_feat.device)
    teacher_dummy[:, 0] = 1.0
    with torch.no_grad():
        teacher_out = teacher(teacher_dummy, input_feat)
        teacher_group_prob = torch.sigmoid(teacher_out["feat_out"])

    species_prob = torch.sigmoid(current_feat_logits[:, valid_mask])
    parent_group_prob = teacher_group_prob[:, mapping[valid_mask]]
    # One-sided penalty: species occurrence should not exceed parent-group occurrence.
    return torch.relu(species_prob - parent_group_prob).pow(2).mean()

def train(args):
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)

    print('reading npy...')
    data = np.load(args.data_dir, allow_pickle=True) #load data from the data_dir
    configure_gear_context(args, data)
    train_idx = np.load(args.train_idx) #load the indices of the training set
    valid_idx = np.load(args.valid_idx) #load the indices of the validation set
    test_idx = np.load(args.test_idx)
    labels = get_label(data, train_idx, args.meta_offset, args.label_dim) #load the labels of the training set
    dataloader = DataLoader(
        train_idx,
        labels,
        rare_positive_oversample=getattr(args, "rare_positive_oversample", False),
        rare_positive_threshold=getattr(args, "rare_positive_threshold", 20),
        rare_positive_weight_power=getattr(args, "rare_positive_weight_power", 1.0),
        rare_positive_max_weight=getattr(args, "rare_positive_max_weight", 5.0),
    )
    class_weights = np.reciprocal(labels.sum(0).astype(float))
    class_weights /= np.amax(class_weights)
    args.class_freq_tensor = torch.from_numpy(labels.sum(0).astype(np.float32))
    args.train_sample_count = int(labels.shape[0])

    n_train = len(train_idx)
    train_idx = train_idx[:int(n_train*args.train_ratio)]


    print("positive label rate:", np.mean(labels)) #print the rate of the positive labels in the training set
    param_setting = "lr-{}_lr-decay_{:.2f}_lr-times_{:.1f}_nll-{:.2f}_l2-{:.2f}_c-{:.2f}".format(args.learning_rate, args.lr_decay_ratio, args.lr_decay_times, args.nll_coeff, args.l2_coeff, args.c_coeff)
    build_path('summary/{}/{}'.format(args.dataset, param_setting))
    build_path('model/model_{}/{}'.format(args.dataset, param_setting))
    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/model_{}/{}'.format(args.dataset, param_setting)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size) # compute the number of iterations in each epoch
    n_iter = one_epoch_iter * args.max_epoch
    print("one_epoch_iter:", one_epoch_iter)
    print("total_iter:", n_iter)

    print("showing the parameters...")
    print(args)

    writer = SummaryWriter(log_dir=summary_dir)

    print('building network...')
    
    #building the model 
    vae = VAE(args).to(device)
    group_consistency_ctx = build_group_consistency_context(args)

    vae.train()
    if args.finetune:
        load_matching_state_dict(vae, args.pretrain_path)

    #log the learning rate 
    writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)

    if args.resume:
        vae.load_state_dict(torch.load(args.checkpoint_path))
        current_step = int(args.checkpoint_path.split('/')[-1].split('-')[-1]) 
        print("loaded model: %s" % args.label_checkpoint_path)
    else:
        current_step = 0

    # smooth means average. Every batch has a mean loss value w.r.t. different losses
    smooth_nll_loss=0.0 # label encoder decoder cross entropy loss
    smooth_nll_loss_x=0.0 # feature encoder decoder cross entropy loss
    smooth_c_loss = 0.0 # label encoder decoder ranking loss
    smooth_c_loss_x=0.0 # feature encoder decoder ranking loss
    smooth_kl_loss = 0.0 # kl divergence
    smooth_total_loss=0.0 # total loss
    smooth_aux_loss = 0.0
    smooth_gear_loss = 0.0
    smooth_macro_f1 = 0.0 # macro_f1 score
    smooth_micro_f1 = 0.0 # micro_f1 score

    best_loss = 1e10
    best_iter = 0
    best_macro_f1 = 0.0 # best macro f1 for ckpt selection in validation
    best_micro_f1 = 0.0 # best micro f1 for ckpt selection in validation
    best_acc = 0.0 # best subset acc for ckpt selction in validation

    temp_label=[]
    temp_pred_x=[]

    best_test_metrics = None


    # training the model
    for one_epoch in range(args.max_epoch):
        if one_epoch:
            scheduler.step()
        print('epoch '+str(one_epoch+1)+' starts!')
        np.random.shuffle(train_idx) # random shuffle the training indices
        n_train = len(train_idx)

        for i in range(int(len(train_idx)/float(args.batch_size))+1):
            optimizer.zero_grad()
            start = i*args.batch_size
            end = min(args.batch_size*(i+1), len(train_idx))
            if end <= start:
                continue
            if getattr(args, "balanced_batches", False):
                batch_indices = np.array(dataloader.sample_idxs(end - start), dtype=int)
            else:
                batch_indices = train_idx[start:end]

            input_feat = get_feat(data, batch_indices, args.meta_offset, args.label_dim, args.feature_dim) # get the features
            input_label = get_label(data, batch_indices, args.meta_offset, args.label_dim) # get the ground-truth labels 
            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)
            input_feat = apply_feature_mask(input_feat, getattr(args, "feature_mask_rate", 0.0))
            dummy_label = torch.zeros(args.label_dim).to(device)
            dummy_label[0] = 1.0  # just activate one label

            mask = input_label.sum(dim=1) > 0
            if not mask.all():
                input_label[~mask] = dummy_label


            output = vae(input_label, input_feat)
            # print("input_label:", input_label)
            # print("Rows with all zeros:", (input_label.sum(dim=1) == 0).nonzero())

            # if torch.isnan(output['label_emb']).any():
            #             print("NaNs in label_emb!")
            #             print("label_emb:", output['label_emb'])
            #             print("input_label that produced it:", input_label)
            



            # for k, v in output.items():
            #     if isinstance(v, torch.Tensor):
            #         has_nan = torch.isnan(v).any().item()
            #         print(f"Any NaNs in output['{k}']? {has_nan}")

            #print("Any NaNs in input labels?", np.isnan(input_label.cpu().numpy()).any())



            #train the model for one step and log the training loss
            gear_penalty = torch.tensor(0.0, device=device)
            if args.residue_sigma == "random":
                pass
            else:
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, _, pred_x = \
                    compute_loss(input_label, output, args)
                aux_group_loss = compute_group_consistency_loss(output['feat_out'], input_feat, group_consistency_ctx)
                total_loss = total_loss + group_consistency_ctx["weight"] * aux_group_loss if group_consistency_ctx is not None else total_loss
                gear_penalty = compute_gear_flip_penalty(vae, input_label, input_feat, output, args)
                total_loss = total_loss + float(getattr(args, "gear_flip_penalty_weight", 0.0)) * gear_penalty

            total_loss.backward()
            optimizer.step()

            train_metrics = evals.compute_metrics(pred_x.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5, all_metrics=False)
            macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']

            smooth_nll_loss += nll_loss
            smooth_nll_loss_x += nll_loss_x
            smooth_c_loss += c_loss
            smooth_c_loss_x += c_loss_x
            smooth_kl_loss += kl_loss
            smooth_total_loss += total_loss
            smooth_aux_loss += aux_group_loss
            smooth_gear_loss += gear_penalty
            smooth_macro_f1 += macro_f1
            smooth_micro_f1 += micro_f1
            
            temp_label.append(input_label) #log the labels
            temp_pred_x.append(pred_x) #log the individual prediction of the probability on each label

            current_step += 1
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', lr, current_step)

            if current_step % args.check_freq==0: #summarize the current training status and print them out
                nll_loss = smooth_nll_loss / float(args.check_freq)
                nll_loss_x = smooth_nll_loss_x / float(args.check_freq)
                c_loss = smooth_c_loss / float(args.check_freq)
                c_loss_x = smooth_c_loss_x / float(args.check_freq)
                kl_loss = smooth_kl_loss / float(args.check_freq)
                total_loss = smooth_total_loss / float(args.check_freq)
                aux_loss = smooth_aux_loss / float(args.check_freq)
                gear_loss = smooth_gear_loss / float(args.check_freq)
                macro_f1 = smooth_macro_f1 / float(args.check_freq)
                micro_f1 = smooth_micro_f1 / float(args.check_freq)
                
                temp_pred_x = [v.cpu().detach().numpy()  for v in temp_pred_x]
                temp_label = [v.cpu().detach().numpy()  for v in temp_label]
                temp_pred_x = np.reshape(np.array(temp_pred_x, dtype=object), (-1))
                temp_label = np.reshape(np.array(temp_label, dtype=object), (-1))
                if torch.isnan(nll_loss):
                    print("NaN in nll_loss!")
                if torch.isnan(kl_loss):
                    print("NaN in kl_loss!")
                if torch.isnan(cpc_loss):
                    print("NaN in cpc_loss!")

                time_str = datetime.datetime.now().isoformat()
                print("step=%d  %s\nmacro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\tcpc_loss=%.6f\taux_group_loss=%.6f\tgear_flip_loss=%.6f\ttotal_loss=%.6f\n" % (
                    current_step, time_str, macro_f1, micro_f1, nll_loss*args.nll_coeff, nll_loss_x*args.nll_coeff, c_loss*args.c_coeff, c_loss_x*args.c_coeff, kl_loss, cpc_loss, aux_loss, gear_loss, total_loss))
                temp_pred_x=[]
                temp_label=[]

                smooth_nll_loss = 0
                smooth_nll_loss_x = 0
                smooth_c_loss = 0
                smooth_c_loss_x = 0
                smooth_kl_loss = 0
                smooth_total_loss = 0
                smooth_aux_loss = 0
                smooth_gear_loss = 0
                smooth_macro_f1 = 0
                smooth_micro_f1 = 0

            if current_step % int(one_epoch_iter*args.save_epoch)==0: #exam the model on validation set
                print("--------------------------------")
                # exam the model on validation set
                current_loss, val_metrics = valid(data, vae, writer, valid_idx, current_step, args, extra={"group_consistency_ctx": group_consistency_ctx})

                optimizer.zero_grad()

                macro_f1, micro_f1 = val_metrics['maF1'], val_metrics['miF1']

                # select the best checkpoint based on some metric on the validation set
                # here we use macro F1 as the selection metric but one can use others
                if val_metrics['maF1'] > best_macro_f1:
                    
                    best_loss = current_loss
                    best_iter = current_step

                    print('saving model')
                    torch.save(vae.state_dict(), model_dir+'/vae-'+str(current_step))

                    print('have saved model to ', model_dir)
                    print()

                    if args.write_to_test_sh:
                        test_sh_path = "script/run_test_%s.sh" % args.dataset
                        if os.path.exists(test_sh_path):
                            ckptFile = open(test_sh_path, "r")
                            command = []
                            for line in ckptFile:
                                arg_lst = line.strip().split(' ')
                                for arg in arg_lst:
                                    if 'model/model_{}/lr-'.format(args.dataset) in arg:
                                        command.append('model/model_{}/{}/vae-{}'.format(args.dataset, param_setting, best_iter))
                                    else:
                                        command.append(arg)
                            ckptFile.close()
                        else:
                            command = ("python main.py --data_dir %s --test_idx %s --label_dim %d --z_dim %d --feature_dim %d --nll_coeff %s --c_coeff %s --batch_size 64 --mode test --emb_size %d --reg gmvae -cp %s" % (args.data_dir, args.test_idx, args.label_dim, args.z_dim, args.feature_dim, args.nll_coeff, args.c_coeff, args.emb_size, 'model/model_{}/{}/vae-{}'.format(args.dataset, param_setting, best_iter))).strip().split(' ')
                        
                        ckptFile = open(test_sh_path, "w")
                        ckptFile.write(" ".join(command)+"\n")
                        ckptFile.close()
                best_macro_f1 = max(best_macro_f1, val_metrics['maF1'])
                best_micro_f1 = max(best_micro_f1, val_metrics['miF1'])
                best_acc = max(best_acc, val_metrics['ACC'])
                
                print("--------------------------------")

    torch.save(vae.state_dict(), model_dir+'/vae-'+str(current_step))


def valid(data, vae, summary_writer, valid_idx, current_step, args, extra=None):
    vae.eval()
    print("performing validation...")
    group_consistency_ctx = None if extra is None else extra.get("group_consistency_ctx")

    all_nll_loss = 0
    all_l2_loss = 0
    all_c_loss = 0
    all_kl_loss = 0
    all_total_loss = 0
    all_aux_loss = 0
    all_gear_loss = 0

    all_pred_x = []
    all_pred_e = []
    all_label = []
    all_input_feat = []

    real_batch_size=min(args.batch_size, len(valid_idx))
    for i in range(int((len(valid_idx)-1)/real_batch_size)+1):
        start = real_batch_size*i
        end = min(real_batch_size*(i+1), len(valid_idx))

        input_feat = get_feat(data,valid_idx[start:end], args.meta_offset, args.label_dim, args.feature_dim)
        input_label = get_label(data,valid_idx[start:end], args.meta_offset, args.label_dim)
        input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
        input_label = deepcopy(input_label).float().to(device)
        dummy_label = torch.zeros(args.label_dim).to(device)
        dummy_label[0] = 1.0
        mask = input_label.sum(dim=1) > 0
        if not mask.all():
            input_label[~mask] = dummy_label

        with torch.no_grad():
            output = vae(input_label, input_feat) 
            total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, pred_e, pred_x = \
                compute_loss(input_label, output, args)
            aux_group_loss = compute_group_consistency_loss(output['feat_out'], input_feat, group_consistency_ctx)
            if group_consistency_ctx is not None:
                total_loss = total_loss + group_consistency_ctx["weight"] * aux_group_loss
            gear_penalty = compute_gear_flip_penalty(vae, input_label, input_feat, output, args)
            total_loss = total_loss + float(getattr(args, "gear_flip_penalty_weight", 0.0)) * gear_penalty
            
        all_nll_loss += nll_loss*(end-start)
        all_c_loss += c_loss*(end-start)
        all_total_loss += total_loss*(end-start)
        all_kl_loss += kl_loss*(end-start)
        all_aux_loss += aux_group_loss*(end-start)
        all_gear_loss += gear_penalty*(end-start)

        
        all_pred_x.append(pred_x)
        all_pred_e.append(pred_e)
        all_label.append(input_label)
        all_input_feat.append(input_feat.detach().cpu())

       

    # collect all predictions and ground-truths
    all_pred_x = torch.cat(all_pred_x).detach().cpu().numpy()
    all_pred_e = torch.cat(all_pred_e).detach().cpu().numpy()
    all_label = torch.cat(all_label).detach().cpu().numpy()
    all_input_feat = torch.cat(all_input_feat).numpy()

    nll_loss = all_nll_loss/len(valid_idx)
    l2_loss = all_l2_loss/len(valid_idx)
    c_loss = all_c_loss/len(valid_idx)
    total_loss = all_total_loss/len(valid_idx)
    kl_loss = all_kl_loss/len(valid_idx)
    aux_loss = all_aux_loss/len(valid_idx)
    gear_loss = all_gear_loss/len(valid_idx)

    def show_results(all_indiv_prob):
        best_val_metrics = compute_best_metrics(all_indiv_prob, all_label)
        time_str = datetime.datetime.now().isoformat()
        acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], best_val_metrics['ebF1'], best_val_metrics['maF1'], best_val_metrics['miF1']

        print("**********************************************")
        print("valid results: %s" % time_str)
        print_metric_summary("overall", best_val_metrics, len(all_label))
        print("nll_loss=%.6f\tc_loss=%.6f\nkl_loss=%.6f\taux_group_loss=%.6f\tgear_flip_loss=%.6f\ttotal_loss=%.6f" % (nll_loss*args.nll_coeff, c_loss*args.c_coeff, kl_loss, aux_loss, gear_loss, total_loss))

        gear_feature_idx = get_gear_feature_index(args)
        if gear_feature_idx is not None and gear_feature_idx < all_input_feat.shape[1]:
            gear_values = all_input_feat[:, gear_feature_idx]
            gear_threshold = 0.5 * (gear_values.min() + gear_values.max())
            sled_mask = gear_values > gear_threshold
            trawl_mask = ~sled_mask

            if sled_mask.any():
                sled_metrics = compute_best_metrics(all_indiv_prob[sled_mask], all_label[sled_mask])
                print_metric_summary("sled", sled_metrics, int(sled_mask.sum()))
                for metric_name, metric_value in sled_metrics.items():
                    if np.isscalar(metric_value):
                        summary_writer.add_scalar(f'valid_sled/{metric_name}', metric_value, current_step)

            if trawl_mask.any():
                trawl_metrics = compute_best_metrics(all_indiv_prob[trawl_mask], all_label[trawl_mask])
                print_metric_summary("trawl", trawl_metrics, int(trawl_mask.sum()))
                for metric_name, metric_value in trawl_metrics.items():
                    if np.isscalar(metric_value):
                        summary_writer.add_scalar(f'valid_trawl/{metric_name}', metric_value, current_step)
        print("**********************************************")
        
        return acc, ha, ebf1, maf1, mif1, best_val_metrics

    acc, ha, ebf1, maf1, mif1, best_val_metrics = show_results(all_pred_x)

    summary_writer.add_scalar('valid/nll_loss', nll_loss, current_step)
    summary_writer.add_scalar('valid/l2_loss', l2_loss, current_step)
    summary_writer.add_scalar('valid/c_loss', c_loss, current_step)
    summary_writer.add_scalar('valid/total_loss',total_loss, current_step)
    summary_writer.add_scalar('valid/aux_group_loss', aux_loss, current_step)
    summary_writer.add_scalar('valid/gear_flip_loss', gear_loss, current_step)
    summary_writer.add_scalar('valid/macro_f1', maf1, current_step)
    summary_writer.add_scalar('valid/micro_f1', mif1, current_step)
    summary_writer.add_scalar('valid/exam_f1', ebf1, current_step)
    summary_writer.add_scalar('valid/acc', acc, current_step)
    summary_writer.add_scalar('valid/ha', ha, current_step)

    vae.train()

    return total_loss, best_val_metrics
