import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log_normal, log_normal_mixture
from losses.distribution_balanced import DistributionBalancedBCELoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.keep_prob)
        self.gear_feature_idx = getattr(args, "gear_feature_idx", None)
        self.use_gear_residual = bool(getattr(args, "gear_residualize", False) and self.gear_feature_idx is not None)
        self.use_gear_interaction = bool(getattr(args, "gear_interaction_design", False) and self.gear_feature_idx is not None)
        if self.use_gear_residual and self.use_gear_interaction:
            raise ValueError("gear_residualize and gear_interaction_design are mutually exclusive.")
        self.use_separated_gear = self.use_gear_residual or self.use_gear_interaction
        self.main_feature_dim = args.feature_dim - 1 if self.use_separated_gear else args.feature_dim
        
        """Feature encoder"""
        self.fx = nn.Sequential(
            nn.Linear(self.main_feature_dim, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)

        """Label encoder"""
        self.label_lookup = nn.Linear(args.label_dim, args.emb_size)
        self.fe = nn.Sequential(
            nn.Linear(args.emb_size, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)

        """Decoder"""
        self.fd = nn.Sequential(
            nn.Linear(self.main_feature_dim + args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.emb_size),
            nn.LeakyReLU()
        )
        if self.use_gear_residual:
            self.gear_residual_head = nn.Sequential(
                nn.Linear(args.latent_dim + 1, getattr(args, "gear_residual_hidden_dim", 16)),
                nn.ReLU(),
                nn.Linear(getattr(args, "gear_residual_hidden_dim", 16), args.label_dim),
            )
        if self.use_gear_interaction:
            interaction_hidden = getattr(args, "gear_interaction_hidden_dim", 16)
            self.gear_interaction_gate = nn.Sequential(
                nn.Linear(args.latent_dim, interaction_hidden),
                nn.ReLU(),
                nn.Linear(interaction_hidden, args.label_dim),
            )

    def split_feature_inputs(self, feature):
        if not self.use_separated_gear:
            return feature, None
        gear_feat = feature[:, self.gear_feature_idx:self.gear_feature_idx + 1]
        ecology_feat = torch.cat(
            [feature[:, :self.gear_feature_idx], feature[:, self.gear_feature_idx + 1:]],
            dim=1,
        )
        return ecology_feat, gear_feat

    def gear_residual_logits(self, latent, gear_feat):
        if not self.use_gear_residual or gear_feat is None:
            return None
        return self.gear_residual_head(torch.cat((latent, gear_feat), dim=1))

    def gear_interaction_logits(self, latent, gear_feat):
        if not self.use_gear_interaction or gear_feat is None:
            return None
        centered_gear = (gear_feat * 2.0) - 1.0
        return self.gear_interaction_gate(latent) * centered_gear

    def label_encode(self, x):
        h0 = self.dropout(F.relu(self.label_lookup(x)))
        h = self.fe(h0)
        mu = self.fe_mu(h)
        logvar = self.fe_logvar(h)
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h = self.fx(x)
        mu = self.fx_mu(h)
        logvar = self.fx_logvar(h)
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        return fx_output

    def decode(self, z):
        d = self.fd(z)
        d = F.normalize(d, dim=1)
        return d

    def label_forward(self, x, feat, gear_feat=None):
        n_label = x.shape[1]
        all_labels = torch.eye(n_label).to(device)
        fe_output = self.label_encode(all_labels)
        mu = fe_output['fe_mu']
        
        z = torch.matmul(x, mu) / x.sum(1, keepdim=True)
        label_emb = self.decode(torch.cat((feat, z), 1))
        label_residual = self.gear_residual_logits(z, gear_feat)
        label_interaction = self.gear_interaction_logits(z, gear_feat)

        fe_output['label_emb'] = label_emb
        fe_output['label_residual_logits'] = label_residual
        fe_output['label_interaction_logits'] = label_interaction
        return fe_output

    def feat_forward(self, x, gear_feat=None):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']
        logvar = fx_output['fx_logvar']

        if not self.training:
            z = mu
            z2 = mu
        else:
            z = reparameterize(mu, logvar)
            z2 = reparameterize(mu, logvar)
        feat_emb = self.decode(torch.cat((x, z), 1))
        feat_emb2 = self.decode(torch.cat((x, z2), 1))
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2
        fx_output['feat_residual_logits'] = self.gear_residual_logits(mu, gear_feat)
        fx_output['feat_residual_logits2'] = self.gear_residual_logits(z2 if self.training else mu, gear_feat)
        fx_output['feat_interaction_logits'] = self.gear_interaction_logits(mu, gear_feat)
        fx_output['feat_interaction_logits2'] = self.gear_interaction_logits(z2 if self.training else mu, gear_feat)
        return fx_output
    
    def forward(self, label, feature):
        ecology_feat, gear_feat = self.split_feature_inputs(feature)
        fe_output = self.label_forward(label, ecology_feat, gear_feat)
        label_emb = fe_output['label_emb']
        

        fx_output = self.feat_forward(ecology_feat, gear_feat)
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']

        embs = self.label_lookup.weight
        label_out = torch.matmul(label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)
        feat_out2 = torch.matmul(feat_emb2, embs)
        if self.use_gear_residual:
            residual_scale = float(getattr(self.args, "gear_residual_scale", 0.25))
            label_out = label_out + residual_scale * fe_output['label_residual_logits']
            feat_out = feat_out + residual_scale * fx_output['feat_residual_logits']
            feat_out2 = feat_out2 + residual_scale * fx_output['feat_residual_logits2']
        if self.use_gear_interaction:
            interaction_scale = float(getattr(self.args, "gear_interaction_scale", 0.25))
            label_out = label_out + interaction_scale * fe_output['label_interaction_logits']
            feat_out = feat_out + interaction_scale * fx_output['feat_interaction_logits']
            feat_out2 = feat_out2 + interaction_scale * fx_output['feat_interaction_logits2']
        
        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs
        output['label_out'] = label_out
        output['feat_out'] = feat_out
        output['feat_out2'] = feat_out2
        output['feat'] = feature
        output['ecology_feat'] = ecology_feat
        output['gear_feat'] = gear_feat
        return output


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


# def compute_loss(input_label, output, args=None):
#     fe_out, fe_mu, fe_logvar, label_emb = \
#         output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
#     fx_out, fx_mu, fx_logvar, feat_emb = \
#         output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
#     fx_out2 = output['feat_out2']
#     embs = output['embs']

#     fx_sample = reparameterize(fx_mu, fx_logvar)
#     fx_var = torch.exp(fx_logvar)
#     fe_var = torch.exp(fe_logvar)
#     kl_loss = (log_normal(fx_sample, fx_mu, fx_var) - \
#         log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

#     pred_e = torch.sigmoid(fe_out)
#     pred_x = torch.sigmoid(fx_out)
#     pred_x2 = torch.sigmoid(fx_out2)

#     def compute_BCE_and_RL_loss(E):
#         #compute negative log likelihood (BCE loss) for each sample point
#         sample_nll = -(
#             torch.log(E) * input_label + torch.log(1 - E) * (1 - input_label)
#         )
#         logprob = -torch.sum(sample_nll, dim=2)

#         #the following computation is designed to avoid the float overflow (log_sum_exp trick)
#         maxlogprob = torch.max(logprob, dim=0)[0]
#         Eprob = torch.mean(torch.exp(logprob - maxlogprob), axis=0)
#         nll_loss = torch.mean(-torch.log(Eprob) - maxlogprob)
#         return nll_loss

#     def supconloss(label_emb, feat_emb, embs, temp=1.0):
#         features = torch.cat((label_emb, feat_emb))
#         labels = torch.cat((input_label, input_label)).float()
#         n_label = labels.shape[1]
#         emb_labels = torch.eye(n_label).to(device)
#         mask = torch.matmul(labels, emb_labels)

#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, embs),
#             temp)
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         exp_logits = torch.exp(logits)
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#         loss = -mean_log_prob_pos
#         loss = loss.mean()
#         return loss

#     nll_loss = compute_BCE_and_RL_loss(pred_e.unsqueeze(0))
#     nll_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
#     nll_loss_x2 = compute_BCE_and_RL_loss(pred_x2.unsqueeze(0))
#     sum_nll_loss = nll_loss + nll_loss_x + nll_loss_x2
#     cpc_loss = supconloss(label_emb, feat_emb, embs)
#     total_loss = sum_nll_loss * args.nll_coeff + kl_loss * 6. + cpc_loss
#     return total_loss, nll_loss, nll_loss_x, 0., 0., kl_loss, cpc_loss, pred_e, pred_x

def compute_loss(input_label, output, args=None):
    eps = 1e-6  # small constant to prevent log(0) and division by 0
    
    # Unpack output
    fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
    fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
    fx_out2 = output['feat_out2']
    embs = output['embs']

    # Clamp logvars to avoid extreme values
    fe_logvar = torch.clamp(fe_logvar, min=-10, max=10)
    fx_logvar = torch.clamp(fx_logvar, min=-10, max=10)

    fx_sample = reparameterize(fx_mu, fx_logvar)
    fx_var = torch.exp(fx_logvar)
    fe_var = torch.exp(fe_logvar)
    if torch.isnan(fe_logvar).any():
        print("NaN in fe_logvar at source!", fe_logvar)
       


    # KL divergence
    kl_loss = (log_normal(fx_sample, fx_mu, fx_var) - log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

    # Predictions (sigmoid applied)
    pred_e = torch.clamp(torch.sigmoid(fe_out), min=eps, max=1 - eps)
    pred_x = torch.clamp(torch.sigmoid(fx_out), min=eps, max=1 - eps)
    pred_x2 = torch.clamp(torch.sigmoid(fx_out2), min=eps, max=1 - eps)

    db_criterion = None
    if args is not None and getattr(args, "use_db_loss", False) and hasattr(args, "class_freq_tensor"):
        db_criterion = DistributionBalancedBCELoss(
            class_freq=args.class_freq_tensor.to(fe_out.device),
            train_num=int(args.train_sample_count),
            map_alpha=float(getattr(args, "db_map_alpha", 0.1)),
            map_beta=float(getattr(args, "db_map_beta", 10.0)),
            map_gamma=float(getattr(args, "db_map_gamma", 0.9)),
            neg_scale=float(getattr(args, "db_neg_scale", 2.0)),
            init_bias_scale=float(getattr(args, "db_init_bias_scale", 0.05)),
        ).to(fe_out.device)

    # BCE and RL Loss with clamping
    def compute_BCE_and_RL_loss(E):
        sample_nll = -(
            torch.log(torch.clamp(E, min=eps)) * input_label + torch.log(torch.clamp(1 - E, min=eps)) * (1 - input_label)
        )
        logprob = -torch.sum(sample_nll, dim=2)
        maxlogprob = torch.max(logprob, dim=0)[0]
        Eprob = torch.mean(torch.exp(logprob - maxlogprob), dim=0)
        nll_loss = torch.mean(-torch.log(Eprob + eps) - maxlogprob)
        return nll_loss

    def compute_db_loss(logits):
        return db_criterion(logits, input_label)

    # Supervised contrastive loss with stability
    def supconloss(label_emb, feat_emb, embs, temp=1.0):
        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(features.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(torch.matmul(features, embs), temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # avoid division by zero
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + eps)

        loss = -mean_log_prob_pos
        return loss.mean()

    # Compute all losses
    if db_criterion is not None:
        nll_loss = compute_db_loss(fe_out)
        nll_loss_x = compute_db_loss(fx_out)
        nll_loss_x2 = compute_db_loss(fx_out2)
    else:
        nll_loss = compute_BCE_and_RL_loss(pred_e.unsqueeze(0))
        nll_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
        nll_loss_x2 = compute_BCE_and_RL_loss(pred_x2.unsqueeze(0))
    sum_nll_loss = nll_loss + nll_loss_x + nll_loss_x2

    cpc_loss = supconloss(label_emb, feat_emb, embs)

    # Combine final loss
    total_loss = sum_nll_loss * args.nll_coeff + kl_loss * 6. + cpc_loss

    return total_loss, nll_loss, nll_loss_x, 0., 0., kl_loss, cpc_loss, pred_e, pred_x
