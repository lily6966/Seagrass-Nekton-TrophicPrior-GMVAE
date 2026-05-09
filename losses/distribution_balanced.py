import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionBalancedBCELoss(nn.Module):
    """Distribution-Balanced BCE loss for long-tailed multi-label data."""

    def __init__(
        self,
        class_freq: torch.Tensor,
        train_num: int,
        map_alpha: float = 0.1,
        map_beta: float = 10.0,
        map_gamma: float = 0.9,
        neg_scale: float = 2.0,
        init_bias_scale: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        class_freq = class_freq.float().clamp_min(1.0)
        self.register_buffer("class_freq", class_freq)
        self.train_num = float(train_num)
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma
        self.neg_scale = neg_scale
        self.init_bias_scale = init_bias_scale
        self.eps = eps

        base_bias = -torch.log(self.train_num / self.class_freq - 1.0 + self.eps)
        self.register_buffer("init_bias", base_bias * (self.init_bias_scale / self.neg_scale))
        self.register_buffer("inv_class_freq", self.train_num / self.class_freq)

    def _rebalance_weight(self, targets: torch.Tensor) -> torch.Tensor:
        pos_count = targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        repeat_rate = (targets * self.inv_class_freq.unsqueeze(0)).sum(dim=1, keepdim=True) / pos_count
        pos_weight = self.inv_class_freq.unsqueeze(0) / repeat_rate.clamp_min(self.eps)
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return torch.where(targets > 0, weight, torch.ones_like(weight))

    def _regularize_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        neg_mask = 1.0 - targets
        logits = logits * targets + logits * neg_mask * self.neg_scale
        logits = logits + neg_mask * self.init_bias.unsqueeze(0)
        return logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self._rebalance_weight(targets)
        logits = self._regularize_logits(logits, targets)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight, reduction="none")
        return loss.sum(dim=1).mean()
