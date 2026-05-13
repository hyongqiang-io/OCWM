"""JEPA loss function and lambda_rec annealing scheduler."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """JEPA latent prediction loss.

    L_jepa = ||z_hat_{t+1} - sg(z_bar_{t+1})||^2

    where z_bar is the EMA target encoder output (stop-gradient).
    """

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, z_predicted, z_target):
        """
        Args:
            z_predicted: predicted next-step latent from dynamics model
                         Can be dict (particle attributes) or tensor
            z_target: EMA encoder output for next step (already detached)
                      Same format as z_predicted

        Returns:
            scalar loss
        """
        if isinstance(z_predicted, dict) and isinstance(z_target, dict):
            return self._dict_loss(z_predicted, z_target)
        return self._tensor_loss(z_predicted, z_target)

    def _tensor_loss(self, pred, target):
        target = target.detach()
        if self.normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)
        return F.mse_loss(pred, target)

    def _dict_loss(self, pred, target):
        """Compute JEPA loss over particle attribute dictionaries."""
        total_loss = 0.0
        n_terms = 0
        # Match on common keys representing particle attributes
        attr_keys = ['z', 'z_features', 'z_depth', 'z_scale', 'z_obj_on', 'z_bg_features']
        for key in attr_keys:
            if key in pred and key in target:
                p = pred[key]
                t = target[key].detach()
                if p.shape == t.shape:
                    total_loss = total_loss + F.mse_loss(p, t)
                    n_terms += 1
        if n_terms == 0:
            raise ValueError("No matching keys found between predicted and target dicts")
        return total_loss / n_terms


class AnnealingScheduler:
    """Linear annealing scheduler for lambda_rec.

    Schedule:
        epoch < start: value = start_value
        start <= epoch <= end: linear interpolation
        epoch > end: value = end_value
    """

    def __init__(self, start_value=1.0, end_value=0.0,
                 start_epoch=30, end_epoch=80):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def get_value(self, epoch):
        """Get annealed value for given epoch."""
        if epoch < self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        # Linear interpolation
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def __repr__(self):
        return (f"AnnealingScheduler(start={self.start_value}→{self.end_value}, "
                f"epochs={self.start_epoch}→{self.end_epoch})")


class CombinedLoss:
    """Combined training loss for LPWM-GNN-Mamba3.

    L_total = lambda_jepa * L_jepa
            + lambda_cv * L_cross_view
            + beta_kl * L_KL
            + lambda_rec(epoch) * L_rec
    """

    def __init__(self, lambda_jepa=1.0, lambda_cross_view=1.0, beta_kl=0.08,
                 lambda_rec_start=1.0, lambda_rec_end=0.0,
                 anneal_start=30, anneal_end=80):
        self.lambda_jepa = lambda_jepa
        self.lambda_cross_view = lambda_cross_view
        self.beta_kl = beta_kl
        self.jepa_loss = JEPALoss()
        self.rec_scheduler = AnnealingScheduler(
            start_value=lambda_rec_start, end_value=lambda_rec_end,
            start_epoch=anneal_start, end_epoch=anneal_end)

    def compute(self, epoch, l_jepa=None, l_cross_view=None, l_kl=None, l_rec=None):
        """Compute combined loss.

        Args:
            epoch: current training epoch
            l_jepa: JEPA prediction loss (scalar tensor)
            l_cross_view: cross-view reconstruction loss (scalar tensor)
            l_kl: KL divergence loss (scalar tensor)
            l_rec: pixel reconstruction loss (scalar tensor)

        Returns:
            total_loss: combined scalar loss
            loss_dict: dict of individual weighted losses for logging
        """
        lambda_rec = self.rec_scheduler.get_value(epoch)

        total = torch.tensor(0.0, device=l_jepa.device if l_jepa is not None else 'cpu')
        loss_dict = {'lambda_rec': lambda_rec}

        if l_jepa is not None:
            weighted_jepa = self.lambda_jepa * l_jepa
            total = total + weighted_jepa
            loss_dict['l_jepa'] = l_jepa.item()
            loss_dict['l_jepa_weighted'] = weighted_jepa.item()

        if l_cross_view is not None:
            weighted_cv = self.lambda_cross_view * l_cross_view
            total = total + weighted_cv
            loss_dict['l_cross_view'] = l_cross_view.item()

        if l_kl is not None:
            weighted_kl = self.beta_kl * l_kl
            total = total + weighted_kl
            loss_dict['l_kl'] = l_kl.item()

        if l_rec is not None:
            weighted_rec = lambda_rec * l_rec
            total = total + weighted_rec
            loss_dict['l_rec'] = l_rec.item()
            loss_dict['l_rec_weighted'] = weighted_rec.item()

        loss_dict['total'] = total.item()
        return total, loss_dict
