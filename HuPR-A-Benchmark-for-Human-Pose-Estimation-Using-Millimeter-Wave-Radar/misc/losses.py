import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TRAINCONFIG
from misc.metrics import get_max_preds
from misc.utils import generateTarget


def KLLoss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence for VAE latent space regularization."""
    klloss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return klloss


def nll_from_cov(gt: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor):
    """
    Negative log-likelihood using full covariance matrix.

    Args:
        gt  : (B, K)     ground truth
        mu  : (B, K)     predictive mean
        cov : (B, K, K)  predictive covariance (SPD)

    Returns:
        nll_loss: scalar NLL loss
        pen: variance penalty term
    """
    B, K = gt.shape
    eye = torch.eye(K, device=cov.device, dtype=cov.dtype)

    # Add jitter for numerical stability (1e-4 too small for some batches)
    eps = 1e-3
    cov_reg = cov + eps * eye
    L = torch.linalg.cholesky(cov_reg)

    diff = (gt - mu).unsqueeze(-1)  # (B, K, 1)
    y = torch.linalg.solve_triangular(L, diff, upper=False)
    maha = (y.square()).sum(dim=(-2, -1))  # Mahalanobis distance (B,)

    logdet = 2.0 * torch.log(torch.diagonal(L, 0, -2, -1)).sum(-1)

    nll = 0.5 * (maha + logdet + K * np.log(2.0 * np.pi))  # (B,)

    pen = TRAINCONFIG["gamma"] * cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (B,)

    return nll.mean() + pen.mean(), pen.mean()


class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.numFrames = self.cfg.DATASET.numFrames
        self.numGroupFrames = self.cfg.DATASET.numGroupFrames
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.lossDecay = self.cfg.TRAINING.lossDecay
        self.alpha = 0.0
        self.beta = 1.0
        self.bce = nn.BCELoss()
    
    def computeLoss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numKeypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.numKeypoints, 2))
        for i in range(len(gt)):
            heatmap, gtKpt = generateTarget(gt[i], self.numKeypoints, self.heatmapSize, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
        preds, preds2 = preds      
        loss1 = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
        loss2 = self.computeBCESingleFrame(preds2.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
        preds2 = preds2.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
        if self.alpha < 1.0:
            self.alpha += self.lossDecay
            self.beta -= self.lossDecay
        if self.lossDecay != -1:
            loss = self.alpha * loss1 + self.beta * loss2
        else:
            loss = loss1 + loss2
        pred2d, _ = get_max_preds(preds2.detach().cpu().numpy())
        gt2d, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        return loss, loss2, pred2d, gt2d

    def computeBCESingleFrame(self, preds, gt):
        loss = self.bce(preds, gt.to(self.device))
        return loss