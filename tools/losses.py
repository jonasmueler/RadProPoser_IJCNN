"""Loss functions for RadProPoser training."""

from typing import Tuple

import numpy as np
import torch

from config import TRAINCONFIG


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between learned Gaussian posterior and standard normal prior.

    Args:
        mu: Latent mean tensor of shape (B, D).
        logvar: Latent log-variance tensor of shape (B, D).

    Returns:
        Scalar KL divergence loss averaged over the batch.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def nll_loss(
    gt: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simplified Negative Log-Likelihood loss assuming diagonal Gaussian.

    Args:
        gt: Ground truth tensor of shape (B, D).
        mu: Predicted mean tensor of shape (B, D).
        var: Predicted variance tensor of shape (B, D).

    Returns:
        Tuple of (NLL loss, variance penalty), both scalars.
    """
    var = torch.clamp(var, min=1e-6)
    pen = TRAINCONFIG["gamma"] * var.mean()
    loss = ((gt - mu) ** 2 / var).mean() + pen
    return loss, pen


def nll_from_cov(
    gt: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Negative log-likelihood loss for full covariance predictions.

    Args:
        gt: Ground truth of shape (B, K).
        mu: Predictive mean of shape (B, K).
        cov: Predictive covariance (SPD) of shape (B, K, K).

    Returns:
        Tuple of (NLL loss, covariance penalty), both scalars.
    """
    B, K = gt.shape
    eps = 1e-6
    eye = torch.eye(K, device=cov.device, dtype=cov.dtype)
    cov = cov + eps * eye

    L = torch.linalg.cholesky(cov)
    diff = (gt - mu).unsqueeze(-1)
    y = torch.linalg.solve_triangular(L, diff, upper=False)
    maha = (y.square()).sum(dim=(-2, -1))

    logdet = 2.0 * torch.log(torch.diagonal(L, 0, -2, -1)).sum(-1)

    nll = 0.5 * (maha + logdet + K * np.log(2.0 * np.pi))

    pen = TRAINCONFIG["gamma"] * cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)

    return nll.mean() + pen.mean(), pen.mean()


def nll_loss_precision(
    gt: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NLL loss using predicted log variance, converting to precision-based Gaussian NLL.

    Args:
        gt: Ground truth tensor.
        mu: Predicted mean tensor.
        log_variance: Predicted log variance (log(sigma^2)).

    Returns:
        Tuple of (NLL loss, precision penalty).
    """
    variance = log_variance
    precision = 1.0 / variance
    log_precision = torch.log(precision)

    pen = -(TRAINCONFIG["gamma"] * log_precision.mean())
    nll = (pen + precision * (gt - mu) ** 2).mean()
    return nll, pen


def laplace_nll(
    y_true: torch.Tensor, mu_pred: torch.Tensor, var_pred: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Laplace negative log-likelihood loss computed from empirical variance.

    Args:
        y_true: Ground truth tensor.
        mu_pred: Empirical mean of predicted samples.
        var_pred: Empirical variance of predicted samples.

    Returns:
        Tuple of (loss, penalty) averaged over batch and dimensions.
    """
    b = torch.sqrt(torch.clamp(var_pred, min=1e-10) / 2.0)
    abs_error = torch.abs(y_true - mu_pred)
    pen = TRAINCONFIG["gamma"] * b
    loss = pen + abs_error / b
    return loss.mean(), pen.mean()
