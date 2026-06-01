from typing import Tuple

import numpy as np
import torch
from scipy.stats import norm

from misc.constants import JOINT_NAMES, NUM_JOINTS


def get_max_preds(batch_heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from score maps.

    Args:
        batch_heatmaps: Numpy array of shape (batch_size, num_joints, height, width)

    Returns:
        Tuple of (preds, maxvals) where preds is (batch_size, num_joints, 2)
    """
    assert isinstance(batch_heatmaps, np.ndarray), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def compute_mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Mean Per-Joint Position Error.

    Args:
        preds: Predictions tensor of shape (N, 51) or (N, 17, 3)
        targets: Ground truth tensor of shape (N, 51) or (N, 17, 3)

    Returns:
        Tuple of (per_joint_mpjpe, total_mpjpe) where per_joint is shape (17,)
    """
    preds = preds.view(-1, NUM_JOINTS, 3)
    targets = targets.view(-1, NUM_JOINTS, 3)

    diff = preds - targets
    dist = torch.norm(diff, dim=-1)

    per_joint = dist.mean(dim=0)
    total = dist.mean()

    return per_joint, total


def procrustes_align(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Align source to target using Procrustes analysis.

    Args:
        target: Target point cloud of shape (17, 3)
        source: Source point cloud to align of shape (17, 3)

    Returns:
        Aligned source tensor of shape (17, 3)
    """
    mu_target = target.mean(dim=0, keepdim=True)
    mu_source = source.mean(dim=0, keepdim=True)

    target_centered = target - mu_target
    source_centered = source - mu_source

    ss_target = (target_centered ** 2).sum()
    ss_source = (source_centered ** 2).sum()

    norm_target = torch.sqrt(ss_target)
    norm_source = torch.sqrt(ss_source)

    target_normalized = target_centered / norm_target
    source_normalized = source_centered / norm_source

    A = target_normalized.T @ source_normalized
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    V = Vt.T
    R = V @ U.T

    trace_TA = S.sum()
    aligned = norm_target * trace_TA * (source_normalized @ R) + mu_target

    return aligned


def compute_pa_mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Procrustes-Aligned Mean Per-Joint Position Error.

    Args:
        preds: Predictions tensor of shape (N, 51) or (N, 17, 3)
        targets: Ground truth tensor of shape (N, 51) or (N, 17, 3)

    Returns:
        Tuple of (per_joint_pa_mpjpe, total_pa_mpjpe) where per_joint is shape (17,)
    """
    preds = preds.view(-1, NUM_JOINTS, 3)
    targets = targets.view(-1, NUM_JOINTS, 3)

    aligned_preds = torch.stack([
        procrustes_align(tgt, pred) for pred, tgt in zip(preds, targets)
    ])

    diff = aligned_preds - targets
    dist = torch.norm(diff, dim=-1)

    per_joint = dist.mean(dim=0)
    total = dist.mean()

    return per_joint, total


def compute_sharpness(variances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sharpness (average predictive variance per joint).

    Args:
        variances: Diagonal variances of shape (N, 51) or (N, 17, 3)

    Returns:
        Tuple of (per_joint_sharpness, total_sharpness) where per_joint is shape (17,)
    """
    variances = variances.view(-1, NUM_JOINTS, 3)
    per_joint_var = variances.sum(dim=-1)
    per_joint = per_joint_var.mean(dim=0)
    total = per_joint.mean()

    return per_joint, total


def compute_ece(preds: np.ndarray, variances: np.ndarray, targets: np.ndarray,
                num_bins: int = 100) -> float:
    """Compute Expected Calibration Error.

    Args:
        preds: Predictions array of shape (N, 51)
        variances: Diagonal variances of shape (N, 51)
        targets: Ground truth array of shape (N, 51)
        num_bins: Number of bins for calibration

    Returns:
        ECE value (float)
    """
    sigma = np.sqrt(variances)
    cdf = norm.cdf(targets, loc=preds, scale=sigma)

    expected_p = np.linspace(0.01, 0.99, num_bins)
    flat_cdf = cdf.flatten()
    observed_p = np.array([(flat_cdf <= p).mean() for p in expected_p])

    ece = np.mean(np.abs(observed_p - expected_p))
    return ece


def print_pose_table(mpjpe_per_joint: torch.Tensor, pa_mpjpe_per_joint: torch.Tensor,
                     mpjpe_total: torch.Tensor, pa_mpjpe_total: torch.Tensor) -> None:
    """Print pose estimation results table.

    Args:
        mpjpe_per_joint: Per-joint MPJPE of shape (17,)
        pa_mpjpe_per_joint: Per-joint PA-MPJPE of shape (17,)
        mpjpe_total: Total MPJPE scalar
        pa_mpjpe_total: Total PA-MPJPE scalar
    """
    print("\n" + "=" * 80)
    print("POSE ESTIMATION PERFORMANCE (cm)")
    print("=" * 80)

    header = f"{'Metric':<12}"
    for name in JOINT_NAMES:
        header += f"{name:>8}"
    header += f"{'Total':>8}"
    print(header)
    print("-" * 80)

    mpjpe_row = f"{'MPJPE':<12}"
    for val in mpjpe_per_joint:
        mpjpe_row += f"{val.item():>8.3f}"
    mpjpe_row += f"{mpjpe_total.item():>8.3f}"
    print(mpjpe_row)

    pa_mpjpe_row = f"{'PA-MPJPE':<12}"
    for val in pa_mpjpe_per_joint:
        pa_mpjpe_row += f"{val.item():>8.3f}"
    pa_mpjpe_row += f"{pa_mpjpe_total.item():>8.3f}"
    print(pa_mpjpe_row)

    print("=" * 80)


def print_uncertainty_table(sharpness_per_joint: torch.Tensor, sharpness_total: torch.Tensor,
                            ece: float) -> None:
    """Print uncertainty metrics table.

    Args:
        sharpness_per_joint: Per-joint sharpness of shape (17,)
        sharpness_total: Total sharpness scalar
        ece: Expected Calibration Error
    """
    print("\n" + "=" * 80)
    print("UNCERTAINTY METRICS")
    print("=" * 80)

    header = f"{'Metric':<12}"
    for name in JOINT_NAMES:
        header += f"{name:>8}"
    header += f"{'Total':>8}"
    print(header)
    print("-" * 80)

    sharpness_row = f"{'Sharpness':<12}"
    for val in sharpness_per_joint:
        sharpness_row += f"{val.item():>8.3f}"
    sharpness_row += f"{sharpness_total.item():>8.3f}"
    print(sharpness_row)

    print("-" * 80)
    print(f"ECE: {ece:.3f}")
    print("=" * 80)
