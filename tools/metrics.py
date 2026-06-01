"""Evaluation metrics for RadProPoser pose estimation."""

import torch

from config import NUM_COORDS, NUM_JOINTS


def mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate the Mean Per Joint Position Error (MPJPE).

    Args:
        preds: Predicted joint positions, shape (batch_size, num_joints * 3)
            or (batch_size, num_joints, 3).
        targets: Ground truth joint positions, same shape as preds.

    Returns:
        Scalar MPJPE for the batch (mean over joints and batch).
    """
    assert preds.shape == targets.shape, (
        "Predictions and targets must have the same shape"
    )

    diff = (
        preds.reshape(preds.size(0), NUM_JOINTS, NUM_COORDS)
        - targets.reshape(targets.size(0), NUM_JOINTS, NUM_COORDS)
    )
    dist = torch.norm(diff, dim=-1)
    return dist.mean()
