import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HighResolution3DNet, blocks_dict
from hrnet3D_config_4D import MODEL_CONFIGS as cfg_4D

sys.path.append(str(Path(__file__).parent.parent))
from tools.config import NUM_COORDS, NUM_DIMS, NUM_JOINTS


class HRRadarPose(nn.Module):
    """Deterministic high-resolution radar pose estimator without uncertainty.

    Single-frame baseline model based on Ho et al. that predicts root joint
    position and offsets for remaining joints.
    """

    def __init__(self) -> None:
        super(HRRadarPose, self).__init__()

        # backbone
        self.backbone = HighResolution3DNet(
            cfg_4D["hr_tiny_feat64_zyx_l4_in64"], full_res_stem=True)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1, 8), stride=(1, 1, 8)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=8,
                      kernel_size=(1, 1, 8), stride=(1, 1, 8)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, NUM_DIMS),
        )

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mean removal and 4D FFT preprocessing to radar input.

        Args:
            x: Raw radar tensor.

        Returns:
            Preprocessed complex tensor with permuted dimensions.
        """
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = torch.fft.fft(
            torch.fft.fft(
                torch.fft.fft(
                    torch.fft.fft(x, dim=-1, norm="forward"),
                    dim=-2, norm="forward"),
                dim=-3, norm="forward"),
            dim=-4, norm="forward")
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def apply_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through the HRNet backbone.

        Args:
            x: Input tensor.

        Returns:
            First-scale feature map from HRNet.
        """
        out = self.backbone(x)
        return out[0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting root position and joint offsets.

        Args:
            x: Raw radar tensor.

        Returns:
            Tuple of (root, offsets) where root is (B, 3) and offsets is (B, 25, 3).
        """
        x = self.pre_process(x)
        x = torch.cat([x.real, x.imag], dim=1)
        x = self.bottleneck(self.apply_backbone(x))
        x = x.reshape(x.shape[0], NUM_JOINTS, NUM_COORDS)
        root = x[:, 0]
        offsets = x[:, 1:]
        return root, offsets

    def forward_inference(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for inference with batch_size=1.

        This model already processes the backbone in a single pass
        (concat along channel dim rather than sequential loop).

        Args:
            x: Raw radar tensor of shape (1, ...).

        Returns:
            Tuple of (root, offsets).
        """
        x = self.pre_process(x)
        x = torch.cat([x.real, x.imag], dim=1)
        x = self.bottleneck(self.apply_backbone(x))
        x = x.reshape(x.shape[0], NUM_JOINTS, NUM_COORDS)
        root = x[:, 0]
        offsets = x[:, 1:]
        return root, offsets
