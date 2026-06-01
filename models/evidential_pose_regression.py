import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma

from common import HighResolution3DNet
from hrnet3D_config import MODEL_CONFIGS
from transformer import Transformer

sys.path.append(str(Path(__file__).parent.parent))
from tools.config import NUM_DIMS


class RadProPoserEvidential(nn.Module):
    """Radar pose estimator using evidential deep learning for uncertainty quantification."""

    def __init__(self) -> None:
        super(RadProPoserEvidential, self).__init__()

        # backbone
        self.backbone = HighResolution3DNet(
            MODEL_CONFIGS["hr_tiny_feat64_zyx_l4_in64"], full_res_stem=True)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=384, out_channels=32,
                      kernel_size=(1, 1, 8), stride=(1, 1, 8)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=8,
                      kernel_size=(1, 1, 8), stride=(1, 1, 8)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        # transformer
        self.former = Transformer()

        self.out_uncertainty = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Linear(256, 128),
            NormalInvGamma(128, NUM_DIMS),
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
        x = x.permute(0, 1, 5, 2, 3, 4)
        return x

    def apply_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and fuse multi-scale features through the HRNet backbone.

        Args:
            x: Input tensor for a single frame.

        Returns:
            Flattened feature vector.
        """
        feature_list = self.backbone(x)

        features = []
        for feature in feature_list:
            helper = F.interpolate(
                feature.float(), size=(4, 4, 64), mode="trilinear")
            features.append(helper)

        features = torch.cat(features, dim=1)
        features = self.bottleneck(features)
        return features

    def sample_aleatoric_from_output(
        self,
        model_output: Tuple[torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor],
        n_samples: int = 500,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Draw aleatoric-only samples from Normal-Inverse-Gamma output.

        Args:
            model_output: Tuple of (mu, v, alpha, beta), each of shape (B, D).
            n_samples: Number of MC draws.
            eps: Small positive value to ensure alpha - 1 >= eps.

        Returns:
            Samples of shape (B, n_samples, D).
        """
        mu, v, alpha, beta = model_output

        alpha_clamped = torch.clamp(alpha, min=1.0 + eps)
        alea_var = beta / (alpha_clamped - 1.0)
        alea_std = torch.sqrt(alea_var)

        B, D = mu.shape
        mu_exp = mu.unsqueeze(1).expand(B, n_samples, D)
        std_exp = alea_std.unsqueeze(1).expand(B, n_samples, D)

        eps_noise = torch.randn_like(std_exp)
        return mu_exp + eps_noise * std_exp

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor
    ]:
        """Forward pass with evidential uncertainty estimation.

        Args:
            x: Raw radar tensor of shape (B, seq_len, range, azimuth, elevation, doppler).

        Returns:
            Tuple of (out_uncertainty, samples) where out_uncertainty is the NIG
            parameters and samples has shape (B, D, n_samples).
        """
        x = self.pre_process(x)

        x_comp = [x.real, x.imag]
        spat_feat = []
        for elmt in x_comp:
            for i in range(elmt.size(1)):
                helper = self.apply_backbone(elmt[:, i].float())
                spat_feat.append(helper)

        spatiotemp = torch.stack(spat_feat, dim=1)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        out_uncertainty = self.out_uncertainty(spatiotemp)
        samples = self.sample_aleatoric_from_output(out_uncertainty)

        return out_uncertainty, samples.permute(0, 2, 1)

    def forward_inference(
        self, x: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor
    ]:
        """Optimized forward for inference with batch_size=1.

        Batches all backbone calls into a single forward pass.

        Args:
            x: Raw radar tensor of shape (1, seq_len, range, azimuth, elevation, doppler).

        Returns:
            Tuple of (out_uncertainty, samples).
        """
        x = self.pre_process(x)

        real_part = x.real.squeeze(0)
        imag_part = x.imag.squeeze(0)
        stacked_input = torch.cat([real_part, imag_part], dim=0)

        spat_feat = self.apply_backbone(stacked_input.float())

        spatiotemp = spat_feat.unsqueeze(0)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        out_uncertainty = self.out_uncertainty(spatiotemp)
        samples = self.sample_aleatoric_from_output(out_uncertainty)

        return out_uncertainty, samples.permute(0, 2, 1)
