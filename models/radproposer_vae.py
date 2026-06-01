import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace

from common import HighResolution3DNet, ResidualMLP, blocks_dict
from hrnet3D_config import MODEL_CONFIGS
from transformer import Transformer

sys.path.append(str(Path(__file__).parent.parent))
from tools.config import MODELNAME, NUM_DIMS


class RadProPoserVAE(nn.Module):
    """VAE-based radar pose estimator with Gaussian latent space and ensemble sampling."""

    def __init__(self) -> None:
        super(RadProPoserVAE, self).__init__()

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

        # latent space
        self.mu = nn.Sequential(nn.Linear(2048, 256))
        self.spread = nn.Sequential(nn.Linear(2048, 256))
        self.variance_scaling = nn.Parameter(torch.tensor(0.1))
        self.n_samples = 500

        # specify latent distribution
        if MODELNAME in ["RPPgaussianGaussian", "RPPgaussianLaplace"]:
            self.latent_dist = "gaussian"
        elif MODELNAME in ["RPPlaplaceLaplace", "RPPlaplaceGaussian"]:
            self.latent_dist = "laplace"
        else:
            warnings.warn(
                "Invalid model configuration detected. "
                "Check the model name and likelihood combination. "
                "This may lead to incorrect training behavior.",
                category=UserWarning,
            )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, NUM_DIMS),
        )

    def sample(
        self, mu: torch.Tensor, logvar: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        """Reparameterization trick for Gaussian latent space.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
            device: Device to place noise tensor on.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device) * self.variance_scaling
        return mu + eps * std

    def sample_laplace(
        self, mu: torch.Tensor, b_raw: torch.Tensor
    ) -> torch.Tensor:
        """Sample from Laplace(mu, b) with softplus-constrained scale.

        Args:
            mu: Mean tensor, shape (..., latent_dim).
            b_raw: Raw (unconstrained) scale tensor, shape (..., latent_dim).

        Returns:
            Differentiable sample from Laplace(mu, b), same shape as mu.
        """
        b = torch.nn.functional.softplus(b_raw) + 1e-6
        dist = Laplace(loc=mu, scale=b)
        return dist.rsample()

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

    def sample_latent(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample from latent space with optional variance scaling override.

        Args:
            mu: Latent mean.
            logvar: Latent log variance.
            alpha: Optional override for variance scaling parameter.

        Returns:
            Sampled latent vector.
        """
        if alpha is not None:
            self.variance_scaling = torch.tensor(alpha)
        out = self.sample(mu, logvar)
        return out

    def get_latent(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode preprocessed radar input to latent distribution parameters.

        Args:
            x: Preprocessed complex radar tensor.

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        device = x.device

        x_comp = [x.real, x.imag]
        spat_feat = []
        for elmt in x_comp:
            for i in range(elmt.size(1)):
                helper = self.apply_backbone(elmt[:, i].float())
                spat_feat.append(helper)

        spatiotemp = torch.cat(spat_feat, dim=1)

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        return mu, logvar

    def forward(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ]:
        """Full forward pass: preprocess, encode, sample, decode.

        Args:
            x: Raw radar tensor of shape (B, seq_len, range, azimuth, elevation, doppler).
            return_samples: If True, return full sample tensor.

        Returns:
            Tuple of (samples_or_None, mu, latent_spread, mu_out, var_out).
        """
        x = self.pre_process(x)
        device = x.device

        x_comp = [x.real, x.imag]
        spat_feat = []
        for elmt in x_comp:
            for i in range(elmt.size(1)):
                helper = self.apply_backbone(elmt[:, i].float())
                spat_feat.append(helper)

        spatiotemp = torch.stack(spat_feat, dim=1)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        mu = self.mu(spatiotemp)

        if self.latent_dist == "laplace":
            b_latent = self.spread(spatiotemp)
            samples = self.decoder(torch.stack(
                [self.sample_laplace(mu, b_latent)
                 for i in range(self.n_samples)], dim=1))

            mu_out = torch.mean(samples, dim=1)
            var_out = torch.var(samples, dim=1)

            samples = samples.permute(0, 2, 1)

            if return_samples:
                return samples, mu, b_latent, mu_out, var_out
            else:
                return None, mu, b_latent, mu_out, var_out

        else:  # gaussian
            logvar = self.spread(spatiotemp)
            samples = self.decoder(torch.stack(
                [self.sample(mu, logvar, device=device)
                 for i in range(self.n_samples)], dim=1))

            mu_out = torch.mean(samples, dim=1)
            var_out = torch.var(samples, dim=1)

            samples = samples.permute(0, 2, 1)

            if return_samples:
                return samples, mu, logvar, mu_out, var_out
            else:
                return None, mu, logvar, mu_out, var_out

    def forward_inference(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ]:
        """Optimized forward for inference with batch_size=1.

        Batches all backbone calls into a single forward pass and uses
        vectorized sampling with batched decoding.

        Args:
            x: Raw radar tensor of shape (1, seq_len, range, azimuth, elevation, doppler).
            return_samples: If True, return full sample tensor.

        Returns:
            Tuple of (samples_or_None, mu, latent_spread, mu_out, var_out).
        """
        x = self.pre_process(x)
        device = x.device

        real_part = x.real.squeeze(0)
        imag_part = x.imag.squeeze(0)
        stacked_input = torch.cat([real_part, imag_part], dim=0)

        spat_feat = self.apply_backbone(stacked_input.float())

        spatiotemp = spat_feat.unsqueeze(0)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        mu = self.mu(spatiotemp)
        spread_param = self.spread(spatiotemp)

        if self.latent_dist == "laplace":
            b_latent = spread_param
            mu_exp = mu.expand(self.n_samples, -1)
            b_exp = b_latent.expand(self.n_samples, -1)

            u = torch.rand_like(mu_exp)
            z = mu_exp - b_exp * torch.sign(u - 0.5) * torch.log1p(
                -2 * torch.abs(u - 0.5))

            decoded = self.decoder(z)

            samples = decoded.unsqueeze(0)
            mu_out = samples.mean(dim=1)
            var_out = samples.var(dim=1)
            samples = samples.permute(0, 2, 1)

            if return_samples:
                return samples, mu, b_latent, mu_out, var_out
            else:
                return None, mu, b_latent, mu_out, var_out

        else:  # Gaussian
            logvar = spread_param
            std = torch.exp(0.5 * logvar)

            mu_exp = mu.expand(self.n_samples, -1)
            std_exp = std.expand(self.n_samples, -1)

            eps = torch.randn_like(mu_exp)
            z = mu_exp + std_exp * eps

            decoded = self.decoder(z)

            samples = decoded.unsqueeze(0)
            mu_out = samples.mean(dim=1)
            var_out = samples.var(dim=1)
            samples = samples.permute(0, 2, 1)

            if return_samples:
                return samples, mu, logvar, mu_out, var_out
            else:
                return None, mu, logvar, mu_out, var_out
