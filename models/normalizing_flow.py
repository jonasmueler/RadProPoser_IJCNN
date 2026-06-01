import sys
from pathlib import Path
from typing import List, Optional, Tuple

import normflows as nf
from normflows import nets
from normflows.flows import (
    ActNorm, BatchNorm, Invertible1x1Conv,
    MaskedAffineFlow, Planar, Radial,
)
from normflows.nets import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HighResolution3DNet
from hrnet3D_config import MODEL_CONFIGS
from transformer import Transformer

sys.path.append(str(Path(__file__).parent.parent))
from tools.config import NUM_DIMS


class SimpleFlowModel(nn.Module):
    """Wrapper for normalizing flow transformations."""

    def __init__(self, flows: List[nn.Module]) -> None:
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sequence of flow transformations.

        Args:
            z: Input tensor of shape (B, D).

        Returns:
            Tuple of (transformed z, accumulated log determinant).
        """
        ld = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        return z, ld


class RadProPoserVAENF(nn.Module):
    """VAE-based radar pose estimator with normalizing flow posterior for flexible uncertainty modeling."""

    def __init__(self) -> None:
        super(RadProPoserVAENF, self).__init__()

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

        # latent space
        self.mu = nn.Sequential(nn.Linear(2048, 256))
        self.sigma = nn.Sequential(nn.Linear(2048, 256))
        self.variance_scaling = nn.Parameter(torch.tensor(0.1))

        self.former = Transformer()

        # decoder
        self.decoder = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, NUM_DIMS))

        # flow posterior setup with BatchNorm
        latent_dim = 256
        flows = []
        for i in range(4):
            mask = torch.tensor(
                [0, 1] * (latent_dim // 2) if i % 2 == 0
                else [1, 0] * (latent_dim // 2),
                dtype=torch.float32,
            )

            s = MLP([latent_dim, latent_dim], output_fn="tanh")
            t = MLP([latent_dim, latent_dim], output_fn="tanh")

            if i == 0:
                flows.append(ActNorm(latent_dim))

            flows.append(MaskedAffineFlow(mask, t, s))

        self.flows = SimpleFlowModel(flows)

    def sample(
        self, mu: torch.Tensor, logvar: torch.Tensor, device: str = "cpu"
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
        features = [
            F.interpolate(feature.float(), size=(4, 4, 64), mode="trilinear")
            for feature in feature_list
        ]
        features = torch.cat(features, dim=1)
        features = self.bottleneck(features)
        return features

    def sample_latent(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample from latent space with optional variance scaling override.

        Args:
            mu: Latent mean.
            sigma: Latent log variance.
            alpha: Optional override for variance scaling parameter.

        Returns:
            Sampled latent vector.
        """
        if alpha is not None:
            self.variance_scaling = torch.tensor(alpha)
        return self.sample(mu, sigma)

    def get_latent(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw radar input to latent distribution parameters.

        Args:
            x: Raw radar tensor (preprocessing is applied internally).

        Returns:
            Tuple of (mu, sigma) for the latent distribution.
        """
        x = self.pre_process(x)
        x_comp = [x.real, x.imag]
        spat_feat = [
            self.apply_backbone(elmt[:, i].float())
            for elmt in x_comp for i in range(elmt.size(1))
        ]
        spat_feat = torch.stack(spat_feat, dim=1)
        spat_feat = self.former(spat_feat)
        spatiotemp = spat_feat.flatten(start_dim=1, end_dim=-1)
        mu = self.mu(spatiotemp)
        sigma = self.sigma(spatiotemp)
        return mu, sigma

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with normalizing flow posterior.

        Args:
            x: Raw radar tensor.
            inference: If True, run MC sampling for uncertainty estimation.
                If False, run training mode with KLD computation.

        Returns:
            Training: (mu_out, kld).
            Inference: (mu_out, var_out, decoded_samples).
        """
        if not inference:
            mu, logvar = self.get_latent(x)
            std = torch.exp(0.5 * logvar)

            norm_scale = torch.randn_like(std)
            z0 = mu + norm_scale * std

            zK, log_det = self.flows(z0)
            zK = zK.squeeze()

            q0 = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            p = torch.distributions.Normal(0.0, 1.0)

            kld = (
                -torch.sum(p.log_prob(zK), dim=-1)
                + torch.sum(q0.log_prob(z0), dim=-1)
                - log_det.view(-1)
            ).mean()

            mu_out = self.decoder(zK)
            return mu_out, kld

        else:
            mu, logvar = self.get_latent(x)
            std = torch.exp(0.5 * logvar)

            n_samples = 500

            mu = mu.unsqueeze(1)
            std = std.unsqueeze(1)
            eps = torch.randn(
                mu.size(0), n_samples, mu.size(2), device=mu.device)

            z0 = mu + eps * std
            z0 = z0.view(-1, z0.size(-1))

            zK, _ = self.flows(z0)
            decoded = self.decoder(zK)

            decoded = decoded.view(mu.size(0), n_samples, -1)
            mu_out = decoded.mean(dim=1)
            var_out = decoded.var(dim=1)

            decoded = decoded.permute(0, 2, 1)

            return mu_out, var_out, decoded

    def forward_inference(
        self,
        x: torch.Tensor,
        n_samples: int = 500,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized forward for inference with batch_size=1.

        Batches all backbone calls into a single forward pass.
        Sampling is already vectorized.

        Args:
            x: Raw radar tensor of shape (1, seq_len, range, azimuth, elevation, doppler).
            n_samples: Number of MC samples.

        Returns:
            Tuple of (mu_out, var_out, decoded_samples).
        """
        x = self.pre_process(x)

        real_part = x.real.squeeze(0)
        imag_part = x.imag.squeeze(0)
        stacked_input = torch.cat([real_part, imag_part], dim=0)

        spat_feat = self.apply_backbone(stacked_input.float())

        spat_feat = spat_feat.unsqueeze(0)
        spat_feat = self.former(spat_feat)
        spatiotemp = spat_feat.flatten(start_dim=1, end_dim=-1)

        mu = self.mu(spatiotemp)
        logvar = self.sigma(spatiotemp)
        std = torch.exp(0.5 * logvar)

        mu = mu.unsqueeze(1)
        std = std.unsqueeze(1)
        eps = torch.randn(
            mu.size(0), n_samples, mu.size(2), device=mu.device)
        z0 = mu + eps * std
        z0 = z0.view(-1, z0.size(-1))

        zK, _ = self.flows(z0)
        decoded = self.decoder(zK)

        decoded = decoded.view(1, n_samples, -1)
        mu_out = decoded.mean(dim=1)
        var_out = decoded.var(dim=1)
        decoded = decoded.permute(0, 2, 1)

        return mu_out, var_out, decoded


if __name__ == "__main__":
    model = RadProPoserVAENF().float().cuda()

    test_data = torch.rand(1, 8, 4, 4, 64, 128).cuda()

    _, _, cdf = model.forward_inference(test_data)
    print(cdf.size())
