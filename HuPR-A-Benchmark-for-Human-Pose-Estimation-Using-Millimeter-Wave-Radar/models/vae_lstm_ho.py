"""RadProPoser VAE with covariance estimation for HuPR radar pose prediction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

from misc.constants import NUM_DIMS
from models.common import HighResolution3DNet, blocks_dict
from models.hrnet3D_config import MODEL_CONFIGS
from models.transformer import Transformer


class RadProPoserVAECov(nn.Module):
    """RadProPoser VAE model with covariance estimation for pose prediction."""

    def __init__(self, n_samples: int = 500) -> None:
        super(RadProPoserVAECov, self).__init__()

        # backbone
        self.backbone = HighResolution3DNet(
            MODEL_CONFIGS["hr_tiny_feat64_zyx_l4_in64"], full_res_stem=True
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(
                in_channels=384, out_channels=32,
                kernel_size=(1, 1, 8), stride=(1, 1, 8),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=32, out_channels=8,
                kernel_size=(1, 1, 8), stride=(1, 1, 8),
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        # transformer - max_seq_len supports up to 64 (2*16*2), dynamically sliced
        self.former = Transformer(seq_len=64)

        # latent space - input dim = 2 * T * R * 128 = 2 * 8 * 2 * 128 = 4096
        latent_input_dim = 4096
        self.mu = nn.Sequential(nn.Linear(latent_input_dim, 512))
        self.spread = nn.Sequential(nn.Linear(latent_input_dim, 512))
        self.variance_scaling = nn.Parameter(torch.tensor(0.1))
        self.n_samples = n_samples
        self.latent_dist = "gaussian"

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, NUM_DIMS),
        )

    def sample(
        self, mu: torch.Tensor, logvar: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        """Sample from Gaussian latent distribution using reparameterization trick.

        Args:
            mu: Mean tensor of shape (B, latent_dim).
            logvar: Log-variance tensor of shape (B, latent_dim).
            device: Device to create noise tensor on.

        Returns:
            Sampled latent tensor of shape (B, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device) * self.variance_scaling
        return mu + eps * std

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        """Apply clutter removal and 4D FFT preprocessing.

        Args:
            x: Input radar tensor.

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
        """Apply backbone network and bottleneck to extract features.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Flattened feature tensor.
        """
        feature_list = self.backbone(x)

        features = []
        for feature in feature_list:
            helper = F.interpolate(
                feature.float(), size=(4, 4, 64), mode="trilinear"
            )
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
            mu: Mean tensor.
            logvar: Log-variance tensor.
            alpha: Optional variance scaling factor.

        Returns:
            Sampled latent tensor.
        """
        if alpha is not None:
            self.variance_scaling = torch.tensor(alpha)
        return self.sample(mu, logvar)

    def ensemble_stats(
        self, preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and covariance from ensemble predictions.

        Args:
            preds: Tensor of shape (B, K, E) - batch x features x ensemble samples.

        Returns:
            Tuple of (mu, cov) where mu is (B, K) and cov is (B, K, K).
        """
        mu = preds.mean(dim=2)
        cov = vmap(lambda x: torch.cov(x, correction=1))(preds)
        return mu, cov

    def ensemble_stats_var(
        self, preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance from ensemble predictions.

        Args:
            preds: Tensor of shape (B, K, E) - batch x features x ensemble samples.

        Returns:
            Tuple of (mu, var) where both are (B, K).
        """
        mu = preds.mean(dim=2)
        var = preds.var(dim=2, unbiased=True)
        return mu, var

    def get_latent(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent distribution parameters from preprocessed input.

        Args:
            x: Preprocessed complex radar tensor.

        Returns:
            Tuple of (mu, logvar) tensors.
        """
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
        """Forward pass with ensemble sampling.

        Args:
            x: Input radar tensor (B, T, R, D, TX, RX, Range).
            return_samples: Whether to return raw samples.

        Returns:
            Tuple of (samples_or_None, mu, logvar, mu_out, cov).
        """
        device = x.device
        B, T, R = x.shape[0], x.shape[1], x.shape[2]

        # Stack real/imag and reshape for batched backbone
        x_stacked = torch.stack([x.real, x.imag], dim=1)
        x_stacked = x_stacked.reshape(
            B * 2 * T * R, *x_stacked.shape[4:]
        ).float()

        # Single batched backbone call instead of 2*T*R sequential calls
        spat_feat = self.apply_backbone(x_stacked)

        # Reshape back to (B, 2*T*R, feat_dim)
        spatiotemp = spat_feat.reshape(B, 2 * T * R, -1)
        spatiotemp = self.former(spatiotemp).flatten(
            start_dim=1, end_dim=-1
        )

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        samples = self.decoder(torch.stack(
            [self.sample(mu, logvar, device=device)
             for _ in range(self.n_samples)],
            dim=1,
        ))

        samples = samples.permute(0, 2, 1)
        mu_out, cov = self.ensemble_stats(samples)

        if return_samples:
            return samples, mu, logvar, mu_out, cov
        return None, mu, logvar, mu_out, cov

    def forward_inference(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ]:
        """Optimized forward for inference with batch_size=1.

        Args:
            x: Input radar tensor.
            return_samples: Whether to return raw samples.

        Returns:
            Tuple of (samples_or_None, mu, logvar, mu_out, cov).
        """
        x = self.pre_process(x)
        device = x.device

        real_part = x.real.squeeze(0)
        imag_part = x.imag.squeeze(0)
        stacked_input = torch.cat([real_part, imag_part], dim=0)

        spat_feat = self.apply_backbone(stacked_input.float())

        spatiotemp = spat_feat.unsqueeze(0)
        spatiotemp = self.former(spatiotemp).flatten(
            start_dim=1, end_dim=-1
        )

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        std = torch.exp(0.5 * logvar)

        mu_exp = mu.expand(self.n_samples, -1)
        std_exp = std.expand(self.n_samples, -1)

        eps = torch.randn_like(mu_exp)
        z = mu_exp + std_exp * eps

        decoded = self.decoder(z)

        samples = decoded.unsqueeze(0).permute(0, 2, 1)
        mu_out, cov = self.ensemble_stats(samples)

        if return_samples:
            return samples, mu, logvar, mu_out, cov
        return None, mu, logvar, mu_out, cov


if __name__ == "__main__":
    model_cov = RadProPoserVAECov().float().cuda()

    test_data = torch.rand(
        4, 8, 2, 64, 4, 4, 256, dtype=torch.complex64
    ).cuda()
    out = model_cov(test_data)
    print(out[1].size())
