import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch.distributions import Laplace
from torch.func import vmap
from yacs.config import CfgNode as CN

from models.common import *
from models.hrnet3D_config import MODEL_CONFIGS
from models.hrnet3D_config_4D import MODEL_CONFIGS as cfg_4D
from models.transformer import Transformer


class HighResolutionModule(nn.Module):
    """Multi-branch high-resolution module with cross-resolution fusion."""

    def __init__(
        self,
        num_branches: int,
        blocks,
        num_blocks: list,
        num_inchannels: list,
        num_channels: list,
        fuse_method: str,
        multi_scale_output: bool = True,
        bn_type=None,
        bn_momentum: float = 0.1,
    ) -> None:
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            bn_type=bn_type,
            bn_momentum=bn_momentum,
        )
        self.fuse_layers = self._make_fuse_layers(
            bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(
        self, num_branches: int, blocks, num_blocks: list, num_inchannels: list, num_channels: list
    ) -> None:
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index: int,
        block,
        num_blocks: list,
        num_channels: list,
        stride: int = 1,
        bn_type=None,
        bn_momentum: float = 0.1,
    ) -> nn.Sequential:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.GroupNorm(8, self.num_inchannels[branch_index]),
                nn.Conv3d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches: int, block, num_blocks: list, num_channels: list, bn_type, bn_momentum: float = 0.1
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum: float = 0.1) -> nn.ModuleList:
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_inchannels[j]),
                            nn.Conv3d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list:
        return self.num_inchannels

    def forward(self, x: list) -> list:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="trilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolution3DNet(nn.Module):
    """High-resolution 3D network backbone for radar pose estimation."""

    def __init__(self, cfg, bn_type=None, bn_momentum=None, **kwargs) -> None:
        super(HighResolution3DNet, self).__init__()
        if kwargs.get('full_res_stem'):
            self.full_res_stem = kwargs['full_res_stem']
        self.layer1_cfg = cfg["LAYER1"]
        self.stage2_cfg = cfg["STAGE2"]
        block = blocks_dict[self.layer1_cfg["BLOCK"]]
        self.layer1 = block(self.layer1_cfg['INPLANES'], self.stage2_cfg['INPLANES'], order='gcr')
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [self.stage2_cfg["INPLANES"]], num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage4_cfg = cfg["STAGE4"] if "STAGE4" in cfg else None
        if not self.stage4_cfg is None:
            num_channels = self.stage4_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage4_cfg["BLOCK"]]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))
            ]
            self.transition3 = self._make_transition_layer(
                pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
            )

            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg,
                num_channels,
                multi_scale_output=True,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
            )

    def _make_transition_layer(
        self, num_channels_pre_layer: list, num_channels_cur_layer: list, bn_type, bn_momentum
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_channels_pre_layer[i]),
                            nn.Conv3d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.ReLU(inplace=False),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.GroupNorm(8, inchannels),
                            nn.Conv3d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.ReLU(inplace=False),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels: list,
        multi_scale_output: bool = True,
        bn_type=None,
        bn_momentum: float = 0.1,
    ) -> Tuple[nn.Sequential, list]:
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                    bn_momentum,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> list:
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        if not self.stage4_cfg is None:
            x_list = []
            for i in range(self.stage4_cfg["NUM_BRANCHES"]):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        return y_list


class ResidualMLP(nn.Module):
    """Residual MLP block with GELU activation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RadProPoserVAECov(nn.Module):
    """RadProPoser VAE model with covariance estimation for pose prediction."""

    def __init__(self, n_samples: int = 500) -> None:
        super(RadProPoserVAECov, self).__init__()

        # backbone
        self.backbone = HighResolution3DNet(
            MODEL_CONFIGS["hr_tiny_feat64_zyx_l4_in64"], full_res_stem=True)

        self.bottleNeck = nn.Sequential(
                nn.Conv3d(in_channels=384, out_channels=32, kernel_size=(1, 1, 8), stride=(1, 1, 8)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 8), stride=(1, 1, 8)),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),

                nn.Flatten(start_dim=1, end_dim=-1)
            )

        # transformer - max_seq_len supports up to 64 (2*16*2), dynamically sliced
        self.former = Transformer(seq_len=64)

        # latent space - input dim = 2 * T * R * 128 = 2 * 8 * 2 * 128 = 4096
        latent_input_dim = 4096
        self.mu = nn.Sequential(nn.Linear(latent_input_dim, 512))
        self.spread = nn.Sequential(nn.Linear(latent_input_dim, 512))
        self.varianceScaling = nn.Parameter(torch.tensor(0.1))
        self.Nsamples = n_samples
        self.latent_dist = "gaussian"

        # decoder
        self.decoder = nn.Sequential(
                                     nn.Linear(512, 128),
                                     nn.Linear(128, 17*3))

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        """Sample from Gaussian latent distribution using reparameterization trick.

        Args:
            mu: Mean tensor of shape (B, latent_dim)
            logvar: Log-variance tensor of shape (B, latent_dim)
            device: Device to create noise tensor on

        Returns:
            Sampled latent tensor of shape (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device) * self.varianceScaling
        return mu + eps * std

    def preProcess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply clutter removal and 4D FFT preprocessing.

        Args:
            x: Input radar tensor

        Returns:
            Preprocessed complex tensor with permuted dimensions
        """
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = torch.fft.fft(torch.fft.fft(torch.fft.fft(torch.fft.fft(
            x, dim=-1, norm="forward"), dim=-2, norm="forward"), dim=-3, norm="forward"), dim=-4, norm="forward")
        x = x.permute(0, 1, 5, 2, 3, 4)
        return x

    def applyBackbone(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backbone network and bottleneck to extract features.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Flattened feature tensor
        """
        featureList = self.backbone(x)

        features = []
        for feature in featureList:
            helper = F.interpolate(feature.float(), size=(4, 4, 64), mode='trilinear')
            features.append(helper)

        features = torch.cat(features, dim=1)
        features = self.bottleNeck(features)

        return features

    def sampleLatent(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        alpha: float = None
    ) -> torch.Tensor:
        """Sample from latent space with optional variance scaling.

        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            alpha: Optional variance scaling factor

        Returns:
            Sampled latent tensor
        """
        if alpha is not None:
            self.varianceScaling = torch.tensor(alpha)
        out = self.sample(mu, logvar)
        return out

    def ensemble_stats(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and covariance from ensemble predictions.

        Args:
            preds: Tensor of shape (B, K, E) - batch x features x ensemble samples

        Returns:
            Tuple of (mu, cov) where mu is (B, K) and cov is (B, K, K)
        """
        mu = preds.mean(dim=2)
        cov = vmap(lambda x: torch.cov(x, correction=1))(preds)
        return mu, cov

    def ensemble_stats_var(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance from ensemble predictions.

        Args:
            preds: Tensor of shape (B, K, E) - batch x features x ensemble samples

        Returns:
            Tuple of (mu, var) where both are (B, K)
        """
        mu = preds.mean(dim=2)
        var = preds.var(dim=2, unbiased=True)
        return mu, var

    def getLatent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent distribution parameters from preprocessed input.

        Args:
            x: Preprocessed complex radar tensor

        Returns:
            Tuple of (mu, logvar) tensors
        """
        device = x.device

        xComp = [x.real, x.imag]
        spatFeat = []
        for elmt in xComp:
            for i in range(elmt.size(1)):
                helper = self.applyBackbone(elmt[:, i].float())
                spatFeat.append(helper)

        spatiotemp = torch.cat(spatFeat, dim=1)

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        return mu, logvar

    def forward(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with ensemble sampling.

        Args:
            x: Input radar tensor (B, T, R, D, TX, RX, Range)
            return_samples: Whether to return raw samples

        Returns:
            Tuple of (samples, mu, logvar, muOut, cov)
        """
        device = x.device
        B, T, R = x.shape[0], x.shape[1], x.shape[2]

        # Stack real/imag and reshape for batched backbone
        x_stacked = torch.stack([x.real, x.imag], dim=1)  # (B, 2, T, R, D, TX, RX, Range)
        x_stacked = x_stacked.reshape(B * 2 * T * R, *x_stacked.shape[4:]).float()

        # Single batched backbone call instead of 2*T*R sequential calls
        spatFeat = self.applyBackbone(x_stacked)  # (B*2*T*R, feat_dim)

        # Reshape back to (B, 2*T*R, feat_dim)
        spatiotemp = spatFeat.reshape(B, 2 * T * R, -1)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        samples = self.decoder(torch.stack(
            [self.sample(mu, logvar, device=device) for i in range(self.Nsamples)], dim=1))

        samples = samples.permute(0, 2, 1)

        muOut, cov = self.ensemble_stats(samples)

        if return_samples:
            return samples, mu, logvar, muOut, cov
        else:
            return None, mu, logvar, muOut, cov

    def forward_inference(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized forward for inference with batch_size=1.

        Args:
            x: Input radar tensor
            return_samples: Whether to return raw samples

        Returns:
            Tuple of (samples, mu, logvar, muOut, cov)
        """
        x = self.preProcess(x)
        device = x.device

        real_part = x.real.squeeze(0)
        imag_part = x.imag.squeeze(0)
        stacked_input = torch.cat([real_part, imag_part], dim=0)

        spatFeat = self.applyBackbone(stacked_input.float())

        spatiotemp = spatFeat.unsqueeze(0)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)

        mu = self.mu(spatiotemp)
        logvar = self.spread(spatiotemp)

        std = torch.exp(0.5 * logvar)

        mu_exp = mu.expand(self.Nsamples, -1)
        std_exp = std.expand(self.Nsamples, -1)

        eps = torch.randn_like(mu_exp)
        z = mu_exp + std_exp * eps

        decoded = self.decoder(z)

        samples = decoded.unsqueeze(0).permute(0, 2, 1)

        muOut, cov = self.ensemble_stats(samples)

        if return_samples:
            return samples, mu, logvar, muOut, cov
        else:
            return None, mu, logvar, muOut, cov


if __name__ == "__main__":
    model_cov = RadProPoserVAECov().float().cuda()

    testData = torch.rand(4, 8, 2, 64, 4, 4, 256, dtype=torch.complex64).cuda()
    out = model_cov(testData)
    print(out[1].size())
