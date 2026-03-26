import torch.nn as nn
import torch.nn.functional as F
from common import *
from hrnet3D_config import *
from hrnet3D_config_4D import MODEL_CONFIGS as cfg_4D
from yacs.config import CfgNode as CN
import torch
from edl_pytorch import NormalInvGamma
from transformer import Transformer

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
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
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
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
        branch_index,
        block,
        num_blocks,
        num_channels,
        stride=1,
        bn_type=None,
        bn_momentum=0.1,
    ):
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
                # nn.BatchNorm3d(num_channels[branch_index] * block.expansion),
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
        self, num_branches, block, num_blocks, num_channels, bn_type, bn_momentum=0.1
    ):
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

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
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
                            # nn.BatchNorm3d(num_inchannels[i]),
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
                                    # nn.BatchNorm3d(num_outchannels_conv3x3),
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
                                    # nn.BatchNorm3d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
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
    def __init__(self, cfg, bn_type=None, bn_momentum=None, **kwargs):
        super(HighResolution3DNet, self).__init__()
        if kwargs['full_res_stem']:
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
        self, num_channels_pre_layer, num_channels_cur_layer, bn_type, bn_momentum
    ):
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
                            # nn.BatchNorm3d(num_channels_cur_layer[i]),
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
                            # nn.BatchNorm3d(outchannels),
                            nn.ReLU(inplace=False),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
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

    def forward(self, x):
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
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class RadProPoserEvidential(nn.Module):
    def __init__(self):
        super(RadProPoserEvidential, self).__init__()

        # backbone 
        self.backbone = HighResolution3DNet(
            MODEL_CONFIGS["hr_tiny_feat64_zyx_l4_in64"], full_res_stem=True)
        
        self.bottleNeck = nn.Sequential(
                nn.Conv3d(in_channels=384, out_channels=32, kernel_size=(1, 1, 8), stride=(1, 1, 8)),  # 3x3x3 conv
                nn.BatchNorm3d(32),  # Batch normalization
                nn.ReLU(inplace=True),  # ReLU activation

                nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 8), stride=(1, 1, 8)),  # 3x3x3 conv
                nn.BatchNorm3d(8),  # Batch normalization
                nn.ReLU(inplace=True),  # ReLU activation

                nn.Flatten(start_dim = 1, end_dim = -1)
            )

        # transformer 
        self.former = Transformer()
        
        self.out_uncertainty = nn.Sequential( 
                                 nn.Linear(2048, 256),
        #                         ResidualMLP(256),
                                 nn.Linear(256, 128), 
                                 NormalInvGamma(128, 78))
        
        
    
    def preProcess(self, 
                   x: torch.Tensor):
        x = x - torch.mean(x, dim = -1, keepdim = True)
        x = torch.fft.fft(torch.fft.fft(torch.fft.fft(torch.fft.fft(x ,dim = -1,  norm = "forward"), dim = -2,  norm = "forward"), dim = -3,  norm = "forward"), dim = -4,  norm = "forward")

        x = x.permute(0, 1, 5, 2, 3, 4) 
        return x

    def applyBackbone(self, x: torch.Tensor):
        # get features
        featureList = self.backbone(x)

        # interpolate to same size
        features = []
        for feature in featureList:
            
            helper = F.interpolate(feature.float(), size=(4, 4, 64), mode='trilinear')
            features.append(helper)
        
        features = torch.cat(features, dim = 1)
        features = self.bottleNeck(features)

        return features
    
    def sample_aleatoric_from_output(self,
        model_output,
        n_samples: int = 500,
        eps: float = 1e-6
            ) -> torch.Tensor:
        """
        Draw aleatoric‐only samples, ensuring alpha>1 so that variance=beta/(alpha-1) >= 0.

        Args:
        model_output: tuple/list of (mu, v, alpha, beta), each [B,D]
        n_samples:    number of MC draws
        eps:          small positive to ensure alpha-1 >= eps

        Returns:
        samples: [B, n_samples, D]
        """
        mu, v, alpha, beta = model_output  # each [B, D]

        # clamp alpha so that alpha - 1 >= eps
        alpha_clamped = torch.clamp(alpha, min=1.0 + eps)

        # aleatoric variance = beta / (alpha_clamped - 1)
        alea_var = beta / (alpha_clamped - 1.0)

        # safe standard deviation
        alea_std = torch.sqrt(alea_var)

        B, D = mu.shape
        mu_exp  = mu   .unsqueeze(1).expand(B, n_samples, D)
        std_exp = alea_std.unsqueeze(1).expand(B, n_samples, D)

        eps_noise = torch.randn_like(std_exp)
        return mu_exp + eps_noise * std_exp
        
    def forward(self, 
                x: torch.Tensor):
        # fft layer 
        x = self.preProcess(x) # watch out with batch = 1
        #device = x.device

        # part in real and imag
        xComp = [x.real, x.imag]
        spatFeat = []
        for elmt in xComp:
            for i in range(elmt.size(1)):
                helper = self.applyBackbone(elmt[:,i].float())
                spatFeat.append(helper)
        
        # process with ransformer layer
        spatiotemp = torch.stack(spatFeat, dim = 1)
        spatiotemp = self.former(spatiotemp).flatten(start_dim = 1, end_dim = -1)
        
        # fuse together to get mu and sgm
        #spatiotemp = torch.cat(spatFeat, dim = 1)

        out_uncertainty = self.out_uncertainty(spatiotemp)
        #out = self.out(spatiotemp)
        samples = self.sample_aleatoric_from_output(out_uncertainty)

        return out_uncertainty, samples.permute(0, 2, 1)
    
    def forward_inference(self, 
                      x: torch.Tensor):
        """
        Optimized forward for inference with batch_size=1.
        - Batches all backbone calls into single forward pass
        """
        # FFT layer
        x = self.preProcess(x)
        
        # === Batched backbone: stack all channels into batch dim ===
        real_part = x.real.squeeze(0)  # [C, H, W]
        imag_part = x.imag.squeeze(0)  # [C, H, W]
        stacked_input = torch.cat([real_part, imag_part], dim=0)  # [2*C, H, W]
        
        # Single backbone pass (2*C treated as batch)
        spatFeat = self.applyBackbone(stacked_input.float())  # [2*C, feat_dim]
        
        # Reshape for transformer: [1, 2*C, feat_dim]
        spatiotemp = spatFeat.unsqueeze(0)
        spatiotemp = self.former(spatiotemp).flatten(start_dim=1, end_dim=-1)
        
        # Output uncertainty params + sampling (already optimized)
        out_uncertainty = self.out_uncertainty(spatiotemp)
        samples = self.sample_aleatoric_from_output(out_uncertainty)
        
        return out_uncertainty, samples.permute(0, 2, 1)



if __name__ == "__main__":
    # radproposer
    model = RadProPoserEvidential().float()
    
    testData = torch.rand(1, 8, 4, 4, 64, 128)
    out = model.forward_inference(testData)
    out = model(testData)
    
    print(out[0][0].size())


