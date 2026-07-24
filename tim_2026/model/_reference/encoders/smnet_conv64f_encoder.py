"""SMNet-exact Conv64F encoder with a ResNet12-compatible interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.encoders.fsl_mamba_encoder import FSLMambaEncoder
from tim_2026.model._reference.encoders.resnet12_encoder import ResNet12Encoder


class Conv64FBlock(nn.Module):
    """Single Conv64F block matching smnet's current backbone."""

    def __init__(self, in_channels: int, out_channels: int, use_pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SMNetConv64FEncoder(nn.Module):
    """Conv64F encoder that exposes the same pooled/map API as ResNet12Encoder."""

    def __init__(
        self,
        image_size: int = 64,
        pool_output: bool = False,
        pool_last: bool = True,
    ) -> None:
        super().__init__()
        self.pool_output = bool(pool_output)
        self.pool_last = bool(pool_last)
        self.out_channels = 64
        self.out_dim = 64
        self.blocks = nn.Sequential(
            Conv64FBlock(3, 64, use_pool=True),
            Conv64FBlock(64, 64, use_pool=True),
            Conv64FBlock(64, 64, use_pool=True),
            Conv64FBlock(64, 64, use_pool=pool_last),
        )

        spatial = int(image_size)
        for _ in range(3):
            spatial = max(1, spatial // 2)
        if pool_last:
            spatial = max(1, spatial // 2)
        self.out_spatial = spatial
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if not self.pool_output:
            return features
        return F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)


def build_resnet12_family_encoder(
    image_size: int = 64,
    backbone_name: str = "resnet12",
    pool_output: bool = False,
    variant: str = "fewshot",
    drop_rate: float = 0.0,
    dropblock_size: int = 5,
    fsl_mamba_base_dim: int = 48,
    fsl_mamba_output_dim: int = 320,
    fsl_mamba_drop_path: float = 0.02,
    fsl_mamba_perturb_sigma: float = 0.0,
) -> nn.Module:
    """Build the legacy ResNet12 encoder or the additive smnet Conv64F option."""

    backbone_name = str(backbone_name).lower()
    if backbone_name == "resnet12":
        return ResNet12Encoder(
            image_size=image_size,
            pool_output=pool_output,
            variant=variant,
            drop_rate=drop_rate,
            dropblock_size=dropblock_size,
        )
    if backbone_name == "conv64f":
        pool_last = variant != "deepbdc"
        return SMNetConv64FEncoder(
            image_size=image_size,
            pool_output=pool_output,
            pool_last=pool_last,
        )
    if backbone_name in {"fsl_mamba", "slim_mamba"}:
        return FSLMambaEncoder(
            image_size=image_size,
            pool_output=pool_output,
            base_dim=fsl_mamba_base_dim,
            output_dim=fsl_mamba_output_dim,
            drop_path=fsl_mamba_drop_path,
            perturb_sigma=fsl_mamba_perturb_sigma,
        )
    raise ValueError(f"Unsupported fewshot_backbone: {backbone_name}")
