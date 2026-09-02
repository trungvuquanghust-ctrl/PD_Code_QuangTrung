"""SMNet-exact Conv64F encoder plus a Vision-Mamba backbone, both exposing a
ResNet12-compatible interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.encoders.fsl_mamba_encoder import FSLMambaEncoder
from tim_2026.model._reference.encoders.resnet12_encoder import ResNet12Encoder

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - optional, requires CUDA build
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


class DropPath(nn.Module):
    """Per-sample stochastic depth (standard implementation)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * (mask / keep_prob)


class VimBlock(nn.Module):
    """Pre-norm residual Mamba block: x = x + drop_path(mixer(norm(x))).

    `mamba_ssm.Mamba` is a bare mixer with no residual or normalization of
    its own. Stacking many of these directly (as raw `blk(x)` calls) removes
    the residual path entirely and is prone to unstable training once you go
    past a few layers. Every reference Mamba/Vim usage wraps the mixer this
    way, so we do too.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "VimBackbone requires mamba-ssm (CUDA build). Install with:\n"
                "  pip install causal-conv1d>=1.1.0 mamba-ssm"
            ) from MAMBA_IMPORT_ERROR
        self.norm = nn.LayerNorm(dim)
        self.mixer = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


class VimBackbone(nn.Module):
    """Vision-Mamba-style backbone for PECT: patch embed + stacked pre-norm
    residual Mamba blocks + learned positional embedding.

    Exposes the same `out_channels` / `out_dim` / `feat_dim` /
    `forward_features` / `forward` interface as `ResNet12Encoder` and
    `SMNetConv64FEncoder` so it plugs into `BaseConv64FewShotModel` without
    any other code changes.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path_rate: float = 0.1,
        pool_output: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                "VimBackbone requires image_size to be divisible by patch_size, "
                f"got image_size={image_size}, patch_size={patch_size}"
            )

        self.pool_output = bool(pool_output)
        self.patch_size = int(patch_size)
        self.grid_size = int(image_size) // int(patch_size)

        # ResNet12/SMNet-compatible interface attributes.
        self.out_channels = int(embed_dim)
        self.out_dim = int(embed_dim)
        self.out_spatial = self.grid_size
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        drop_path_schedule = torch.linspace(0.0, float(drop_path_rate), max(depth, 1)).tolist()
        self.blocks = nn.ModuleList(
            [
                VimBlock(
                    embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path=drop_path_schedule[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = self.patch_embed(x)  # (B, embed_dim, grid, grid)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, grid*grid, embed_dim)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        feature_map = tokens.transpose(1, 2).reshape(
            batch_size, self.out_channels, self.grid_size, self.grid_size
        )
        return feature_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_features(x)
        if not self.pool_output:
            return feature_map
        return F.adaptive_avg_pool2d(feature_map, 1).view(feature_map.size(0), -1)


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
    vision_mamba_patch_size: int = 16,
    vision_mamba_embed_dim: int = 192,
    vision_mamba_depth: int = 12,
    vision_mamba_d_state: int = 16,
    vision_mamba_drop_path: float = 0.1,
) -> nn.Module:
    """Build the legacy ResNet12 encoder, or one of the additive options
    (Conv64F / FSL-Mamba / Vision-Mamba)."""

    backbone_name = str(backbone_name).lower()

    if backbone_name == "vision_mamba":
        return VimBackbone(
            image_size=image_size,
            patch_size=vision_mamba_patch_size,
            embed_dim=vision_mamba_embed_dim,
            depth=vision_mamba_depth,
            d_state=vision_mamba_d_state,
            drop_path_rate=vision_mamba_drop_path,
            pool_output=pool_output,
        )
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
