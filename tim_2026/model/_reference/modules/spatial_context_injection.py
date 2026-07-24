"""Spatial Context Injection (SCI) via depthwise conv on the projected token grid.

Enriches each token with neighbor context AFTER projection but BEFORE cost
matrix computation.  This fixes the root cause of spurious matches: two
background patches that share similar local texture but differ in context
(one surrounded by PD, one surrounded by noise) will diverge after SCI,
raising their cross-class cost without harming genuine same-class pairs.

Near-identity init: all conv weights start at zero, so the first forward
pass is identical to the baseline.  Gradients from the OT loss gradually
push the weights toward useful spatial filters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialContextLayer(nn.Module):
    """One depthwise conv layer on the token spatial grid with residual + LayerNorm."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd >= 3, got {kernel_size}")
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.norm = nn.LayerNorm(dim)
        with torch.no_grad():
            nn.init.zeros_(self.dw_conv.weight)
            nn.init.zeros_(self.dw_conv.bias)

    def forward(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            tokens: (B, L, D)
            spatial_hw: (H, W) where H * W == L
        Returns:
            (B, L, D) enriched tokens via tokens + dw_conv(LayerNorm(tokens))

        Pre-norm design: LayerNorm → conv → residual add.
        At zero-init the conv output is 0, so output == input (true near-identity).
        """
        B, L, D = tokens.shape
        H, W = spatial_hw
        x = self.norm(tokens)
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, L, D)
        return tokens + x


class SpatialContextInjection(nn.Module):
    """Multi-layer depthwise context injection on the projected token spatial grid.

    Stacks ``num_layers`` SpatialContextLayer blocks, each expanding the
    effective receptive field by one hop.  With kernel_size=3 and 2 layers,
    the effective field covers 5x5 — enough for a standard 5x5 token grid.

    Parameter count: num_layers * (token_dim * kernel_size^2 + token_dim * 2)
    For token_dim=128, kernel_size=3, num_layers=2: ~2,560 params.
    """

    def __init__(
        self,
        token_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.token_dim = int(token_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.layers = nn.ModuleList(
            SpatialContextLayer(token_dim, kernel_size) for _ in range(num_layers)
        )

    def forward(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            tokens: (B, L, D) projected tokens
            spatial_hw: (H, W), must satisfy H * W == L
        Returns:
            (B, L, D) context-enriched tokens, same shape
        """
        H, W = int(spatial_hw[0]), int(spatial_hw[1])
        B, L, D = tokens.shape
        if L != H * W:
            raise ValueError(
                f"SCI: token count L={L} does not match spatial_hw {H}x{W}={H*W}"
            )
        for layer in self.layers:
            tokens = layer(tokens, (H, W))
        return tokens

    def diagnostics(
        self, original: torch.Tensor, enriched: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return scalar diagnostic tensors for logging."""
        ref = original.detach().float()
        out = enriched.detach().float()
        change = (out - ref).norm() / ref.norm().clamp_min(1e-8)
        cosine_shift = 1.0 - (
            (ref * out).sum() / (ref.norm() * out.norm()).clamp_min(1e-8)
        )
        diag: dict[str, torch.Tensor] = {
            "sci/token_change_ratio": change,
            "sci/token_cosine_shift": cosine_shift,
        }
        for i, layer in enumerate(self.layers):
            w = layer.dw_conv.weight.detach().float()
            diag[f"sci/layer{i}_weight_norm"] = w.norm()
            diag[f"sci/layer{i}_weight_max"] = w.abs().max()
        return diag
