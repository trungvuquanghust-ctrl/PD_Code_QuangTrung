"""Spatial Mamba-style token processors for intra-image sequences."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from tim_2026.model._reference.ssm.common import SelectiveStateSpaceCell, run_selective_scan


def build_2d_position_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build normalized 2D coordinates for a row-major token sequence."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square() + 1e-6)
    position = torch.stack(
        [yy, xx, yy * xx, yy.square(), xx.square(), radius],
        dim=-1,
    )
    return position.reshape(1, height * width, -1)


class BidirectionalSpatialMambaBlock(nn.Module):
    """Bidirectional selective scan over spatial tokens with explicit 2D position."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = max(dim, dim * ffn_multiplier)
        self.input_norm = nn.LayerNorm(dim)
        self.position_proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.context_proj = nn.Linear(dim, dim, bias=False)
        self.forward_cell = SelectiveStateSpaceCell(dim, state_dim)
        self.backward_cell = SelectiveStateSpaceCell(dim, state_dim)
        self.mix_proj = nn.Linear(dim * 3, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process `(Batch, H*W, Dim)` tokens and return the same shape."""
        if tokens.dim() != 3:
            raise ValueError(
                f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}"
            )

        height, width = spatial_hw
        if tokens.shape[1] != height * width:
            raise ValueError(
                "Token count does not match spatial_hw: "
                f"tokens={tokens.shape[1]} spatial_hw={spatial_hw}"
            )

        batch_size = tokens.shape[0]
        conditioning = self.position_proj(
            build_2d_position_grid(height, width, tokens.device, tokens.dtype)
        ).expand(batch_size, -1, -1)

        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1)
            if context.shape != (batch_size, tokens.shape[-1]):
                raise ValueError(
                    "context must have shape (Batch, Dim) or (Dim,), "
                    f"got {tuple(context.shape)}"
                )
            conditioning = conditioning + self.context_proj(context).unsqueeze(1)

        normalized_tokens = self.input_norm(tokens)
        scan_inputs = normalized_tokens + conditioning

        forward_outputs, _ = run_selective_scan(
            self.forward_cell,
            scan_inputs,
            conditioning=conditioning,
        )

        backward_outputs, _ = run_selective_scan(
            self.backward_cell,
            torch.flip(scan_inputs, dims=[1]),
            conditioning=torch.flip(conditioning, dims=[1]),
        )
        backward_outputs = torch.flip(backward_outputs, dims=[1])

        mixed = self.mix_proj(
            torch.cat([forward_outputs, backward_outputs, conditioning], dim=-1)
        )
        tokens = self.output_norm(tokens + mixed)
        tokens = tokens + self.feed_forward(self.feed_forward_norm(tokens))
        return tokens


class IntraImageMambaEncoder(nn.Module):
    """Stack bidirectional spatial scan blocks over image tokens."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        depth: int = 1,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.blocks = nn.ModuleList(
            [
                BidirectionalSpatialMambaBlock(
                    dim=dim,
                    state_dim=state_dim,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process `(Batch, H*W, Dim)` tokens and preserve the shape."""
        for block in self.blocks:
            tokens = block(tokens, spatial_hw=spatial_hw, context=context)
        return tokens
