"""Structural Token Augmentation — inject spatial position and local contrast
into each token AFTER projection, before OT matching.

Unlike spatial context enrichment (depthwise conv on feature map), this
operates on the projected token sequence and appends lightweight structural
descriptors that encode WHERE a token sits and HOW prominent it is relative
to neighbors.  These features are class-agnostic (position + contrast),
so they generalize to unseen classes without meta-overfitting.

When disabled, output === input (zero-regression).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class StructuralTokenAugmentation(nn.Module):
    """Append structural descriptors to projected tokens.

    Descriptors per token (all computed from the token set, no learned params
    except optional projection):

    - **Normalized position** (y, x) in [0, 1]: encodes spatial location.
    - **Local contrast**: token energy vs mean neighbor energy.  High = peak,
      low = background valley.
    - **Rank percentile**: where this token's energy falls among all L tokens.
      1.0 = brightest, 0.0 = dimmest.

    These 4 scalars are projected via a small linear layer to ``struct_dim``
    and concatenated to the semantic token, producing tokens of dimension
    ``token_dim + struct_dim``.  A final linear layer projects back to
    ``token_dim`` so the rest of the pipeline (L2-norm, cost matrix, UOT)
    is unchanged.

    The back-projection is initialized near zero so early training ≈ baseline.
    """

    def __init__(
        self,
        token_dim: int,
        struct_dim: int = 16,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.struct_dim = int(struct_dim)
        self.num_raw_features = 4

        self.struct_proj = nn.Sequential(
            nn.Linear(self.num_raw_features, self.struct_dim, bias=True),
            nn.GELU(),
        )
        self.fuse = nn.Linear(self.token_dim + self.struct_dim, self.token_dim, bias=False)
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.fuse.weight[:, : self.token_dim])
            nn.init.zeros_(self.fuse.weight[:, self.token_dim :])

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            tokens: ``(B, L, D)`` projected tokens.
            spatial_hw: ``(H, W)`` of the feature map grid so we can
                compute 2-D positions and neighbor contrast.

        Returns:
            Augmented tokens ``(B, L, D)`` — same shape, enriched content.
        """
        B, L, D = tokens.shape
        H, W = spatial_hw

        struct_features = self._compute_structural_features(tokens, H, W)
        struct_emb = self.struct_proj(struct_features)
        fused = torch.cat([tokens, struct_emb], dim=-1)
        return self.fuse(fused)

    def _compute_structural_features(
        self,
        tokens: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B, L, D = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        energy = tokens.detach().norm(dim=-1)

        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        if H > 1:
            ys = ys / (H - 1)
        if W > 1:
            xs = xs / (W - 1)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos_y = grid_y.reshape(1, L).expand(B, -1)
        pos_x = grid_x.reshape(1, L).expand(B, -1)

        energy_map = energy.reshape(B, 1, H, W)
        pad_map = torch.nn.functional.pad(energy_map, (1, 1, 1, 1), mode="replicate")
        neighbor_sum = torch.zeros_like(energy_map)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_sum = neighbor_sum + pad_map[
                    :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W
                ]
        neighbor_mean = neighbor_sum / 8.0
        contrast = (energy_map - neighbor_mean).squeeze(1).reshape(B, L)
        c_abs_max = contrast.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        contrast = contrast / c_abs_max

        sorted_indices = energy.argsort(dim=-1)
        rank = torch.zeros_like(energy)
        rank.scatter_(
            1,
            sorted_indices,
            torch.linspace(0, 1, L, device=device, dtype=dtype).unsqueeze(0).expand(B, -1),
        )

        return torch.stack([pos_y, pos_x, contrast, rank], dim=-1)

    def diagnostics(self, tokens_before: torch.Tensor, tokens_after: torch.Tensor) -> dict[str, torch.Tensor]:
        change = (
            (tokens_after.detach().float() - tokens_before.detach().float()).norm()
            / tokens_before.detach().float().norm().clamp_min(1e-8)
        )
        w = self.fuse.weight.detach().float()
        struct_weight_norm = w[:, self.token_dim :].norm()
        semantic_weight_norm = w[:, : self.token_dim].norm()
        return {
            "struct/token_change_ratio": change,
            "struct/struct_weight_norm": struct_weight_norm,
            "struct/semantic_weight_norm": semantic_weight_norm,
            "struct/struct_vs_semantic_ratio": struct_weight_norm / semantic_weight_norm.clamp_min(1e-8),
        }
