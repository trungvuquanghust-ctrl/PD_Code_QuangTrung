"""Multi-scale spatial pyramid token aggregation for Ours-Final.

MSPTA converts a backbone feature map into a concatenated set of spatial
tokens at multiple pooling scales.  It also returns per-token probability
weights that can be used as non-uniform UOT marginals after multiplying by the
episode transport budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.fewshot_common import feature_map_to_tokens


MSPTA_MASS_MODES = frozenset({"area", "balanced_area", "compact_area", "uniform"})


@dataclass(frozen=True)
class MSPTAScaleTokens:
    name: str
    scale: int
    spatial_hw: tuple[int, int]
    tokens: torch.Tensor
    weights: torch.Tensor


def parse_mspta_scales(scales: str | Sequence[int | str]) -> tuple[int, ...]:
    """Parse MSPTA scale factors.

    A scale of 1 keeps the original feature-map grid.  A scale of k > 1 pools
    to ceil(H / k) by ceil(W / k).
    """

    if isinstance(scales, str):
        raw_items = [item.strip() for item in scales.replace("x", ",").split(",") if item.strip()]
    else:
        raw_items = []
        for item in scales:
            if isinstance(item, str):
                raw_items.extend(part.strip() for part in item.replace("x", ",").split(",") if part.strip())
            else:
                raw_items.append(item)
    if not raw_items:
        raise ValueError("mspta_scales must contain at least one scale")

    parsed = tuple(int(item) for item in raw_items)
    if any(scale <= 0 for scale in parsed):
        raise ValueError(f"mspta_scales must be positive, got {parsed}")
    if 1 not in parsed:
        raise ValueError("mspta_scales must include 1 so the fine token grid is preserved")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"mspta_scales contains duplicate values: {parsed}")
    return tuple(sorted(parsed))


def normalize_mspta_mass_mode(value: str | None) -> str:
    mode = "area" if value is None else str(value).strip().lower().replace("-", "_")
    if mode not in MSPTA_MASS_MODES:
        raise ValueError(f"Unsupported mspta_mass_mode: {value}. Expected one of {sorted(MSPTA_MASS_MODES)}")
    return mode


class MSPTATokenizer(nn.Module):
    """Build a multi-scale token pyramid and marginal weights.

    The returned weights sum to 1 for each image.  Ours-Final multiplies them
    by each ECOT rho budget inside the existing UOT code path.

    mass_mode='area' preserves the standalone MSPTA script exactly: each token
    gets nominal k*k area.  mass_mode='balanced_area' gives every scale the
    same total budget while still making coarser tokens heavier within that
    scale.  mass_mode='compact_area' keeps the same scale balance but shifts
    mass inside each scale away from full-height vertical-band saliency.
    """

    def __init__(
        self,
        scales: str | Sequence[int | str] = (1, 2, 3),
        mass_mode: str = "area",
        compact_floor: float = 0.05,
        vertical_suppression: float = 1.0,
    ) -> None:
        super().__init__()
        self.scales = parse_mspta_scales(scales)
        self.mass_mode = normalize_mspta_mass_mode(mass_mode)
        if not 0.0 <= float(compact_floor) <= 1.0:
            raise ValueError("compact_floor must be in [0, 1]")
        if float(vertical_suppression) < 0.0:
            raise ValueError("vertical_suppression must be non-negative")
        self.compact_floor = float(compact_floor)
        self.vertical_suppression = float(vertical_suppression)
        self.scale_names = tuple("fine" if scale == 1 else f"s{scale}" for scale in self.scales)

    def _compact_guidance(self, feature_map: torch.Tensor, guidance_map: torch.Tensor | None) -> torch.Tensor:
        if guidance_map is None:
            guidance = feature_map.detach().float().norm(dim=1, keepdim=True)
        else:
            guidance = guidance_map.detach().float()
            if guidance.dim() != 4:
                raise ValueError(f"guidance_map must have shape (B, 1, H, W), got {tuple(guidance.shape)}")
            if guidance.shape[1] != 1:
                guidance = guidance.mean(dim=1, keepdim=True)
            if guidance.shape[-2:] != feature_map.shape[-2:]:
                guidance = F.interpolate(
                    guidance,
                    size=feature_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        g_min = guidance.amin(dim=(-2, -1), keepdim=True)
        guidance = guidance - g_min
        g_max = guidance.amax(dim=(-2, -1), keepdim=True)
        eps = torch.finfo(guidance.dtype).eps
        guidance = guidance / g_max.clamp_min(eps)
        column_profile = guidance.mean(dim=2, keepdim=True)
        compact = (guidance - float(self.vertical_suppression) * column_profile).clamp_min(0.0)
        return compact.to(device=feature_map.device, dtype=feature_map.dtype)

    def forward(
        self,
        feature_map: torch.Tensor,
        guidance_map: torch.Tensor | None = None,
    ) -> list[MSPTAScaleTokens]:
        if feature_map.dim() != 4:
            raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
        batch_size, _, height, width = feature_map.shape
        compact_guidance = (
            self._compact_guidance(feature_map, guidance_map)
            if self.mass_mode == "compact_area"
            else None
        )
        records: list[MSPTAScaleTokens] = []
        for name, scale in zip(self.scale_names, self.scales):
            if scale == 1:
                pooled = feature_map
                spatial_hw = (int(height), int(width))
            else:
                spatial_hw = (math.ceil(int(height) / scale), math.ceil(int(width) / scale))
                pooled = F.adaptive_avg_pool2d(feature_map, output_size=spatial_hw)

            tokens = feature_map_to_tokens(pooled)
            token_count = int(tokens.shape[-2])
            if self.mass_mode == "uniform":
                raw_weight = feature_map.new_ones((batch_size, token_count))
            elif self.mass_mode == "balanced_area":
                raw_weight = feature_map.new_full(
                    (batch_size, token_count),
                    float(height * width) / float(token_count),
                )
            elif self.mass_mode == "compact_area":
                pooled_guidance = F.adaptive_avg_pool2d(compact_guidance, output_size=spatial_hw)
                token_guidance = feature_map_to_tokens(pooled_guidance).squeeze(-1).clamp_min(0.0)
                uniform = feature_map.new_full((batch_size, token_count), 1.0 / float(token_count))
                guidance_sum = token_guidance.sum(dim=-1, keepdim=True)
                token_prob = token_guidance / guidance_sum.clamp_min(torch.finfo(feature_map.dtype).eps)
                token_prob = torch.where(guidance_sum > 0.0, token_prob, uniform)
                floor = feature_map.new_tensor(float(self.compact_floor))
                raw_weight = (1.0 - floor) * token_prob + floor * uniform
            else:
                raw_weight = feature_map.new_full((batch_size, token_count), float(scale * scale))
            records.append(
                MSPTAScaleTokens(
                    name=name,
                    scale=scale,
                    spatial_hw=spatial_hw,
                    tokens=tokens,
                    weights=raw_weight,
                )
            )

        total_weight = torch.cat([record.weights for record in records], dim=-1).sum(
            dim=-1,
            keepdim=True,
        )
        normalized_records = []
        for record in records:
            normalized_records.append(
                MSPTAScaleTokens(
                    name=record.name,
                    scale=record.scale,
                    spatial_hw=record.spatial_hw,
                    tokens=record.tokens,
                    weights=record.weights / total_weight.clamp_min(torch.finfo(feature_map.dtype).eps),
                )
            )
        return normalized_records


__all__ = [
    "MSPTA_MASS_MODES",
    "MSPTAScaleTokens",
    "MSPTATokenizer",
    "normalize_mspta_mass_mode",
    "parse_mspta_scales",
]
