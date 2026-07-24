"""Multi-scale tokenization utilities for OT-style few-shot heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.fewshot_common import feature_map_to_tokens


@dataclass(frozen=True)
class MultiScaleTokenSpec:
    name: str
    output_size: tuple[int, int] | str | None


def parse_multiscale_pool_sizes(pool_sizes: str | Sequence[str | tuple[int, int]]) -> tuple[MultiScaleTokenSpec, ...]:
    if isinstance(pool_sizes, str):
        raw_items = [item.strip() for item in pool_sizes.split(",") if item.strip()]
    else:
        raw_items = list(pool_sizes)
    if not raw_items:
        raise ValueError("multiscale_pool_sizes must contain at least one scale")

    output_sizes: list[tuple[int, int] | str | None] = []
    for item in raw_items:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"Invalid multiscale pool tuple: {item}")
            height, width = int(item[0]), int(item[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"multiscale pool sizes must be positive, got {item}")
            output_sizes.append((height, width))
            continue

        text = str(item).strip().lower().replace(" ", "")
        if text in {"original", "orig", "identity", "native"}:
            output_sizes.append(None)
            continue
        if text in {"half", "h//2", "hxw//2"}:
            output_sizes.append("half")
            continue
        if "x" in text:
            parts = text.split("x")
            if len(parts) != 2:
                raise ValueError(f"Invalid multiscale pool size: {item}")
            height, width = int(parts[0]), int(parts[1])
        else:
            height = width = int(text)
        if height <= 0 or width <= 0:
            raise ValueError(f"multiscale pool sizes must be positive, got {item}")
        output_sizes.append((height, width))

    if output_sizes[0] is None:
        base_labels = ["fine", "medium", "coarse", "global"]
    elif len(output_sizes) == 2 and output_sizes[-1] == (1, 1):
        base_labels = ["medium", "coarse"]
    else:
        base_labels = ["fine", "medium", "coarse", "global"]
    specs = [
        MultiScaleTokenSpec(
            base_labels[idx] if idx < len(base_labels) else f"scale{idx + 1}",
            output_size,
        )
        for idx, output_size in enumerate(output_sizes)
    ]
    return tuple(specs)


class MultiScaleTokenizer(nn.Module):
    """Create token sets at multiple adaptive-pooling scales from one feature map."""

    def __init__(self, pool_sizes: str | Sequence[str | tuple[int, int]]) -> None:
        super().__init__()
        self.specs = parse_multiscale_pool_sizes(pool_sizes)
        self.scale_names = tuple(spec.name for spec in self.specs)

    def forward(self, feature_map: torch.Tensor) -> list[tuple[str, tuple[int, int], torch.Tensor]]:
        if feature_map.dim() != 4:
            raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
        height, width = int(feature_map.shape[-2]), int(feature_map.shape[-1])
        token_sets: list[tuple[str, tuple[int, int], torch.Tensor]] = []
        for spec in self.specs:
            if spec.output_size is None:
                pooled = feature_map
                spatial_hw = (height, width)
            elif spec.output_size == "half":
                spatial_hw = (max(1, height // 2), max(1, width // 2))
                pooled = F.adaptive_avg_pool2d(feature_map, output_size=spatial_hw)
            else:
                spatial_hw = spec.output_size
                pooled = F.adaptive_avg_pool2d(feature_map, output_size=spatial_hw)
            token_sets.append((spec.name, spatial_hw, feature_map_to_tokens(pooled)))
        return token_sets
