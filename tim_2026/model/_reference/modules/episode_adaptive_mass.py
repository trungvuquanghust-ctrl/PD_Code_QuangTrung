"""Episode-adaptive transported-mass prediction for HROT."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.hyperbolic.poincare_ops import PoincareBall, frechet_mean_poincare, hyperbolic_variance


@dataclass
class HyperbolicStats:
    mean_hyp: torch.Tensor
    mean_tangent: torch.Tensor
    variance: torch.Tensor


def summarize_hyperbolic_tokens(tokens: torch.Tensor, ball: PoincareBall) -> HyperbolicStats:
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() != 3:
        raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")

    mean_hyp = frechet_mean_poincare(tokens, ball)
    mean_tangent = ball.logmap0(mean_hyp)
    variance = hyperbolic_variance(tokens, mean_hyp, ball)
    return HyperbolicStats(mean_hyp=mean_hyp, mean_tangent=mean_tangent, variance=variance)


class EpisodeAdaptiveMass(nn.Module):
    """Predict pairwise transported mass from query/class hyperbolic summaries."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        min_mass: float = 0.1,
        default_mass: float = 0.8,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if input_dim is not None and input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 < min_mass <= 1.0:
            raise ValueError("min_mass must be in (0, 1]")
        if not min_mass <= default_mass < 1.0:
            raise ValueError("default_mass must be in [min_mass, 1)")

        self.min_mass = float(min_mass)
        self.input_dim = int(input_dim) if input_dim is not None else 2 * int(embed_dim) + 3
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.LayerNorm(max(1, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
            nn.Sigmoid(),
        )

        final_linear = self.network[-2]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("EpisodeAdaptiveMass expects the penultimate layer to be nn.Linear")
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, math.log(default_mass / (1.0 - default_mass)))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"features last dim must be {self.input_dim}, got {features.shape[-1]}")
        network_dtype = self.network[0].weight.dtype
        rho = self.network(features.to(dtype=network_dtype)).squeeze(-1)
        return rho.clamp(min=self.min_mass, max=1.0)

    def forward_from_stats(
        self,
        query_stats: HyperbolicStats,
        class_stats: HyperbolicStats,
    ) -> torch.Tensor:
        query_mean = query_stats.mean_tangent
        class_mean = class_stats.mean_tangent
        query_var = query_stats.variance
        class_var = class_stats.variance

        cosine = F.cosine_similarity(
            query_mean.unsqueeze(1),
            class_mean.unsqueeze(0),
            dim=-1,
            eps=1e-6,
        )
        features = torch.cat(
            [
                query_mean.unsqueeze(1).expand(-1, class_mean.shape[0], -1),
                class_mean.unsqueeze(0).expand(query_mean.shape[0], -1, -1),
                query_var.unsqueeze(1).unsqueeze(-1).expand(-1, class_mean.shape[0], 1),
                class_var.unsqueeze(0).unsqueeze(-1).expand(query_mean.shape[0], -1, 1),
                cosine.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.forward_features(features)

    def forward(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
        ball: PoincareBall,
    ) -> torch.Tensor:
        query_was_single = query_tokens.dim() == 2
        class_was_single = class_tokens.dim() == 2
        query_stats = summarize_hyperbolic_tokens(query_tokens, ball)
        class_stats = summarize_hyperbolic_tokens(class_tokens, ball)
        rho = self.forward_from_stats(query_stats, class_stats)
        if query_was_single and class_was_single:
            return rho.squeeze(0).squeeze(0)
        return rho
