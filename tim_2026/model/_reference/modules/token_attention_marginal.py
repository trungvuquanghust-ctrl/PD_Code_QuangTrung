"""Learned token-attention marginals for fixed-budget optimal transport."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("temperature must be positive")
    return math.log(math.expm1(value))


class TokenAttentionMarginal(nn.Module):
    """Assign a learned, budget-preserving OT mass to each token."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        uniform_floor: float = 0.1,
        detach_weights: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= float(uniform_floor) <= 1.0:
            raise ValueError("uniform_floor must be in [0, 1]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.scorer = nn.Sequential(
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.raw_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(float(temperature)), dtype=torch.float32)
        )
        self.uniform_floor = float(uniform_floor)
        self.detach_weights = bool(detach_weights)
        self.eps = float(eps)
        self.register_buffer(
            "uniform_floor_value",
            torch.tensor(self.uniform_floor, dtype=torch.float32),
        )
        self.register_buffer(
            "detach_weights_value",
            torch.tensor(float(self.detach_weights), dtype=torch.float32),
        )

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature).clamp_min(1e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        rho: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.dim() < 2:
            raise ValueError(
                "tokens must have shape (..., Tokens, Dim), "
                f"got {tuple(tokens.shape)}"
            )
        token_count = int(tokens.shape[-2])
        if token_count <= 0:
            raise ValueError("tokens must contain at least one token")

        logits = self.scorer(tokens).squeeze(-1)
        tau = self.temperature.to(device=logits.device, dtype=logits.dtype)
        attention = torch.softmax(logits / tau, dim=-1)
        uniform = torch.full_like(attention, 1.0 / float(token_count))
        weights = (
            (1.0 - self.uniform_floor) * attention
            + self.uniform_floor * uniform
        )
        if self.detach_weights:
            weights = weights.detach()

        rho_tensor = torch.as_tensor(
            rho,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        try:
            rho_tensor = torch.broadcast_to(rho_tensor, tokens.shape[:-2])
        except RuntimeError as exc:
            raise ValueError(
                f"rho shape {tuple(rho_tensor.shape)} is not broadcastable to "
                f"token leading shape {tuple(tokens.shape[:-2])}"
            ) from exc
        marginals = weights * rho_tensor.unsqueeze(-1)

        entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=-1)
        entropy_denom = math.log(float(max(token_count, 2)))
        scorer_norm = torch.stack(
            [parameter.norm() for parameter in self.scorer.parameters()]
        ).norm()
        diagnostics = {
            "token_marginal/entropy": entropy.mean().detach(),
            "token_marginal/normalized_entropy": (
                entropy / entropy_denom
            ).mean().detach(),
            "token_marginal/max_weight": weights.amax(dim=-1).mean().detach(),
            "token_marginal/min_weight": weights.amin(dim=-1).mean().detach(),
            "token_marginal/l1_from_uniform": (
                weights - uniform
            ).abs().sum(dim=-1).mean().detach(),
            "token_marginal/logit_std": logits.std(
                dim=-1,
                unbiased=False,
            ).mean().detach(),
            "token_marginal/temperature": tau.detach(),
            "token_marginal/uniform_floor": tokens.new_tensor(
                self.uniform_floor
            ),
            "token_marginal/mass_error": (
                marginals.sum(dim=-1) - rho_tensor
            ).abs().max().detach(),
            "token_marginal/scorer_norm": scorer_norm.detach(),
        }
        return marginals, diagnostics


__all__ = ["TokenAttentionMarginal"]
