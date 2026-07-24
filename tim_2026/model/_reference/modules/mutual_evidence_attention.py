"""Mutual-evidence attention marginals for fixed-budget ECOT.

MEA builds query-conditioned query/support token marginals from the same
token-token cost table that ECOT will later transport.  It is deliberately
small: a learnable mixture strength and a learnable attention temperature.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sigmoid(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("inverse sigmoid expects a value in (0, 1)")
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


class MutualEvidenceAttentionMarginal(nn.Module):
    """Build budget-preserving ECOT marginals from bidirectional attention.

    Given a query/support token cost matrix C for each query-class-shot pair,
    MEA forms a normalized attention table A = softmax(-standardize(C) / tau).
    Averaging row-normalized attention gives the support prior, while averaging
    column-normalized attention gives the query prior.  Both priors are mixed
    with uniform mass before the fixed ECOT budget rho is applied.
    """

    def __init__(
        self,
        *,
        eta_init: float = 0.35,
        temperature_init: float = 0.70,
        entropy_reg: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < float(eta_init) < 1.0:
            raise ValueError("eta_init must be in (0, 1)")
        if float(temperature_init) <= 0.0:
            raise ValueError("temperature_init must be positive")
        if float(entropy_reg) < 0.0:
            raise ValueError("entropy_reg must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.raw_eta = nn.Parameter(torch.tensor(_inverse_sigmoid(float(eta_init)), dtype=torch.float32))
        self.raw_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(float(temperature_init)), dtype=torch.float32)
        )
        self.entropy_reg = float(entropy_reg)
        self.eps = float(eps)

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature).clamp_min(self.eps)

    def _rho_view(
        self,
        rho: float | torch.Tensor,
        *,
        num_query: int,
        num_pairs: int,
        way_num: int,
        shot_num: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rho_tensor = torch.as_tensor(rho, device=reference.device, dtype=reference.dtype)
        if rho_tensor.dim() == 0:
            return rho_tensor.view(1, 1).expand(num_query, num_pairs)
        if rho_tensor.dim() == 2 and tuple(rho_tensor.shape) == (num_query, num_pairs):
            return rho_tensor
        if rho_tensor.dim() == 3 and tuple(rho_tensor.shape) == (num_query, way_num, shot_num):
            return rho_tensor.reshape(num_query, num_pairs)
        raise ValueError(
            "rho must be scalar or shaped (NumQuery, Way*Shot)/(NumQuery, Way, Shot), "
            f"got {tuple(rho_tensor.shape)}"
        )

    def _attention_logits(self, flat_cost: torch.Tensor) -> torch.Tensor:
        cost = flat_cost
        center = cost.mean(dim=(-1, -2), keepdim=True)
        scale = cost.std(dim=(-1, -2), unbiased=False, keepdim=True).clamp_min(self.eps)
        temperature = self.temperature.to(device=cost.device, dtype=cost.dtype)
        return -(cost - center) / (scale * temperature)

    def _mix_with_uniform(self, attention_pi: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
        eta = self.eta.to(device=attention_pi.device, dtype=attention_pi.dtype)
        pi = (1.0 - eta) * uniform + eta * attention_pi
        pi = pi.clamp_min(self.eps)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if bool((eta == 0).detach().cpu().item()):
            pi = uniform
        return pi

    def _diagnostics(
        self,
        query_pi: torch.Tensor,
        support_pi: torch.Tensor,
        query_uniform: torch.Tensor,
        support_uniform: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_entropy = -(query_pi * query_pi.clamp_min(self.eps).log()).sum(dim=-1)
        support_entropy = -(support_pi * support_pi.clamp_min(self.eps).log()).sum(dim=-1)
        query_norm_entropy = query_entropy / math.log(float(max(query_pi.shape[-1], 2)))
        support_norm_entropy = support_entropy / math.log(float(max(support_pi.shape[-1], 2)))
        entropy_floor = query_pi.new_tensor(0.35)
        entropy_loss = 0.5 * (
            torch.relu(entropy_floor - query_norm_entropy).pow(2).mean()
            + torch.relu(entropy_floor - support_norm_entropy).pow(2).mean()
        )
        query_kl = (query_pi * (query_pi.clamp_min(self.eps) / query_uniform.clamp_min(self.eps)).log()).sum(dim=-1)
        support_kl = (
            support_pi * (support_pi.clamp_min(self.eps) / support_uniform.clamp_min(self.eps)).log()
        ).sum(dim=-1)
        return {
            "mea_query_entropy": query_entropy,
            "mea_support_entropy": support_entropy,
            "mea_query_peak_ratio": query_pi.max(dim=-1).values / query_pi.mean(dim=-1).clamp_min(self.eps),
            "mea_support_peak_ratio": support_pi.max(dim=-1).values / support_pi.mean(dim=-1).clamp_min(self.eps),
            "mea_query_uniform_kl": query_kl,
            "mea_support_uniform_kl": support_kl,
            "mea_entropy_loss": entropy_loss,
            "mea_aux_loss": query_pi.new_tensor(float(self.entropy_reg)) * entropy_loss,
        }

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        rho: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (NumQuery, Way*Shot, QueryTokens, SupportTokens), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={way_num * shot_num}")

        logits = self._attention_logits(flat_cost)
        support_attention = torch.softmax(logits, dim=-1)
        query_attention = torch.softmax(logits, dim=-2)

        support_attention_pi = support_attention.mean(dim=-2)
        query_attention_pi = query_attention.mean(dim=-1)
        support_uniform = flat_cost.new_full((num_query, num_pairs, support_len), 1.0 / float(support_len))
        query_uniform = flat_cost.new_full((num_query, num_pairs, query_len), 1.0 / float(query_len))

        support_pi = self._mix_with_uniform(support_attention_pi, support_uniform)
        query_pi = self._mix_with_uniform(query_attention_pi, query_uniform)
        rho_view = self._rho_view(
            rho,
            num_query=num_query,
            num_pairs=num_pairs,
            way_num=way_num,
            shot_num=shot_num,
            reference=flat_cost,
        )
        query_marginal = rho_view.unsqueeze(-1) * query_pi
        support_marginal = rho_view.unsqueeze(-1) * support_pi

        expected = rho_view
        torch._assert(torch.isfinite(query_marginal).all(), "MEA query marginal contains non-finite values")
        torch._assert(torch.isfinite(support_marginal).all(), "MEA support marginal contains non-finite values")
        torch._assert((query_marginal > 0.0).all(), "MEA query marginal must be positive")
        torch._assert((support_marginal > 0.0).all(), "MEA support marginal must be positive")
        torch._assert(
            torch.isclose(query_marginal.sum(dim=-1), expected, atol=1e-5, rtol=1e-4).all(),
            "MEA query marginal must preserve the rho budget",
        )
        torch._assert(
            torch.isclose(support_marginal.sum(dim=-1), expected, atol=1e-5, rtol=1e-4).all(),
            "MEA support marginal must preserve the rho budget",
        )

        shape_prefix = (num_query, int(way_num), int(shot_num))
        aux: dict[str, torch.Tensor | Any] = {
            "mea_query_pi": query_pi.reshape(*shape_prefix, query_len),
            "mea_support_pi": support_pi.reshape(*shape_prefix, support_len),
            "mea_query_marginal": query_marginal.reshape(*shape_prefix, query_len),
            "mea_support_marginal": support_marginal.reshape(*shape_prefix, support_len),
            "mea_query_attention": query_attention.reshape(*shape_prefix, query_len, support_len),
            "mea_support_attention": support_attention.reshape(*shape_prefix, query_len, support_len),
            "mea_eta": self.eta.to(device=flat_cost.device, dtype=flat_cost.dtype),
            "mea_temperature": self.temperature.to(device=flat_cost.device, dtype=flat_cost.dtype),
        }
        diagnostics = self._diagnostics(query_pi, support_pi, query_uniform, support_uniform)
        aux.update({key: value.reshape(*shape_prefix) if value.dim() == 2 else value for key, value in diagnostics.items()})
        return query_marginal, support_marginal, aux  # type: ignore[return-value]


__all__ = ["MutualEvidenceAttentionMarginal"]
