"""POT-guided token marginals for Ours-Final UOT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.modules.fast_partial_ot import fast_partial_wasserstein


class POTGuideMarginals(nn.Module):
    """Use a cheap no-grad POT plan to build non-uniform UOT marginals."""

    def __init__(
        self,
        *,
        fixed_s: float = 0.5,
        adaptive_s: bool = False,
        s_min: float = 0.2,
        s_max: float = 0.8,
        epsilon: float = 0.05,
        max_iter: int = 50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < float(fixed_s) <= 1.0:
            raise ValueError("pot_guide_s must be in (0, 1]")
        if not 0.0 <= float(s_min) < float(s_max) <= 1.0:
            raise ValueError("pot_guide_s_min and pot_guide_s_max must satisfy 0 <= min < max <= 1")
        if float(epsilon) <= 0.0:
            raise ValueError("pot_guide_epsilon must be positive")
        if int(max_iter) <= 0:
            raise ValueError("pot_guide_max_iter must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.fixed_s = float(fixed_s)
        self.adaptive_s = bool(adaptive_s)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.eps = float(eps)

        self.raw_alpha = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.s_predictor = (
            nn.Sequential(
                nn.Linear(4, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            if self.adaptive_s
            else None
        )
        if self.s_predictor is not None:
            nn.init.zeros_(self.s_predictor[-1].weight)
            nn.init.zeros_(self.s_predictor[-1].bias)

    def _predict_s(self, cost: torch.Tensor) -> torch.Tensor:
        leading_shape = cost.shape[:-2]
        if self.s_predictor is None:
            return cost.new_full(leading_shape, self.fixed_s)

        row_min = cost.detach().amin(dim=-1)
        stats = torch.stack(
            (
                row_min.mean(dim=-1),
                row_min.std(dim=-1, unbiased=False),
                row_min.amin(dim=-1),
                row_min.amax(dim=-1),
            ),
            dim=-1,
        )
        network_dtype = self.s_predictor[0].weight.dtype
        raw = self.s_predictor(stats.to(dtype=network_dtype)).squeeze(-1)
        raw = raw.to(device=cost.device, dtype=cost.dtype)
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(raw)

    def forward(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if cost.dim() < 3:
            raise ValueError(f"cost must have shape (..., Lq, Ls), got {tuple(cost.shape)}")
        query_len = cost.shape[-2]
        support_len = cost.shape[-1]
        leading_shape = cost.shape[:-2]
        if tuple(rho.shape) != tuple(leading_shape):
            raise ValueError(f"rho shape {tuple(rho.shape)} does not match cost leading shape {tuple(leading_shape)}")

        uniform_q = cost.new_full((*leading_shape, query_len), 1.0 / float(query_len))
        uniform_s = cost.new_full((*leading_shape, support_len), 1.0 / float(support_len))
        s = self._predict_s(cost).to(device=cost.device, dtype=cost.dtype)

        with torch.no_grad():
            pot_plan = fast_partial_wasserstein(
                cost=cost.detach(),
                a=uniform_q,
                b=uniform_s,
                transport_mass=s.detach(),
                reg=self.epsilon,
                max_iter=self.max_iter,
                eps=self.eps,
            )

        w_q = pot_plan.sum(dim=-1)
        w_s = pot_plan.sum(dim=-2)
        if self.s_predictor is not None:
            s_scale = s / s.detach().clamp_min(self.eps)
            w_q = w_q * s_scale.unsqueeze(-1)
            w_s = w_s * s_scale.unsqueeze(-1)

        alpha = torch.sigmoid(self.raw_alpha).to(device=cost.device, dtype=cost.dtype)
        temperature = self.log_tau.exp().to(device=cost.device, dtype=cost.dtype).clamp_min(self.eps)
        query_prob = F.softmax(w_q / temperature, dim=-1)
        support_prob = F.softmax(w_s / temperature, dim=-1)
        rho = rho.to(device=cost.device, dtype=cost.dtype)
        a = rho.unsqueeze(-1) * (alpha * query_prob + (1.0 - alpha) * uniform_q)
        b = rho.unsqueeze(-1) * (alpha * support_prob + (1.0 - alpha) * uniform_s)

        diagnostics = {
            "pot_guide/alpha": alpha.detach(),
            "pot_guide/temperature": temperature.detach(),
            "pot_guide/s_mean": s.detach().mean(),
            "pot_guide/pot_sparsity": (pot_plan < self.eps).to(dtype=cost.dtype).mean().detach(),
            "pot_guide/marginal_q_max": a.detach().amax(),
            "pot_guide/marginal_q_min": a.detach().amin(),
        }
        return a, b, diagnostics


__all__ = ["POTGuideMarginals"]
