"""Episode-gated shrinkage marginals (EGSM) for fixed-budget ECOT / UOT.

Blends a maximum-entropy (uniform) token prior with a cost-derived candidate
prior.  A per-query gate kappa in [kappa_min, kappa_max] is produced by a small
MLP on ambiguity statistics of the episode cost tensor, so transport can move
away from uniform only when observed costs appear discriminative.

When ``enable_adaptive_rho`` is True, a second MLP head predicts a per-query
rho offset from the same psi features, producing rho_adaptive in
[base_rho - delta_max, base_rho + delta_max].  The predicted rho is returned
in the aux dict for the caller to use as the effective transport mass budget.
Gradient from Sinkhorn to rho is clipped to avoid ill-conditioning.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clip_rho_grad(grad: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """Clip rho gradients to prevent Sinkhorn-induced explosion."""
    return grad.clamp(-max_norm, max_norm)


class EpisodeGatedShrinkageMarginal(nn.Module):
    """Budget-preserving ECOT marginals: (1-kappa)*uniform + kappa*candidate.

    Candidate marginals follow the same cost-only construction as CCDM
    (cross-class margin on the query axis; min-cost attractiveness on support),
    but temperature values are fixed scalars (no extra learnable taus here).
    Candidate statistics are computed from stop-gradient, episode-normalized
    costs so the backbone learns the transport match instead of learning to
    game the marginal prior.
    """

    def __init__(
        self,
        *,
        psi_dim: int = 5,
        hidden_dim: int = 32,
        candidate_tau_q: float = 1.0,
        candidate_tau_b: float = 1.0,
        kappa_min: float = 0.05,
        kappa_max: float = 0.35,
        eps: float = 1e-8,
        enable_adaptive_rho: bool = False,
        rho_delta_max: float = 0.15,
        rho_grad_clip: float = 1.0,
        rho_reg_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(candidate_tau_q) <= 0.0 or float(candidate_tau_b) <= 0.0:
            raise ValueError("candidate_tau_q and candidate_tau_b must be positive")
        if not 0.0 <= float(kappa_min) < float(kappa_max) <= 1.0:
            raise ValueError("require 0 <= kappa_min < kappa_max <= 1")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if float(rho_delta_max) <= 0.0:
            raise ValueError("rho_delta_max must be positive")

        self.psi_dim = int(psi_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("candidate_tau_q", torch.tensor(float(candidate_tau_q)))
        self.register_buffer("candidate_tau_b", torch.tensor(float(candidate_tau_b)))
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.eps = float(eps)
        self.enable_adaptive_rho = bool(enable_adaptive_rho)
        self.rho_delta_max = float(rho_delta_max)
        self.rho_grad_clip = float(rho_grad_clip)
        self.rho_reg_lambda = float(rho_reg_lambda)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.psi_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.gate_mlp[0].weight, gain=0.1)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight, gain=0.1)
        nn.init.constant_(self.gate_mlp[2].bias, -3.0)

        if self.enable_adaptive_rho:
            self.rho_head = nn.Sequential(
                nn.Linear(self.psi_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.xavier_uniform_(self.rho_head[0].weight, gain=0.1)
            nn.init.zeros_(self.rho_head[0].bias)
            nn.init.zeros_(self.rho_head[2].weight)
            nn.init.zeros_(self.rho_head[2].bias)
        else:
            self.rho_head = None

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

    @staticmethod
    def _episode_psi(
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        eps: float,
    ) -> torch.Tensor:
        """Permutation-invariant (over ways) ambiguity features per query."""
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        w = int(way_num)
        k = int(shot_num)
        if num_pairs != w * k:
            raise ValueError(f"pair dim {num_pairs} != Way*Shot={w * k}")
        cost_5d = flat_cost.reshape(num_query, w, k, query_len, support_len)
        best_per_class = cost_5d.amin(dim=(2, 4))

        if w >= 2:
            top2, _ = best_per_class.topk(2, dim=1, largest=False)
            margin = (top2[:, 1, :] - top2[:, 0, :]).clamp_min(0.0)
            mean_margin = margin.mean(dim=-1)
            margin_std = margin.std(dim=-1, unbiased=False).clamp_min(eps)
            class_means = cost_5d.mean(dim=(2, 3, 4))
            p_cls = torch.softmax(-class_means / 0.5, dim=-1)
            ent = -(p_cls * p_cls.clamp_min(eps).log()).sum(dim=-1)
            ent_norm = ent / math.log(float(max(w, 2)))
        else:
            mean_margin = best_per_class.squeeze(1).mean(dim=-1)
            margin_std = best_per_class.squeeze(1).std(dim=-1, unbiased=False).clamp_min(eps)
            ent_norm = flat_cost.new_zeros(num_query)

        flat = flat_cost.reshape(num_query, -1)
        snr = flat.mean(dim=-1) / (flat.std(dim=-1, unbiased=False).clamp_min(eps))

        shot_mean = cost_5d.mean(dim=(3, 4))
        shot_var = shot_mean.var(dim=-1, unbiased=False).mean(dim=-1)

        psi = torch.stack(
            [
                torch.log1p(mean_margin.clamp_min(0.0)),
                torch.log1p(snr.clamp_min(0.0)),
                ent_norm.clamp(0.0, 1.0),
                torch.log1p(shot_var.clamp_min(0.0)),
                torch.log1p(margin_std),
            ],
            dim=-1,
        )
        return torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)

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
                f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        w = int(way_num)
        k = int(shot_num)
        rho_view = self._rho_view(
            rho,
            num_query=num_query,
            num_pairs=num_pairs,
            way_num=w,
            shot_num=k,
            reference=flat_cost,
        )

        tau_q = self.candidate_tau_q.to(device=flat_cost.device, dtype=flat_cost.dtype)
        tau_b = self.candidate_tau_b.to(device=flat_cost.device, dtype=flat_cost.dtype)

        prior_cost = flat_cost.detach()
        cost_5d = prior_cost.reshape(num_query, w, k, query_len, support_len)
        best_per_class = cost_5d.amin(dim=(2, 4))

        if w >= 2:
            top2, _ = best_per_class.topk(2, dim=1, largest=False)
            margin = top2[:, 1, :] - top2[:, 0, :]
            margin_scale = margin.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
            query_pi = torch.softmax((margin / margin_scale) / tau_q.clamp_min(self.eps), dim=-1)
        else:
            query_pi = flat_cost.new_full((num_query, query_len), 1.0 / float(query_len))

        attract = prior_cost.amin(dim=-2)
        attract_centered = attract - attract.mean(dim=-1, keepdim=True)
        attract_scale = attract_centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        support_pi = torch.softmax(-(attract_centered / attract_scale) / tau_b.clamp_min(self.eps), dim=-1)

        uq = 1.0 / float(query_len)
        us = 1.0 / float(support_len)

        psi = self._episode_psi(prior_cost, way_num=w, shot_num=k, eps=self.eps)
        gate_dtype = self.gate_mlp[0].weight.dtype
        logit = self.gate_mlp(psi.to(dtype=gate_dtype)).squeeze(-1).to(
            dtype=flat_cost.dtype,
            device=flat_cost.device,
        )
        span = self.kappa_max - self.kappa_min
        kappa = self.kappa_min + span * torch.sigmoid(logit)

        kappa_col = kappa.unsqueeze(-1)
        blended_query = (1.0 - kappa_col) * uq + kappa_col * query_pi
        kappa_pair = kappa.unsqueeze(-1).unsqueeze(-1)
        blended_support = (1.0 - kappa_pair) * us + kappa_pair * support_pi

        blended_query_pairs = blended_query.unsqueeze(1).expand(num_query, num_pairs, query_len)
        query_marginal = rho_view.unsqueeze(-1) * blended_query_pairs
        support_marginal = rho_view.unsqueeze(-1) * blended_support

        egsm_aux_loss = flat_cost.new_zeros((), dtype=flat_cost.dtype, device=flat_cost.device)

        rho_adaptive: torch.Tensor | None = None
        if self.rho_head is not None:
            rho_base_scalar = rho_view[:, 0].detach()
            rho_logit = self.rho_head(psi.to(dtype=gate_dtype)).squeeze(-1).to(
                dtype=flat_cost.dtype, device=flat_cost.device,
            )
            rho_adaptive = rho_base_scalar + self.rho_delta_max * torch.tanh(rho_logit)
            rho_adaptive = rho_adaptive.clamp(min=self.eps, max=1.0)
            if rho_adaptive.requires_grad:
                rho_adaptive.register_hook(
                    lambda g: _clip_rho_grad(g, max_norm=self.rho_grad_clip)
                )
            egsm_aux_loss = egsm_aux_loss + self.rho_reg_lambda * (
                rho_adaptive - rho_base_scalar
            ).pow(2).mean()

        aux: dict[str, Any] = {
            "egsm_kappa": kappa,
            "egsm_psi": psi,
            "egsm_query_pi": query_pi,
            "egsm_support_pi": support_pi.reshape(num_query, w, k, support_len),
            "egsm_candidate_tau_q": tau_q,
            "egsm_candidate_tau_b": tau_b,
            "egsm_aux_loss": egsm_aux_loss,
        }
        if rho_adaptive is not None:
            aux["egsm_rho_adaptive"] = rho_adaptive
        return query_marginal, support_marginal, aux


__all__ = ["EpisodeGatedShrinkageMarginal"]
