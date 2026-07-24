"""Cross-Class Discriminative Marginals for fixed-budget ECOT.

CCDM derives non-uniform UOT marginals directly from the cost matrix using
cross-class contrast.  The query marginal up-weights tokens whose best-match
cost clearly separates one class from all others (cross-class margin).  The
support marginal up-weights tokens that are strongly attracted by at least one
query token (min-cost cross-reference).

Both marginals degrade gracefully under noise: when all classes look equally
distant (noise tokens), the cross-class margin collapses and mass reverts to
near-uniform; when no query token is nearby (noise support tokens), the
attractiveness signal is weak and mass is suppressed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


class CrossClassDiscriminativeMarginal(nn.Module):
    """Budget-preserving ECOT marginals from cross-class cost discrimination.

    Query marginal: tokens with large best-match cost margin between the
    nearest and second-nearest class receive more transport mass.

    Support marginal: tokens that strongly attract at least one query token
    (low min-cost over query dimension) receive more mass.

    Both priors are normalized to probability vectors and scaled by rho so
    that ``a.sum(-1) == rho`` and ``b.sum(-1) == rho``.
    """

    def __init__(
        self,
        *,
        tau_q_init: float = 0.50,
        tau_b_init: float = 0.50,
        entropy_reg: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if float(tau_q_init) <= 0.0:
            raise ValueError("tau_q_init must be positive")
        if float(tau_b_init) <= 0.0:
            raise ValueError("tau_b_init must be positive")
        if float(entropy_reg) < 0.0:
            raise ValueError("entropy_reg must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.raw_tau_q = nn.Parameter(
            torch.tensor(_inverse_softplus(float(tau_q_init)), dtype=torch.float32)
        )
        self.raw_tau_b = nn.Parameter(
            torch.tensor(_inverse_softplus(float(tau_b_init)), dtype=torch.float32)
        )
        self.entropy_reg = float(entropy_reg)
        self.eps = float(eps)

    @property
    def tau_q(self) -> torch.Tensor:
        return F.softplus(self.raw_tau_q).clamp_min(self.eps)

    @property
    def tau_b(self) -> torch.Tensor:
        return F.softplus(self.raw_tau_b).clamp_min(self.eps)

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        rho: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute CCDM marginals from the pairwise cost matrix.

        Args:
            flat_cost: ``(Nq, Way*Shot, Lq, Ls)`` token-token cost matrix.
            way_num: number of classes in the episode.
            shot_num: number of shots per class.
            rho: scalar or ``(Nq, Way*Shot)`` transport budget.

        Returns:
            query_marginal: ``(Nq, Way*Shot, Lq)`` with ``sum == rho``.
            support_marginal: ``(Nq, Way*Shot, Ls)`` with ``sum == rho``.
            aux: diagnostics including ``ccdm_support_entropy`` for shot
                weighting and ``ccdm_aux_loss`` for training regularization.
        """
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (Nq, Way*Shot, Lq, Ls), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(
                f"flat_cost pair dim {num_pairs} != Way*Shot={way_num * shot_num}"
            )

        tau_q = self.tau_q.to(device=flat_cost.device, dtype=flat_cost.dtype)
        tau_b = self.tau_b.to(device=flat_cost.device, dtype=flat_cost.dtype)

        # -- Query marginal: cross-class cost margin --
        # Reshape to (Nq, Way, Shot, Lq, Ls) to compute per-class best-match.
        cost_5d = flat_cost.reshape(num_query, way_num, shot_num, query_len, support_len)
        # best(i, c) = min over shots k and support tokens j
        best_per_class = cost_5d.amin(dim=(2, 4))  # (Nq, Way, Lq)

        if way_num >= 2:
            top2, _ = best_per_class.topk(2, dim=1, largest=False)  # (Nq, 2, Lq)
            margin = top2[:, 1, :] - top2[:, 0, :]  # (Nq, Lq)
        else:
            margin = best_per_class.squeeze(1)  # (Nq, Lq) — fallback for 1-way

        query_pi = torch.softmax(margin / tau_q, dim=-1)  # (Nq, Lq)

        # Broadcast query_pi identically to all (Way*Shot) pairs.
        query_pi_flat = query_pi.unsqueeze(1).expand(
            num_query, num_pairs, query_len
        )  # (Nq, Way*Shot, Lq)

        # -- Support marginal: min-cost cross-reference --
        # attract(j, c, k) = min_i C(q_i, z_{c,k,j})
        attract = flat_cost.amin(dim=-2)  # (Nq, Way*Shot, Ls)
        support_pi = torch.softmax(-attract / tau_b, dim=-1)  # (Nq, Way*Shot, Ls)

        # -- Apply rho budget --
        rho_t = torch.as_tensor(rho, device=flat_cost.device, dtype=flat_cost.dtype)
        if rho_t.dim() == 0:
            rho_view = rho_t.view(1, 1).expand(num_query, num_pairs)
        elif rho_t.dim() == 2:
            rho_view = rho_t
        elif rho_t.dim() == 3:
            rho_view = rho_t.reshape(num_query, num_pairs)
        else:
            raise ValueError(f"rho has unexpected shape {tuple(rho_t.shape)}")

        query_marginal = rho_view.unsqueeze(-1) * query_pi_flat
        support_marginal = rho_view.unsqueeze(-1) * support_pi

        # -- Diagnostics & entropy --
        # Support entropy per shot for shot-weighting downstream.
        sp_log = support_pi.clamp_min(self.eps).log()
        support_entropy_flat = -(support_pi * sp_log).sum(dim=-1)  # (Nq, Way*Shot)
        support_entropy = support_entropy_flat.reshape(
            num_query, way_num, shot_num
        )
        max_entropy = math.log(float(max(support_len, 2)))
        support_norm_entropy = support_entropy / max_entropy

        qp_log = query_pi.clamp_min(self.eps).log()
        query_entropy = -(query_pi * qp_log).sum(dim=-1)  # (Nq,)
        query_norm_entropy = query_entropy / math.log(float(max(query_len, 2)))

        entropy_floor = flat_cost.new_tensor(0.30)
        entropy_loss = 0.5 * (
            torch.relu(entropy_floor - query_norm_entropy).pow(2).mean()
            + torch.relu(entropy_floor - support_norm_entropy).pow(2).mean()
        )
        aux_loss = flat_cost.new_tensor(float(self.entropy_reg)) * entropy_loss

        shape_prefix = (num_query, way_num, shot_num)
        aux: dict[str, torch.Tensor] = {
            "ccdm_query_pi": query_pi,
            "ccdm_support_pi": support_pi.reshape(*shape_prefix, support_len),
            "ccdm_query_entropy": query_entropy,
            "ccdm_query_norm_entropy": query_norm_entropy,
            "ccdm_support_entropy": support_entropy,
            "ccdm_support_norm_entropy": support_norm_entropy,
            "ccdm_tau_q": tau_q,
            "ccdm_tau_b": tau_b,
            "ccdm_entropy_loss": entropy_loss,
            "ccdm_aux_loss": aux_loss,
        }
        if way_num >= 2:
            aux["ccdm_margin"] = margin
            aux["ccdm_mean_margin"] = margin.mean()

        return query_marginal, support_marginal, aux


__all__ = ["CrossClassDiscriminativeMarginal"]
