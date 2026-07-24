"""Nuisance-Referenced Optimal Transport (NR-OT) scoring head.

Full theory in ``NR_OT_DESIGN_NOTE.markdown``.  Compact summary:

PD scalograms decompose latently into a shared vertical-streak **background**
``B`` (near-identical across all four classes) and a sparse **discriminative
signature** ``S_c`` (the bright PD flame: where in frequency, how many blobs,
how spread).  Uniform-marginal UOT (Ours-Final) lets the abundant,
class-invariant ``B -> B`` transport dominate both the transported cost ``C``
and mass ``M``, so the threshold-mass score ``T*M - C`` barely separates
classes.  KL relaxation makes it worse: it drops the *expensive* signal tokens
(no cheap partner) and keeps the *cheap* background -- exactly backwards.

NR-OT introduces an episode-level, class-agnostic **nuisance reference** and
three coupled mechanisms:

  (A) Background reference ``beta_{-c}``: the empirical measure formed by
      pooling the support tokens of every OTHER class.  Class signatures differ
      across ways, so pooling averages them out and leaves the shared
      background.  ``beta_{-c}`` is a free, class-agnostic "null / background
      hypothesis" obtained directly from the episode -- no segmentation, no PD
      prior.

  (B) Background-novelty marginals: a token's transport mass is proportional to
      its departure (nearest-neighbour cost) from ``beta_{-c}``.  Tokens
      explained by the background get ~0 mass; the flame receives the budget.
      This inverts UOT's failure mode -- the solver now spends mass on signal.

  (C) Common-mode debiased score:
          score_c = scale * [ E(q, S_c) - E(q, beta_{-c}) ],   E = T*M - C.
      The background contributes ~equally to both transports and cancels,
      leaving the evidence that is *specific to class c above the background
      hypothesis*.  Because ``beta_{-c}`` is leave-class-out, this is a
      principled one-vs-rest contrastive score.

The solver reused here (``sinkhorn_unbalanced_log``) is the same verified
log-domain KL-UOT scaling used by Ours-Final, so NR-OT stays inside the exact
transport regime of the base model (eps, tau_q, tau_c, rho).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    sinkhorn_unbalanced_log,
)


def _inverse_softplus(value: float) -> float:
    """Return ``x`` such that ``softplus(x) == value`` (value > 0)."""
    import math

    value = float(value)
    if value <= 0.0:
        raise ValueError(f"threshold init must be positive, got {value}")
    # softplus(x) = log(1 + e^x); invert: x = log(e^value - 1).
    if value > 20.0:
        return value  # softplus ~ identity for large args
    return math.log(math.expm1(value))


class NuisanceReferencedOT(nn.Module):
    """Background-debiased, novelty-marginal partial transport scoring head.

    Forward inputs are the per-image local token sets already produced by the
    Ours-Final pipeline (projected, optionally L2-normalized).  All transport
    hyper-parameters are passed at call time so NR-OT shares the base model's
    transport regime exactly.

    The only learnable state is a positive evidence threshold ``T`` (mirrors
    Ours-Final's ``transport_cost_threshold``).
    """

    def __init__(
        self,
        *,
        novelty_temp: float = 0.5,
        threshold_init: float = 0.0625,
        uniform_reference_marginal: bool = True,
    ) -> None:
        super().__init__()
        if novelty_temp <= 0.0:
            raise ValueError(f"novelty_temp must be positive, got {novelty_temp}")
        self.novelty_temp = float(novelty_temp)
        self.uniform_reference_marginal = bool(uniform_reference_marginal)
        self.raw_threshold = nn.Parameter(
            torch.tensor(float(_inverse_softplus(threshold_init)))
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean cost between unit-norm token sets.

        ``x``: (..., Lx, D), ``y``: (..., Ly, D) -> (..., Lx, Ly).
        For L2-normalized rows, ``||x-y||^2 = 2 - 2 <x,y>``.
        """
        dot = torch.matmul(x, y.transpose(-1, -2))
        return (2.0 - 2.0 * dot).clamp_min(0.0)

    def _novelty_marginal(
        self,
        tokens: torch.Tensor,
        background: torch.Tensor,
        rho: float,
        eps: float,
    ) -> torch.Tensor:
        """Mass proportional to a token's departure from the background measure.

        ``tokens``: (..., L, D) unit-norm.  ``background``: (M, D) unit-norm.
        Returns marginal (..., L) summing to ``rho`` along the last axis.

        novelty(i)  = 1 - max_b <token_i, beta_b>      (== 0.5 * NN squared dist)
        z(i)        = (novelty(i) - mean_i) / std_i     (per-image standardization)
        a(i)        = rho * softmax_i( z(i) / novelty_temp )

        Standardizing the novelty across tokens before the softmax makes the
        concentration a *relative* anomaly score ("how many std above the
        typical token is this token's departure from the background"), so the
        temperature has a consistent meaning regardless of the absolute feature
        scale / dimensionality.  Background tokens cluster near z=0; the sparse
        PD flame sits in the positive tail and captures the budget.
        """
        # affinity to the nearest background atom (cosine, tokens are unit-norm)
        affinity = torch.matmul(tokens, background.transpose(-1, -2)).amax(dim=-1)
        novelty = (1.0 - affinity).clamp_min(0.0)
        mean = novelty.mean(dim=-1, keepdim=True)
        std = novelty.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        z = (novelty - mean) / std
        weight = torch.softmax(z / self.novelty_temp, dim=-1)
        return rho * weight

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        score_scale: float,
        eps: float,
        tau_q: float,
        tau_c: float,
        rho: float,
        sinkhorn_iters: int = 60,
        sinkhorn_tol: float = 1e-5,
        return_evidence_payload: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(logits, diagnostics)``.

        ``query_tokens``: (Q, L, D).  ``support_tokens``: (W, K, L, D).
        ``logits``: (Q, W) row-centered class scores.
        """
        if query_tokens.dim() != 3:
            raise ValueError(
                f"query_tokens must be (Q, L, D), got {tuple(query_tokens.shape)}"
            )
        if support_tokens.dim() != 4:
            raise ValueError(
                f"support_tokens must be (W, K, L, D), got {tuple(support_tokens.shape)}"
            )
        W, K, L, D = support_tokens.shape
        if (W, K) != (int(way_num), int(shot_num)):
            raise ValueError(
                f"support_tokens {tuple(support_tokens.shape)} disagrees with "
                f"way/shot={way_num}/{shot_num}"
            )
        if W < 2:
            raise ValueError(
                "NR-OT requires way_num >= 2 to form a leave-class-out background"
            )

        eps_f = float(eps)
        q = F.normalize(query_tokens, p=2, dim=-1, eps=eps_f)            # (Q, L, D)
        s = F.normalize(support_tokens, p=2, dim=-1, eps=eps_f)          # (W, K, L, D)
        s_flat = s.reshape(W, K * L, D)                                  # (W, M_c, D)
        Q = q.shape[0]
        Lq = q.shape[1]
        Ls = L
        threshold = F.softplus(self.raw_threshold).clamp_min(eps_f)

        class_scores: list[torch.Tensor] = []
        ref_scores: list[torch.Tensor] = []
        novelty_mass_frac: list[torch.Tensor] = []
        class_plans: list[torch.Tensor] = []
        class_costs: list[torch.Tensor] = []
        ref_plans: list[torch.Tensor] = []
        ref_costs: list[torch.Tensor] = []
        query_marginals: list[torch.Tensor] = []
        support_marginals: list[torch.Tensor] = []

        for c in range(W):
            s_c = s_flat[c]                                             # (M_c, D)
            other = [w for w in range(W) if w != c]
            beta_c = s_flat[other].reshape(len(other) * K * L, D)       # (M_b, D)

            # (B) novelty marginals w.r.t. the leave-class-out background.
            a_q = self._novelty_marginal(q, beta_c, rho, eps_f)         # (Q, L)
            b_c = self._novelty_marginal(s_c, beta_c, rho, eps_f)       # (M_c,)
            b_c = b_c.unsqueeze(0).expand(Q, -1)                        # (Q, M_c)

            # --- class transport:  E(q, S_c) ------------------------------
            cost_c = self._pairwise_sq_dist(q, s_c.unsqueeze(0).expand(Q, -1, -1))
            plan_c = sinkhorn_unbalanced_log(
                cost_c, a_q, b_c,
                tau_q=tau_q, tau_c=tau_c, eps=eps_f,
                max_iter=int(sinkhorn_iters), tol=float(sinkhorn_tol),
            )
            cost_term_c = compute_transport_cost(plan_c, cost_c)        # (Q,)
            mass_c = compute_transported_mass(plan_c)                   # (Q,)
            E_c = threshold * mass_c - cost_term_c

            # --- reference transport:  E(q, beta_{-c}) --------------------
            # Same query measure a_q (essential for common-mode cancellation),
            # target switched to the background reference.
            if self.uniform_reference_marginal:
                b_beta = a_q.new_full((Q, beta_c.shape[0]), rho / beta_c.shape[0])
            else:
                b_beta = self._novelty_marginal(beta_c, s_c, rho, eps_f)
                b_beta = b_beta.unsqueeze(0).expand(Q, -1)
            cost_b = self._pairwise_sq_dist(q, beta_c.unsqueeze(0).expand(Q, -1, -1))
            plan_b = sinkhorn_unbalanced_log(
                cost_b, a_q, b_beta,
                tau_q=tau_q, tau_c=tau_c, eps=eps_f,
                max_iter=int(sinkhorn_iters), tol=float(sinkhorn_tol),
            )
            cost_term_b = compute_transport_cost(plan_b, cost_b)        # (Q,)
            mass_b = compute_transported_mass(plan_b)                   # (Q,)
            E_b = threshold * mass_b - cost_term_b

            # (C) common-mode debiased class score.
            class_scores.append(E_c)
            ref_scores.append(E_b)
            if return_evidence_payload:
                class_plans.append(plan_c)
                class_costs.append(cost_c)
                ref_plans.append(plan_b)
                ref_costs.append(cost_b)
                query_marginals.append(a_q)
                support_marginals.append(b_c)

            # diagnostic: fraction of query mass on the top-20% novel tokens
            with torch.no_grad():
                k_top = max(1, int(a_q.shape[-1]) // 5)
                top_mass = a_q.topk(k_top, dim=-1).values.sum(dim=-1)
                novelty_mass_frac.append((top_mass / rho).mean())

        E_class = torch.stack(class_scores, dim=-1)                    # (Q, W)
        E_ref = torch.stack(ref_scores, dim=-1)                        # (Q, W)
        logits = float(score_scale) * (E_class - E_ref)
        logits = logits - logits.mean(dim=1, keepdim=True)

        diagnostics: dict[str, torch.Tensor] = {
            "nr_ot/threshold": threshold.detach(),
            "nr_ot/class_evidence_mean": E_class.detach().mean(),
            "nr_ot/ref_evidence_mean": E_ref.detach().mean(),
            "nr_ot/debias_gap_mean": (E_class - E_ref).detach().mean(),
            "nr_ot/novelty_top20_mass_frac": torch.stack(novelty_mass_frac).mean(),
        }
        if return_evidence_payload:
            class_plan = torch.stack(class_plans, dim=1)                 # (Q, W, L, K*L)
            class_cost = torch.stack(class_costs, dim=1)                 # (Q, W, L, K*L)
            class_plan_by_shot = class_plan.view(Q, W, Lq, K, Ls).permute(0, 1, 3, 2, 4)
            class_cost_by_shot = class_cost.view(Q, W, Lq, K, Ls).permute(0, 1, 3, 2, 4)
            support_marginal = torch.stack(support_marginals, dim=1).view(Q, W, K, Ls)
            diagnostics.update(
                {
                    # Payload for NR-OT-specific evidence export.  The class
                    # plan is reshaped to the same (Q, W, K, Lq, Ls) contract
                    # used by the existing UOT visualizer.
                    "nr_ot_class_transport_plan": class_plan_by_shot.detach(),
                    "nr_ot_class_cost_matrix": class_cost_by_shot.detach(),
                    "nr_ot_ref_transport_plan": torch.stack(ref_plans, dim=1).detach(),
                    "nr_ot_ref_cost_matrix": torch.stack(ref_costs, dim=1).detach(),
                    "nr_ot_query_marginal": torch.stack(query_marginals, dim=1).detach(),
                    "nr_ot_support_marginal": support_marginal.detach(),
                    "nr_ot_class_shot_transported_mass": class_plan_by_shot.sum(dim=(-1, -2)).detach(),
                    "nr_ot_class_shot_expected_mass": support_marginal.sum(dim=-1).detach(),
                    "nr_ot_class_evidence": E_class.detach(),
                    "nr_ot_ref_evidence": E_ref.detach(),
                    "nr_ot_debias_gap": (E_class - E_ref).detach(),
                    "nr_ot_transport_cost_threshold": threshold.detach(),
                }
            )
        return logits, diagnostics
