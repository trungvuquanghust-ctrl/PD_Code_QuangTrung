"""Cost-Derived Evidence Marginals for Unbalanced Optimal Transport.

Computes class-conditioned, episode-adaptive token marginals from the
pairwise transport cost matrix.  Instead of treating every spatial token
with uniform mass (rho / L), each token receives mass proportional to
its evidence relevance: tokens whose cost to the paired set is low
(strong match) are up-weighted, while background/noise tokens are
suppressed.

The marginals are *class-conditioned*: a query token receives different
mass when matched against different support classes, because its match
quality varies across classes.

Paper formulation
-----------------
Standard UOT:  a_i = rho / L  (uniform)
Evidence UOT:  a_i^c = rho * softmax_i( evidence_i^c / tau_m )

where evidence_i^c = -tau_e * log sum_j exp(-C(q_i, s_{c,j}) / tau_e)
is the soft minimum cost of matching query token i to class c.

Ablation modes
--------------
- ``query_only``:   evidence marginals for query tokens only
- ``support_only``: evidence marginals for support tokens only
- ``both``:         evidence marginals for both sides
- ``rival_aware``:  query marginals subtract rival-class evidence
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-8

EVIDENCE_MODES = frozenset({"query_only", "support_only", "both", "rival_aware"})


def _normalize_evidence_mode(value: str | None) -> str:
    name = "both" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "query": "query_only",
        "q": "query_only",
        "support": "support_only",
        "s": "support_only",
        "all": "both",
        "full": "both",
        "rival": "rival_aware",
        "discriminative": "rival_aware",
        "disc": "rival_aware",
    }
    name = aliases.get(name, name)
    if name not in EVIDENCE_MODES:
        raise ValueError(
            f"Unsupported evidence_mode: {value}. "
            f"Expected one of {sorted(EVIDENCE_MODES)}"
        )
    return name


def compute_query_evidence(
    cost: torch.Tensor,
    tau_evidence: float,
    eps: float = EPS,
) -> torch.Tensor:
    """Negative soft-min cost for each query token against support tokens.

    For query token *i* matched against a support set, the evidence score
    is the negated soft minimum of the cost row::

        e_i = tau_e * logsumexp(-C[i, :] / tau_e, dim=-1)

    This equals ``-softmin(C[i, :])`` up to a constant.  A token with a
    nearby support match (low cost) yields a *high* evidence score; a
    background token far from all support tokens yields a *low* (very
    negative) score.  Feeding these into ``softmax`` in
    ``evidence_to_marginal`` therefore concentrates mass on evidence-rich
    tokens and suppresses background.

    Parameters
    ----------
    cost : Tensor of shape ``(..., Lq, Ls)``
        Pairwise cost matrix (e.g. squared Euclidean distance, non-negative).
    tau_evidence : float
        Temperature for the soft-min.  Smaller values approach hard min.

    Returns
    -------
    Tensor of shape ``(..., Lq)``
        Per-query-token evidence scores (higher = more evidence).
    """
    tau = max(float(tau_evidence), eps)
    # logsumexp(-C/tau) ≈ -min(C)/tau  →  tau*logsumexp ≈ -min(C)
    # good match (small C) → ≈ 0;  bad match (large C) → very negative
    return tau * torch.logsumexp(-cost / tau, dim=-1)


def compute_support_evidence(
    cost: torch.Tensor,
    tau_evidence: float,
    eps: float = EPS,
) -> torch.Tensor:
    """Negative soft-min cost for each support token against query tokens.

    Same convention as :func:`compute_query_evidence` but reduces over
    the query dimension (``dim=-2``).

    Parameters
    ----------
    cost : Tensor of shape ``(..., Lq, Ls)``
    tau_evidence : float

    Returns
    -------
    Tensor of shape ``(..., Ls)``
        Per-support-token evidence scores (higher = more evidence).
    """
    tau = max(float(tau_evidence), eps)
    return tau * torch.logsumexp(-cost / tau, dim=-2)


def evidence_to_marginal(
    evidence: torch.Tensor,
    tau_marginal: float,
    rho: torch.Tensor | float,
    eps: float = EPS,
) -> torch.Tensor:
    """Convert evidence scores to UOT marginal probabilities.

    ``marginal_i = rho * softmax_i(evidence / tau_m)``

    Parameters
    ----------
    evidence : Tensor of shape ``(..., L)``
    tau_marginal : float
    rho : scalar or broadcastable Tensor

    Returns
    -------
    Tensor of shape ``(..., L)``
    """
    tau = max(float(tau_marginal), eps)
    prob = torch.softmax(evidence / tau, dim=-1)
    if isinstance(rho, (int, float)):
        return float(rho) * prob
    return rho * prob


def compute_rival_discriminative_query_evidence(
    cost: torch.Tensor,
    *,
    way_num: int,
    shot_num: int,
    tau_evidence: float,
    rival_margin: float = 0.5,
    eps: float = EPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return query evidence after subtracting class-excluded rival evidence.

    This factors out the ``mode='rival_aware'`` query branch so other
    Ours-Final components can reuse the same discriminative signal without
    reimplementing the rival aggregation.
    """
    if cost.dim() != 4:
        raise ValueError(
            f"cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(cost.shape)}"
        )
    num_query, num_pairs, query_len, _support_len = cost.shape
    way_num = int(way_num)
    shot_num = int(shot_num)
    if way_num <= 1:
        raise ValueError("rival-aware evidence requires at least two classes")
    if shot_num <= 0 or num_pairs != way_num * shot_num:
        raise ValueError(
            f"cost pair dimension {num_pairs} != way*shot={way_num * shot_num}"
        )
    if float(rival_margin) < 0.0:
        raise ValueError("rival_margin must be non-negative")

    q_evidence = compute_query_evidence(cost, tau_evidence, eps)
    q_evidence_by_class = q_evidence.reshape(
        num_query,
        way_num,
        shot_num,
        query_len,
    )
    class_evidence = q_evidence_by_class.mean(dim=2)
    rival_sum = class_evidence.sum(dim=1, keepdim=True) - class_evidence
    rival_evidence = rival_sum / float(max(way_num - 1, 1))
    disc_evidence = class_evidence - float(rival_margin) * rival_evidence
    disc_evidence_expanded = (
        disc_evidence.unsqueeze(2)
        .expand(-1, -1, shot_num, -1)
        .reshape(num_query, num_pairs, query_len)
    )
    diagnostics = {
        "evidence/rival_margin": cost.new_tensor(float(rival_margin)),
        "evidence/class_evidence_std": class_evidence.std(dim=1).mean().detach(),
    }
    return disc_evidence_expanded, diagnostics


class CostEvidenceMarginals(nn.Module):
    """Cost-derived evidence marginals for UOT.

    Computes episode-conditioned, class-discriminative token marginals
    from the pairwise transport cost matrix.

    Parameters
    ----------
    tau_evidence : float
        Temperature for the soft-min evidence computation.
        Smaller values approximate hard minimum (sharper evidence).
    tau_marginal : float
        Temperature for converting evidence to marginal probabilities.
        Smaller values produce peakier (more concentrated) marginals.
    mode : str
        One of ``'query_only'``, ``'support_only'``, ``'both'``,
        ``'rival_aware'``.
    rival_margin : float
        Scaling factor for rival-class evidence subtraction in
        ``'rival_aware'`` mode.  Higher values make marginals more
        class-discriminative.
    detach_cost : bool
        If True (default), detach the cost matrix before computing
        evidence.  Prevents gradient flow through the evidence
        computation, improving training stability.
    """

    def __init__(
        self,
        tau_evidence: float = 0.1,
        tau_marginal: float = 1.0,
        mode: str = "both",
        rival_margin: float = 0.5,
        detach_cost: bool = True,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        if tau_evidence <= 0.0:
            raise ValueError("tau_evidence must be positive")
        if tau_marginal <= 0.0:
            raise ValueError("tau_marginal must be positive")
        if rival_margin < 0.0:
            raise ValueError("rival_margin must be non-negative")
        self.tau_evidence = float(tau_evidence)
        self.tau_marginal = float(tau_marginal)
        self.mode = _normalize_evidence_mode(mode)
        self.rival_margin = float(rival_margin)
        self.detach_cost = bool(detach_cost)
        self.eps = float(eps)

    def extra_repr(self) -> str:
        return (
            f"tau_evidence={self.tau_evidence}, "
            f"tau_marginal={self.tau_marginal}, "
            f"mode={self.mode!r}, "
            f"rival_margin={self.rival_margin}, "
            f"detach_cost={self.detach_cost}"
        )

    def forward(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Compute evidence marginals from cost matrix.

        Parameters
        ----------
        cost : Tensor of shape ``(Nq, Way*Shot, Lq, Ls)``
            Pairwise transport cost.
        rho : Tensor
            Transport budget (scalar or broadcastable).
        way_num, shot_num : int
            Episode structure.

        Returns
        -------
        query_marginal : Tensor of shape ``(Nq, Way*Shot, Lq)`` or None
            Class-conditioned query marginals.  None if mode is
            ``'support_only'``.
        support_marginal : Tensor of shape ``(Way, Shot, Ls)`` or None
            Support marginals.  None if mode is ``'query_only'``.
        diagnostics : dict
            Diagnostic tensors for logging.
        """
        if cost.dim() != 4:
            raise ValueError(
                f"cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = cost.shape
        if num_pairs != way_num * shot_num:
            raise ValueError(
                f"cost pair dimension {num_pairs} != way*shot={way_num * shot_num}"
            )

        c = cost.detach() if self.detach_cost else cost
        rho_val = rho.to(device=cost.device, dtype=cost.dtype)

        diagnostics: dict[str, torch.Tensor] = {}
        query_marginal: torch.Tensor | None = None
        support_marginal: torch.Tensor | None = None

        compute_query = self.mode in ("query_only", "both", "rival_aware")
        compute_support = self.mode in ("support_only", "both", "rival_aware")

        if compute_query:
            q_evidence = compute_query_evidence(c, self.tau_evidence, self.eps)

            if self.mode == "rival_aware" and way_num > 1:
                q_evidence, rival_diag = compute_rival_discriminative_query_evidence(
                    c,
                    way_num=way_num,
                    shot_num=shot_num,
                    tau_evidence=self.tau_evidence,
                    rival_margin=self.rival_margin,
                    eps=self.eps,
                )
                diagnostics.update(rival_diag)

            query_marginal = evidence_to_marginal(
                q_evidence, self.tau_marginal, rho_val, self.eps
            )
            diagnostics["evidence/query_entropy"] = (
                -(query_marginal / rho_val.clamp_min(self.eps))
                .clamp_min(self.eps)
                .log()
                .mul(query_marginal / rho_val.clamp_min(self.eps))
                .sum(dim=-1)
                .mean()
                .detach()
            )
            diagnostics["evidence/query_marginal_max"] = query_marginal.amax(dim=-1).mean().detach()
            diagnostics["evidence/query_marginal_min"] = query_marginal.amin(dim=-1).mean().detach()

        if compute_support:
            s_evidence_per_pair = compute_support_evidence(c, self.tau_evidence, self.eps)

            s_evidence_per_shot = s_evidence_per_pair.mean(dim=0)
            s_evidence_per_shot = s_evidence_per_shot.reshape(
                way_num, shot_num, support_len
            )

            support_marginal = evidence_to_marginal(
                s_evidence_per_shot, self.tau_marginal, rho_val, self.eps
            )
            diagnostics["evidence/support_entropy"] = (
                -(support_marginal / rho_val.clamp_min(self.eps))
                .clamp_min(self.eps)
                .log()
                .mul(support_marginal / rho_val.clamp_min(self.eps))
                .sum(dim=-1)
                .mean()
                .detach()
            )
            diagnostics["evidence/support_marginal_max"] = support_marginal.amax(dim=-1).mean().detach()
            diagnostics["evidence/support_marginal_min"] = support_marginal.amin(dim=-1).mean().detach()

        diagnostics["evidence/mode"] = cost.new_tensor(
            float(list(EVIDENCE_MODES).index(self.mode))
        )
        diagnostics["evidence/tau_evidence"] = cost.new_tensor(self.tau_evidence)
        diagnostics["evidence/tau_marginal"] = cost.new_tensor(self.tau_marginal)

        return query_marginal, support_marginal, diagnostics
