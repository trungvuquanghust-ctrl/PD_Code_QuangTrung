"""
FGW-UOT Solver: Fused Gromov-Wasserstein Unbalanced Optimal Transport.

References:
  Vayer et al. "Optimal Transport for structured data" (2019)
  Chizat et al. "Scaling algorithms for unbalanced OT" (2018)
"""

from typing import Tuple

import torch


def pairwise_sq_l2(
    A: torch.Tensor,   # (..., N, D)
    B: torch.Tensor,   # (..., M, D)
) -> torch.Tensor:     # (..., N, M)
    """Batched pairwise squared Euclidean distance. Safe for autograd."""
    A_sq = (A * A).sum(dim=-1, keepdim=True)
    B_sq = (B * B).sum(dim=-1, keepdim=True)
    AB = torch.matmul(A, B.transpose(-2, -1))
    return (A_sq + B_sq.transpose(-2, -1) - 2.0 * AB).clamp_min(0.0)


def normalize_intra_dist(
    D: torch.Tensor,   # (..., N, N)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize an intra-distribution distance matrix by its batch mean."""
    return D / D.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)


def _sinkhorn_balanced_log(
    cost: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    log_kernel = -cost / float(eps)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max_iter):
        prev_log_u = log_u
        prev_log_v = log_v
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        if tol > 0.0:
            delta = torch.maximum(
                (log_u - prev_log_u).abs().amax(),
                (log_v - prev_log_v).abs().amax(),
            )
            if delta < tol:
                break
    return (log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).exp()


def sinkhorn_uot_log(
    cost: torch.Tensor,    # (B, N, M)
    log_a: torch.Tensor,   # (B, N)
    log_b: torch.Tensor,   # (B, M)
    tau: float,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:         # (B, N, M)
    """
    Log-domain stabilized KL-relaxed unbalanced Sinkhorn solver.

    Solves:
      min_P <P,C> + tau KL(P 1|a) + tau KL(P'1|b)
            + eps KL(P|a otimes b)
    """
    if cost.dim() != 3:
        raise ValueError(f"cost must have shape (B, N, M), got {tuple(cost.shape)}")
    if log_a.shape != cost.shape[:2]:
        raise ValueError(f"log_a must have shape {tuple(cost.shape[:2])}, got {tuple(log_a.shape)}")
    if log_b.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError(f"log_b must have shape {(cost.shape[0], cost.shape[2])}, got {tuple(log_b.shape)}")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    rho_s = float(tau) / (float(tau) + float(eps))
    if rho_s > 0.999:
        return _sinkhorn_balanced_log(cost, log_a, log_b, eps, max(max_iter, 1000), tol)

    log_kernel = -cost / float(eps)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        prev_log_v = log_v

        log_v = rho_s * (log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2))
        log_u = rho_s * (log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1))

        if tol > 0.0:
            delta = torch.maximum(
                (log_u - prev_log_u).abs().amax(),
                (log_v - prev_log_v).abs().amax(),
            )
            if delta < tol:
                break

    log_P = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return log_P.exp()


def fgw_uot_solve(
    C_feat: torch.Tensor,         # (B, Tq, Ts)
    D_q: torch.Tensor,            # (B, Tq, Tq)
    D_s: torch.Tensor,            # (B, Ts, Ts)
    log_a: torch.Tensor,          # (B, Tq)
    log_b: torch.Tensor,          # (B, Ts)
    alpha: torch.Tensor,          # ()
    tau: float,
    eps: float,
    fgw_iters: int = 8,
    sinkhorn_iters: int = 60,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linearized squared-loss FGW-UOT fixed-point solver.

    P_prev is detached for fixed-point stability, while D_q and D_s retain
    gradients so the structure term still trains the backbone.
    """
    alpha_c = alpha.clamp(0.0, 1.0)

    P = (log_a.unsqueeze(-1) + log_b.unsqueeze(-2)).exp()
    P = P / P.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

    C_t = C_feat
    for _ in range(fgw_iters):
        P_prev = P.detach()
        a_prev = P_prev.sum(dim=-1, keepdim=True)
        b_prev = P_prev.sum(dim=-2, keepdim=True)
        term1 = torch.bmm(D_q * D_q, a_prev)
        term2 = torch.bmm(b_prev, D_s * D_s)
        cross_term = torch.bmm(torch.bmm(D_q, P_prev), D_s)
        gw_cost = (term1 + term2 - 2.0 * cross_term).clamp_min(0.0)
        C_t = (1.0 - alpha_c) * C_feat + alpha_c * gw_cost
        P = sinkhorn_uot_log(
            C_t,
            log_a,
            log_b,
            tau=tau,
            eps=eps,
            max_iter=sinkhorn_iters,
            tol=tol,
        )

    return P, C_t
