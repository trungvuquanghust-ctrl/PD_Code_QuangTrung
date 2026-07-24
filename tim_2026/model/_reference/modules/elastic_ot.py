"""Entropic Elastic Optimal Transport with adaptive matched mass."""

from __future__ import annotations

import torch

from tim_2026.model._reference.modules.unbalanced_ot import sinkhorn_balanced_log


def _validate_elastic_inputs(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sigma: float,
) -> None:
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if a.shape != cost.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError(
            f"b must have shape {tuple(cost.shape[:-2] + (cost.shape[-1],))}, "
            f"got {tuple(b.shape)}"
        )
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("a and b must be non-negative")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")


def build_elastic_augmented_problem(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert capacity-constrained Elastic OT to balanced OT.

    The real block solves

    ``min <P, cost>  s.t. P1 <= a, P^T1 <= b``.

    A final slack row and column absorb unmatched capacity. The ``sigma``
    terms add the same constant to every feasible augmented plan, so they do
    not change the real transport block.
    """
    _validate_elastic_inputs(cost, a, b, sigma)
    rows, cols = cost.shape[-2:]
    augmented_cost = cost.new_empty(cost.shape[:-2] + (rows + 1, cols + 1))
    augmented_cost[..., :rows, :cols] = cost
    augmented_cost[..., :rows, cols] = float(sigma)
    augmented_cost[..., rows, :cols] = float(sigma)
    augmented_cost[..., rows, cols] = 2.0 * float(sigma)

    augmented_a = torch.cat([a, b.sum(dim=-1, keepdim=True)], dim=-1)
    augmented_b = torch.cat([b, a.sum(dim=-1, keepdim=True)], dim=-1)
    return augmented_cost, augmented_a, augmented_b


def sinkhorn_elastic_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
    sigma: float = 1.0,
    max_iter: int = 60,
    tol: float = 1e-5,
    return_augmented: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve entropic Elastic OT using one augmented balanced Sinkhorn."""
    augmented_cost, augmented_a, augmented_b = build_elastic_augmented_problem(
        cost,
        a,
        b,
        sigma=sigma,
    )
    augmented_plan = sinkhorn_balanced_log(
        augmented_cost,
        augmented_a,
        augmented_b,
        eps=eps,
        max_iter=max_iter,
        tol=tol,
    )
    plan = augmented_plan[..., :-1, :-1]
    if return_augmented:
        return plan, augmented_plan, augmented_a, augmented_b
    return plan


def elastic_capacity_residuals(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return positive violations of the Elastic OT capacity constraints."""
    if plan.shape != a.shape + (b.shape[-1],):
        raise ValueError(
            f"plan must have shape {tuple(a.shape + (b.shape[-1],))}, "
            f"got {tuple(plan.shape)}"
        )
    row_violation = (plan.sum(dim=-1) - a).clamp_min(0.0)
    column_violation = (plan.sum(dim=-2) - b).clamp_min(0.0)
    return {
        "row_violation": row_violation.amax(dim=-1),
        "column_violation": column_violation.amax(dim=-1),
    }
