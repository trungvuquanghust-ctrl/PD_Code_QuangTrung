"""Differentiable dustbin optimal transport for token rejection."""

from __future__ import annotations

import torch


EPS = 1e-8


def _validate_scores(
    scores: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float,
) -> None:
    if scores.dim() < 2:
        raise ValueError(f"scores must have at least 2 dims, got {tuple(scores.shape)}")
    if a.shape != scores.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(scores.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != scores.shape[:-2] + (scores.shape[-1],):
        raise ValueError(
            f"b must have shape {tuple(scores.shape[:-2] + (scores.shape[-1],))}, "
            f"got {tuple(b.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("a and b must be non-negative")
    residual = (a.sum(dim=-1) - b.sum(dim=-1)).abs()
    if torch.any(residual > 1e-5):
        raise ValueError("dustbin OT requires equal real query/support total mass")


def _broadcast_leading(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    leading_shape = reference.shape[:-2]
    if tensor.dim() == 0:
        return tensor.expand(leading_shape)
    return torch.broadcast_to(tensor, leading_shape)


def _uniform_real_marginals(cost: torch.Tensor, rho: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = cost.shape[-2:]
    rho_tensor = _broadcast_leading(rho, cost)
    a = rho_tensor.unsqueeze(-1).expand(*cost.shape[:-2], rows) / float(rows)
    b = rho_tensor.unsqueeze(-1).expand(*cost.shape[:-2], cols) / float(cols)
    return a, b


def build_dustbin_augmented_scores(
    scores: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Append one dustbin row and one dustbin column to a score matrix."""
    rows, cols = scores.shape[-2:]
    augmented = scores.new_empty(scores.shape[:-2] + (rows + 1, cols + 1))
    augmented[..., :rows, :cols] = scores
    alpha_tensor = torch.as_tensor(alpha, device=scores.device, dtype=scores.dtype)
    alpha_tensor = torch.broadcast_to(alpha_tensor, scores.shape[:-2])
    augmented[..., :rows, cols] = alpha_tensor.unsqueeze(-1).expand(*scores.shape[:-2], rows)
    augmented[..., rows, :cols] = alpha_tensor.unsqueeze(-1).expand(*scores.shape[:-2], cols)
    augmented[..., rows, cols] = alpha_tensor
    return augmented


def log_dustbin_sinkhorn(
    scores: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    rho: torch.Tensor | float = 1.0,
    max_iter: int = 60,
    temperature: float = 0.1,
    tol: float = 1e-5,
    return_augmented: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve balanced score-maximizing OT after adding dustbin row/column.

    Real rows and columns keep total mass ``rho``. The added row/column each
    carry the opposite side's total mass, letting real-real mass vary while the
    augmented problem remains balanced and differentiable.
    """
    if a is None or b is None:
        a, b = _uniform_real_marginals(scores, rho)
    else:
        a = a.to(device=scores.device, dtype=scores.dtype)
        b = b.to(device=scores.device, dtype=scores.dtype)
    _validate_scores(scores, a, b, temperature)

    dustbin_row = b.sum(dim=-1, keepdim=True)
    dustbin_col = a.sum(dim=-1, keepdim=True)
    augmented_a = torch.cat([a, dustbin_row], dim=-1)
    augmented_b = torch.cat([b, dustbin_col], dim=-1)
    augmented_scores = build_dustbin_augmented_scores(scores, alpha)

    log_a = torch.log(augmented_a.clamp_min(EPS))
    log_b = torch.log(augmented_b.clamp_min(EPS))
    log_kernel = augmented_scores / float(temperature)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        prev_log_v = log_v
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        delta = torch.maximum(
            (log_u - prev_log_u).abs().amax(),
            (log_v - prev_log_v).abs().amax(),
        )
        if tol > 0.0 and delta < tol:
            break

    augmented_plan = torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    plan = augmented_plan[..., :-1, :-1]
    if return_augmented:
        return plan, augmented_plan, augmented_a, augmented_b, augmented_scores
    return plan


def dustbin_transport_from_cost(
    cost: torch.Tensor,
    threshold: torch.Tensor | float,
    alpha: torch.Tensor | float,
    rho: torch.Tensor | float,
    *,
    temperature: float = 0.1,
    max_iter: int = 60,
    tol: float = 1e-5,
    return_augmented: bool = False,
) -> dict[str, torch.Tensor]:
    """Build ``threshold - cost`` scores and solve dustbin OT."""
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    threshold_tensor = torch.as_tensor(threshold, device=cost.device, dtype=cost.dtype)
    scores = threshold_tensor - cost
    a, b = _uniform_real_marginals(cost, rho)
    plan, augmented_plan, augmented_a, augmented_b, augmented_scores = log_dustbin_sinkhorn(
        scores,
        alpha,
        a=a,
        b=b,
        max_iter=max_iter,
        temperature=temperature,
        tol=tol,
        return_augmented=True,
    )
    out = {
        "plan": plan,
        "augmented_plan": augmented_plan,
        "augmented_a": augmented_a,
        "augmented_b": augmented_b,
        "augmented_scores": augmented_scores,
        "real_a": a,
        "real_b": b,
        "scores": scores,
        "transport_cost": (plan * cost).sum(dim=(-1, -2)),
        "real_mass": plan.sum(dim=(-1, -2)),
        "accepted_score": (plan * scores).sum(dim=(-1, -2)),
    }
    if not return_augmented:
        out = {key: value for key, value in out.items() if not key.startswith("augmented")}
    return out


def dustbin_transport_diagnostics(
    plan: torch.Tensor,
    augmented_plan: torch.Tensor,
    augmented_a: torch.Tensor,
    augmented_b: torch.Tensor,
    scores: torch.Tensor,
    rho: torch.Tensor | float,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Return mass and numerical diagnostics for a solved dustbin OT plan."""
    if plan.shape != scores.shape:
        raise ValueError(f"plan must have shape {tuple(scores.shape)}, got {tuple(plan.shape)}")
    if augmented_plan.shape != scores.shape[:-2] + (scores.shape[-2] + 1, scores.shape[-1] + 1):
        raise ValueError("augmented_plan shape is inconsistent with scores")

    rho_tensor = _broadcast_leading(rho, scores).clamp_min(eps)
    real_mass = plan.sum(dim=(-1, -2))
    query_dustbin_mass_per_token = augmented_plan[..., :-1, -1]
    support_dustbin_mass_per_token = augmented_plan[..., -1, :-1]
    query_dustbin_mass = query_dustbin_mass_per_token.sum(dim=-1)
    support_dustbin_mass = support_dustbin_mass_per_token.sum(dim=-1)
    accepted_score = (plan * scores).sum(dim=(-1, -2))
    query_best_score = scores.amax(dim=-1)
    rejected_query_score = (
        query_dustbin_mass_per_token * query_best_score
    ).sum(dim=-1) / query_dustbin_mass.clamp_min(eps)

    row_residual = (augmented_plan.sum(dim=-1) - augmented_a).abs().amax(dim=-1)
    column_residual = (augmented_plan.sum(dim=-2) - augmented_b).abs().amax(dim=-1)
    mass_residual = (
        augmented_plan.sum(dim=(-1, -2)) - augmented_a.sum(dim=-1)
    ).abs()

    return {
        "real_mass": real_mass,
        "real_mass_fraction": real_mass / rho_tensor,
        "query_dustbin_mass": query_dustbin_mass,
        "support_dustbin_mass": support_dustbin_mass,
        "query_dustbin_fraction": query_dustbin_mass / rho_tensor,
        "support_dustbin_fraction": support_dustbin_mass / rho_tensor,
        "dustbin_self_mass": augmented_plan[..., -1, -1],
        "accepted_edge_score_mean": accepted_score / real_mass.clamp_min(eps),
        "rejected_query_score_mean": rejected_query_score,
        "row_residual": row_residual,
        "column_residual": column_residual,
        "mass_residual": mass_residual,
    }
