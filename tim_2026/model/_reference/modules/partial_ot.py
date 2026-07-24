"""Partial optimal transport solvers with native PyTorch and POT backends.

The core problem is the standard partial Wasserstein transport:

    min_P <P, C>
    s.t. P 1 <= a, P^T 1 <= b, P >= 0, sum(P) = m.

This is useful when only a fraction of two token measures should be matched,
for example foreground/evidence regions in cluttered feature maps.
"""

from __future__ import annotations

import torch

from tim_2026.model._reference.modules.fast_partial_ot import (
    fast_partial_wasserstein,
    fast_partial_wasserstein_nesterov,
)

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is unavailable
    ot = None


EPS = 1e-8


def _validate_partial_inputs(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
) -> None:
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if a.shape != cost.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError(f"b must have shape {tuple(cost.shape[:-2] + (cost.shape[-1],))}, got {tuple(b.shape)}")
    if reg <= 0.0:
        raise ValueError("reg must be positive")
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("a and b must be non-negative")


def _prepare_transport_mass(
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None,
    transport_mass_ratio: float | torch.Tensor | None = None,
    mass_ratio: float | torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    return resolve_partial_transport_mass(
        a,
        b,
        transport_mass=transport_mass,
        transport_mass_ratio=transport_mass_ratio,
        mass_ratio=mass_ratio,
        eps=eps,
    )


def _as_leading_tensor(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
    leading_shape: torch.Size,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        resolved = value.to(device=reference.device, dtype=reference.dtype)
        if resolved.dim() == 0:
            return resolved.expand(leading_shape)
        if tuple(resolved.shape) != tuple(leading_shape):
            raise ValueError(
                f"{name} tensor must be scalar or have leading shape "
                f"{tuple(leading_shape)}, got {tuple(resolved.shape)}"
            )
        return resolved
    return reference.new_full(leading_shape, float(value))


def resolve_partial_transport_mass(
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    transport_mass_ratio: float | torch.Tensor | None = None,
    mass_ratio: float | torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Resolve absolute or relative transported mass for partial OT.

    ``transport_mass`` is an absolute amount. ``transport_mass_ratio`` and
    ``mass_ratio`` are aliases for a fraction of ``min(sum(a), sum(b))``.
    """
    leading_shape = a.shape[:-1]
    max_mass = torch.minimum(a.sum(dim=-1), b.sum(dim=-1))
    ratio = transport_mass_ratio

    if transport_mass_ratio is not None and mass_ratio is not None:
        raise ValueError("Only one of transport_mass_ratio and mass_ratio may be set")
    if ratio is None:
        ratio = mass_ratio
    if transport_mass is not None and ratio is not None:
        raise ValueError("transport_mass cannot be set together with transport_mass_ratio or mass_ratio")

    if transport_mass is None and ratio is None:
        mass = max_mass
    elif ratio is not None:
        ratio_tensor = _as_leading_tensor(
            ratio,
            reference=a,
            leading_shape=leading_shape,
            name="transport_mass_ratio",
        )
        if torch.any(ratio_tensor < 0.0) or torch.any(ratio_tensor > 1.0):
            raise ValueError("transport_mass_ratio and mass_ratio must be in [0, 1]")
        mass = ratio_tensor * max_mass
    else:
        mass = _as_leading_tensor(
            transport_mass,
            reference=a,
            leading_shape=leading_shape,
            name="transport_mass",
        )

    if torch.any(mass < 0.0):
        raise ValueError("transport_mass must be non-negative")
    if torch.any(mass > max_mass + eps):
        raise ValueError("transport_mass must be <= min(sum(a), sum(b)) for every problem")
    return torch.minimum(mass, max_mass)


def partial_transport_plan_residuals(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-problem partial OT constraint residuals."""
    if plan.shape != a.shape + (b.shape[-1],):
        raise ValueError(f"plan must have shape {tuple(a.shape + (b.shape[-1],))}, got {tuple(plan.shape)}")
    if b.shape != plan.shape[:-2] + (plan.shape[-1],):
        raise ValueError(f"b must have shape {tuple(plan.shape[:-2] + (plan.shape[-1],))}, got {tuple(b.shape)}")
    leading_shape = plan.shape[:-2]
    if isinstance(transport_mass, torch.Tensor):
        mass = transport_mass.to(device=plan.device, dtype=plan.dtype)
    else:
        mass = plan.new_full(leading_shape, float(transport_mass))
    if mass.dim() == 0:
        mass = mass.expand(leading_shape)
    else:
        if tuple(mass.shape) != tuple(leading_shape):
            raise ValueError(
                "transport_mass must be scalar or have leading shape "
                f"{tuple(leading_shape)}, got {tuple(mass.shape)}"
            )

    row_violation = (plan.sum(dim=-1) - a).clamp_min(0.0).amax(dim=-1)
    col_violation = (plan.sum(dim=-2) - b).clamp_min(0.0).amax(dim=-1)
    mass_residual = (plan.sum(dim=(-1, -2)) - mass).abs()
    return {
        "row_violation": row_violation,
        "column_violation": col_violation,
        "mass_residual": mass_residual,
    }


def validate_partial_transport_plan(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor,
    tol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Validate partial OT row/column inequalities and total transported mass."""
    residuals = partial_transport_plan_residuals(plan, a, b, transport_mass)
    if torch.any(residuals["row_violation"] > tol):
        raise FloatingPointError("partial OT row marginal inequality residual exceeds tolerance")
    if torch.any(residuals["column_violation"] > tol):
        raise FloatingPointError("partial OT column marginal inequality residual exceeds tolerance")
    if torch.any(residuals["mass_residual"] > tol):
        raise FloatingPointError("partial OT transported mass residual exceeds tolerance")
    return residuals


def _slack_outer_plan(
    row_slack: torch.Tensor,
    col_slack: torch.Tensor,
    mass: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    row_total = row_slack.sum(dim=-1, keepdim=True)
    col_total = col_slack.sum(dim=-1, keepdim=True)
    row_weights = row_slack / row_total.clamp_min(eps)
    col_weights = col_slack / col_total.clamp_min(eps)
    feasible = (row_total.squeeze(-1) > eps) & (col_total.squeeze(-1) > eps) & (mass > eps)
    filler = row_weights.unsqueeze(-1) * col_weights.unsqueeze(-2) * mass.reshape(*mass.shape, 1, 1)
    return torch.where(feasible.reshape(*feasible.shape, 1, 1), filler, torch.zeros_like(filler))


def _repair_partial_transport_plan(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Project numerical leftovers back into the partial OT feasible set."""
    plan = plan.clamp_min(0.0)
    mass_view = transport_mass.reshape(*transport_mass.shape, 1, 1)

    if torch.all(transport_mass <= eps):
        return torch.zeros_like(plan)

    plan = plan * mass_view / plan.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)

    row_scale = torch.minimum(
        a.unsqueeze(-1) / plan.sum(dim=-1, keepdim=True).clamp_min(eps),
        torch.ones_like(a).unsqueeze(-1),
    )
    plan = plan * row_scale
    col_scale = torch.minimum(
        b.unsqueeze(-2) / plan.sum(dim=-2, keepdim=True).clamp_min(eps),
        torch.ones_like(b).unsqueeze(-2),
    )
    plan = plan * col_scale

    missing = (transport_mass - plan.sum(dim=(-1, -2))).clamp_min(0.0)
    row_slack = (a - plan.sum(dim=-1)).clamp_min(0.0)
    col_slack = (b - plan.sum(dim=-2)).clamp_min(0.0)
    return plan + _slack_outer_plan(row_slack, col_slack, missing, eps)


def require_pot() -> None:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is required for this partial OT backend but is not installed")


def entropic_partial_wasserstein(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    reg: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = EPS,
    transport_mass_ratio: float | torch.Tensor | None = None,
    mass_ratio: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable entropic partial Wasserstein plan.

    This follows the Bregman projection formulation used by POT's
    ``ot.partial.entropic_partial_wasserstein`` and supports arbitrary leading
    batch dimensions.
    """
    _validate_partial_inputs(cost, a, b, reg)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    mass = resolve_partial_transport_mass(
        a,
        b,
        transport_mass=transport_mass,
        transport_mass_ratio=transport_mass_ratio,
        mass_ratio=mass_ratio,
        eps=eps,
    )
    mass_view = mass.reshape(*mass.shape, 1, 1)

    if torch.all(mass <= eps):
        return torch.zeros_like(cost)

    shifted_cost = cost - cost.amin(dim=(-1, -2), keepdim=True)
    kernel = torch.exp(-shifted_cost / float(reg))
    kernel = kernel * mass_view / kernel.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)

    q1 = torch.ones_like(kernel)
    q2 = torch.ones_like(kernel)
    q3 = torch.ones_like(kernel)
    one_row = torch.ones_like(a).unsqueeze(-1)
    one_col = torch.ones_like(b).unsqueeze(-2)

    for iter_idx in range(int(max_iter)):
        previous = kernel

        kernel = kernel * q1
        row_scale = torch.minimum(
            a.unsqueeze(-1) / kernel.sum(dim=-1, keepdim=True).clamp_min(eps),
            one_row,
        )
        row_projected = kernel * row_scale
        q1 = q1 * previous / row_projected.clamp_min(eps)

        previous_row = row_projected
        row_projected = row_projected * q2
        col_scale = torch.minimum(
            b.unsqueeze(-2) / row_projected.sum(dim=-2, keepdim=True).clamp_min(eps),
            one_col,
        )
        col_projected = row_projected * col_scale
        q2 = q2 * previous_row / col_projected.clamp_min(eps)

        previous_col = col_projected
        col_projected = col_projected * q3
        kernel = col_projected * mass_view / col_projected.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
        q3 = q3 * previous_col / kernel.clamp_min(eps)

        if not torch.isfinite(kernel).all():
            raise FloatingPointError(f"partial OT numerical error at iteration {iter_idx}")
        if tol > 0.0 and iter_idx % 10 == 0:
            delta = (kernel - previous).abs().amax()
            if delta < tol:
                break

    return _repair_partial_transport_plan(kernel, a, b, mass, eps=eps)


def _pot_apply_pairwise(
    solver,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    require_pot()
    if cost.dim() == 2:
        return solver(a, b, cost, mass)

    leading_shape = cost.shape[:-2]
    cost_flat = cost.reshape(-1, cost.shape[-2], cost.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    mass_flat = mass.reshape(-1)
    plans = [
        solver(a_flat[idx], b_flat[idx], cost_flat[idx], mass_flat[idx])
        for idx in range(cost_flat.shape[0])
    ]
    return torch.stack(plans, dim=0).reshape(*leading_shape, cost.shape[-2], cost.shape[-1])


def partial_wasserstein_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    nb_dummies: int = 1,
    reg: float = 1.0,
    entropic: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-9,
    eps: float = EPS,
    transport_mass_ratio: float | torch.Tensor | None = None,
    mass_ratio: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """POT partial Wasserstein wrapper.

    ``entropic=False`` calls POT's exact ``partial_wasserstein``. ``entropic=True``
    calls ``entropic_partial_wasserstein`` for a regularized reference solver.
    """
    _validate_partial_inputs(cost, a, b, reg=reg if entropic else 1.0)
    if nb_dummies <= 0:
        raise ValueError("nb_dummies must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    mass = resolve_partial_transport_mass(
        a,
        b,
        transport_mass=transport_mass,
        transport_mass_ratio=transport_mass_ratio,
        mass_ratio=mass_ratio,
        eps=eps,
    )

    def solver(
        pair_a: torch.Tensor,
        pair_b: torch.Tensor,
        pair_cost: torch.Tensor,
        pair_mass: torch.Tensor,
    ) -> torch.Tensor:
        mass_value = float(pair_mass.detach().cpu())
        if mass_value <= eps:
            return torch.zeros_like(pair_cost)
        if entropic:
            plan = ot.partial.entropic_partial_wasserstein(
                pair_a,
                pair_b,
                pair_cost,
                reg=float(reg),
                m=mass_value,
                numItermax=int(max_iter),
                stopThr=float(tol),
                verbose=False,
                log=False,
            )
        else:
            plan = ot.partial.partial_wasserstein(
                pair_a,
                pair_b,
                pair_cost,
                m=mass_value,
                nb_dummies=int(nb_dummies),
                log=False,
            )
        return plan.to(device=pair_cost.device, dtype=pair_cost.dtype)

    return _pot_apply_pairwise(solver, cost, a, b, mass)


def solve_partial_transport(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    *,
    transport_mass_ratio: float | torch.Tensor | None = None,
    mass_ratio: float | torch.Tensor | None = None,
    backend: str = "fast",
    reg: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    nb_dummies: int = 1,
    exact: bool = False,
    dummy_cost_scale: float = 1.0,
    momentum: float = 0.9,
    eps: float = EPS,
) -> torch.Tensor:
    """Dispatch helper for reusable partial OT matching blocks.

    Backends:
        "fast" (default): Augmented balanced Sinkhorn in log-domain.
            ~2-3x faster than "native" with exact POT feasibility.
        "fast_nesterov": Same as "fast" + Nesterov momentum (~15-20 iters).
        "native": Original Bregman projections (slower, ~100 iters).
        "pot": External POT library (exact LP or entropic).
    """
    backend = str(backend).lower()

    if backend in ("fast", "fast_nesterov"):
        mass = resolve_partial_transport_mass(
            a, b,
            transport_mass=transport_mass,
            transport_mass_ratio=transport_mass_ratio,
            mass_ratio=mass_ratio,
            eps=eps,
        )
        if backend == "fast_nesterov":
            return fast_partial_wasserstein_nesterov(
                cost, a, b,
                transport_mass=mass,
                reg=reg,
                max_iter=max_iter,
                tol=tol,
                dummy_cost_scale=dummy_cost_scale,
                momentum=momentum,
                eps=eps,
            )
        return fast_partial_wasserstein(
            cost, a, b,
            transport_mass=mass,
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            dummy_cost_scale=dummy_cost_scale,
            eps=eps,
        )

    if backend == "native":
        if exact:
            raise ValueError("exact=True is only supported by backend='pot'")
        return entropic_partial_wasserstein(
            cost,
            a,
            b,
            transport_mass=transport_mass,
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            transport_mass_ratio=transport_mass_ratio,
            mass_ratio=mass_ratio,
        )
    if backend == "pot":
        return partial_wasserstein_pot(
            cost,
            a,
            b,
            transport_mass=transport_mass,
            nb_dummies=nb_dummies,
            reg=reg,
            entropic=not exact,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            transport_mass_ratio=transport_mass_ratio,
            mass_ratio=mass_ratio,
        )
    raise ValueError(f"Unsupported partial OT backend: {backend}")


def compute_partial_transport_cost(plan: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    if plan.shape != cost.shape:
        raise ValueError(f"plan must have shape {tuple(cost.shape)}, got {tuple(plan.shape)}")
    return (plan * cost).sum(dim=(-1, -2))


def compute_partial_transported_mass(plan: torch.Tensor) -> torch.Tensor:
    return plan.sum(dim=(-1, -2))
