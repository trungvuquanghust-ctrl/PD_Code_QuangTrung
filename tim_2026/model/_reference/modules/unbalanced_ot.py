"""Balanced and unbalanced Sinkhorn solvers with native and POT backends."""

from __future__ import annotations

import warnings

import torch

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is unavailable
    ot = None


EPS = 1e-8


def _validate_transport_inputs(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float) -> None:
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if a.shape != cost.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError(f"b must have shape {tuple(cost.shape[:-2] + (cost.shape[-1],))}, got {tuple(b.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("a and b must be non-negative")


def validate_balanced_marginals(
    a: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-5,
) -> None:
    """Validate that balanced OT marginals have equal total mass."""
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(f"a and b leading shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    residual = (a.sum(dim=-1) - b.sum(dim=-1)).abs()
    if torch.any(residual > tol):
        raise ValueError("balanced Sinkhorn requires equal total mass in a and b")


def balanced_transport_plan_residuals(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-problem row/column residuals for a balanced OT plan."""
    if plan.shape != a.shape + (b.shape[-1],):
        raise ValueError(f"plan must have shape {tuple(a.shape + (b.shape[-1],))}, got {tuple(plan.shape)}")
    if b.shape != plan.shape[:-2] + (plan.shape[-1],):
        raise ValueError(f"b must have shape {tuple(plan.shape[:-2] + (plan.shape[-1],))}, got {tuple(b.shape)}")
    row_residual = (plan.sum(dim=-1) - a).abs().amax(dim=-1)
    column_residual = (plan.sum(dim=-2) - b).abs().amax(dim=-1)
    mass_residual = (plan.sum(dim=(-1, -2)) - a.sum(dim=-1)).abs()
    return {
        "row_residual": row_residual,
        "column_residual": column_residual,
        "mass_residual": mass_residual,
    }


def validate_balanced_transport_plan(
    plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Validate row and column marginals for a balanced Sinkhorn plan."""
    residuals = balanced_transport_plan_residuals(plan, a, b)
    if torch.any(residuals["row_residual"] > tol):
        raise FloatingPointError("balanced Sinkhorn row marginal residual exceeds tolerance")
    if torch.any(residuals["column_residual"] > tol):
        raise FloatingPointError("balanced Sinkhorn column marginal residual exceeds tolerance")
    if torch.any(residuals["mass_residual"] > tol):
        raise FloatingPointError("balanced Sinkhorn total mass residual exceeds tolerance")
    return residuals


def resolve_ot_backend(backend: str = "native") -> str:
    backend = str(backend).lower()
    if backend == "auto":
        return "native"
    if backend in {"native", "pot"}:
        if backend == "pot" and ot is None:
            raise ImportError("POT backend requested but POT is not installed")
        return backend
    raise ValueError(f"Unsupported OT backend: {backend}")


def require_pot() -> None:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is required for this solver path but is not installed")


def sinkhorn_balanced_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Log-domain entropic Sinkhorn solver for balanced OT.

    Solves `P = diag(u) exp(-C / eps) diag(v)` with fixed row
    marginal `a` and column marginal `b`.
    """
    _validate_transport_inputs(cost, a, b, eps)
    validate_balanced_marginals(a, b, tol=max(float(tol) * 10.0, 1e-5))
    log_a = torch.log(a.clamp_min(EPS))
    log_b = torch.log(b.clamp_min(EPS))
    log_kernel = -cost / float(eps)

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

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


def sinkhorn_unbalanced_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Log-domain Sinkhorn solver for KL-relaxed unbalanced OT.

    This matches POT's entropy-regularized UOT scaling
    `ot.unbalanced.sinkhorn_unbalanced(..., reg_type="entropy")`:
    `K = exp(-C / eps)` and scaling exponents
    `rho = tau / (tau + eps)`.
    """
    _validate_transport_inputs(cost, a, b, eps)
    if tau_q <= 0.0 or tau_c <= 0.0:
        raise ValueError("tau_q and tau_c must be positive")

    log_a = torch.log(a.clamp_min(EPS))
    log_b = torch.log(b.clamp_min(EPS))
    log_kernel = -cost / float(eps)
    rho_q = float(tau_q) / (float(tau_q) + float(eps))
    rho_c = float(tau_c) / (float(tau_c) + float(eps))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        prev_log_v = log_v
        log_u = rho_q * (log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1))
        log_v = rho_c * (log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2))
        delta = torch.maximum(
            (log_u - prev_log_u).abs().amax(),
            (log_v - prev_log_v).abs().amax(),
        )
        if tol > 0.0 and delta < tol:
            break

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


def sinkhorn_unbalanced_log_adaptive(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: torch.Tensor,
    tau_c: torch.Tensor,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Log-domain UOT with coordinate-wise marginal relaxation.

    ``tau_q`` and ``tau_c`` weight the generalized KL penalty for each query
    and support token. Larger values make the corresponding marginal harder to
    destroy; smaller values let the solver discard unreliable local evidence.
    """
    _validate_transport_inputs(cost, a, b, eps)
    if tau_q.shape != a.shape:
        raise ValueError(f"tau_q must have shape {tuple(a.shape)}, got {tuple(tau_q.shape)}")
    if tau_c.shape != b.shape:
        raise ValueError(f"tau_c must have shape {tuple(b.shape)}, got {tuple(tau_c.shape)}")
    tau_q = tau_q.to(device=cost.device, dtype=cost.dtype)
    tau_c = tau_c.to(device=cost.device, dtype=cost.dtype)
    if torch.any(tau_q <= 0.0) or torch.any(tau_c <= 0.0):
        raise ValueError("adaptive tau_q and tau_c must be positive")

    log_a = torch.log(a.clamp_min(EPS))
    log_b = torch.log(b.clamp_min(EPS))
    log_kernel = -cost / float(eps)
    rho_q = tau_q / (tau_q + float(eps))
    rho_c = tau_c / (tau_c + float(eps))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        prev_log_v = log_v
        log_u = rho_q * (
            log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        )
        log_v = rho_c * (
            log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        )
        delta = torch.maximum(
            (log_u - prev_log_u).abs().amax(),
            (log_v - prev_log_v).abs().amax(),
        )
        if tol > 0.0 and delta < tol:
            break

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


def _pot_apply_pairwise(
    solver,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is not installed")
    if cost.dim() == 2:
        return solver(a, b, cost)

    leading_shape = cost.shape[:-2]
    cost_flat = cost.reshape(-1, cost.shape[-2], cost.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    plans = [solver(a_flat[idx], b_flat[idx], cost_flat[idx]) for idx in range(cost_flat.shape[0])]
    return torch.stack(plans, dim=0).reshape(*leading_shape, cost.shape[-2], cost.shape[-1])


def sinkhorn_balanced_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    validate_balanced_marginals(a, b, tol=max(float(tol) * 10.0, 1e-5))
    require_pot()

    def solver(pair_a: torch.Tensor, pair_b: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
        return ot.sinkhorn(
            pair_a,
            pair_b,
            pair_cost,
            reg=float(eps),
            method="sinkhorn",
            numItermax=int(max_iter),
            stopThr=float(tol),
            verbose=False,
            warn=False,
        )

    return _pot_apply_pairwise(solver, cost, a, b)


def sinkhorn_unbalanced_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    require_pot()
    if tau_q <= 0.0 or tau_c <= 0.0:
        raise ValueError("tau_q and tau_c must be positive")

    def solver(pair_a: torch.Tensor, pair_b: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="If reg_type = entropy, then the matrix c is overwritten by the one matrix.",
            )
            return ot.unbalanced.sinkhorn_unbalanced(
                pair_a,
                pair_b,
                pair_cost,
                reg=float(eps),
                reg_m=(float(tau_q), float(tau_c)),
                method="sinkhorn",
                reg_type="entropy",
                numItermax=int(max_iter),
                stopThr=float(tol),
                verbose=False,
                log=False,
            )

    return _pot_apply_pairwise(solver, cost, a, b)


def barycenter_balanced_pot(
    histograms: torch.Tensor,
    cost: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """POT fixed-support entropic Wasserstein barycenter.

    Args:
        histograms: `(NumMeasures, SupportSize)` distributions.
        cost: `(SupportSize, SupportSize)` ground cost.
        weights: `(NumMeasures,)` barycentric weights.
    """
    if histograms.dim() != 2:
        raise ValueError(f"histograms must have shape (NumMeasures, SupportSize), got {tuple(histograms.shape)}")
    if cost.shape != (histograms.shape[1], histograms.shape[1]):
        raise ValueError(
            "cost must have shape (SupportSize, SupportSize), "
            f"got {tuple(cost.shape)} for support size {histograms.shape[1]}"
        )
    if weights.shape != (histograms.shape[0],):
        raise ValueError(f"weights must have shape ({histograms.shape[0]},), got {tuple(weights.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    require_pot()
    kwargs = dict(
        reg=float(eps),
        weights=weights,
        method=str(method),
        numItermax=int(max_iter),
        stopThr=float(tol),
        verbose=False,
        log=False,
    )
    try:
        return ot.bregman.barycenter(histograms.transpose(0, 1), cost, warn=False, **kwargs)
    except TypeError:
        return ot.bregman.barycenter(histograms.transpose(0, 1), cost, **kwargs)


def barycenter_unbalanced_pot(
    histograms: torch.Tensor,
    cost: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    reg_m: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """POT fixed-support entropic unbalanced Wasserstein barycenter."""
    if histograms.dim() != 2:
        raise ValueError(f"histograms must have shape (NumMeasures, SupportSize), got {tuple(histograms.shape)}")
    if cost.shape != (histograms.shape[1], histograms.shape[1]):
        raise ValueError(
            "cost must have shape (SupportSize, SupportSize), "
            f"got {tuple(cost.shape)} for support size {histograms.shape[1]}"
        )
    if weights.shape != (histograms.shape[0],):
        raise ValueError(f"weights must have shape ({histograms.shape[0]},), got {tuple(weights.shape)}")
    if eps <= 0.0 or reg_m <= 0.0:
        raise ValueError("eps and reg_m must be positive")

    require_pot()
    solver = getattr(ot, "barycenter_unbalanced", None)
    if solver is None:
        solver = ot.unbalanced.barycenter_unbalanced
    return solver(
        histograms.transpose(0, 1),
        cost,
        float(eps),
        float(reg_m),
        method=str(method),
        weights=weights,
        numItermax=int(max_iter),
        stopThr=float(tol),
        verbose=False,
        log=False,
    )


def compute_transport_cost(plan: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    return (plan * cost).sum(dim=(-1, -2))


def compute_transported_mass(plan: torch.Tensor) -> torch.Tensor:
    return plan.sum(dim=(-1, -2))


def generalized_kl_divergence(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Generalized KL divergence used by KL-relaxed UOT marginals."""
    target_safe = target.clamp_min(eps)
    return (
        source * (torch.log(source.clamp_min(eps)) - torch.log(target_safe))
        - source
        + target
    ).sum(dim=-1)


def compute_unbalanced_classification_energy(
    plan: torch.Tensor,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float = EPS,
) -> torch.Tensor:
    """KL-relaxed UOT classification energy evaluated at a Sinkhorn plan.

    This intentionally excludes the entropy regularization term used to compute
    the plan, but includes marginal KL penalties so dropped mass is scored.
    """
    if tau_q <= 0.0 or tau_c <= 0.0:
        raise ValueError("tau_q and tau_c must be positive")
    _validate_transport_inputs(cost, a, b, eps=1.0)
    if plan.shape != cost.shape:
        raise ValueError(f"plan must have shape {tuple(cost.shape)}, got {tuple(plan.shape)}")

    row_mass = plan.sum(dim=-1)
    col_mass = plan.sum(dim=-2)
    transport_cost = compute_transport_cost(plan, cost)
    source_kl = generalized_kl_divergence(row_mass, a, eps=eps)
    target_kl = generalized_kl_divergence(col_mass, b, eps=eps)
    return transport_cost + float(tau_q) * source_kl + float(tau_c) * target_kl


def compute_unbalanced_total_objective_entropy(
    plan: torch.Tensor,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float | None = None,
    entropy_reg: float | None = None,
    numerical_eps: float = EPS,
) -> torch.Tensor:
    """Full entropy-regularized UOT objective for ``reg_type='entropy'``.

    Returns ``<P,C> + tau_q KL(P1,a) + tau_c KL(P^T1,b)
    + eps * sum(P log(P) - P)``. ``entropy_reg`` is accepted as a
    descriptive alias for the entropic regularization coefficient.
    """
    if eps is not None and entropy_reg is not None:
        raise ValueError("Only one of eps and entropy_reg may be set")
    entropy_weight = eps if eps is not None else entropy_reg
    if entropy_weight is None:
        raise ValueError("eps or entropy_reg must be provided")
    if entropy_weight <= 0.0:
        raise ValueError("eps and entropy_reg must be positive")
    classification_energy = compute_unbalanced_classification_energy(
        plan,
        cost,
        a,
        b,
        tau_q=tau_q,
        tau_c=tau_c,
        eps=numerical_eps,
    )
    entropy = (plan * torch.log(plan.clamp_min(numerical_eps)) - plan).sum(dim=(-1, -2))
    return classification_energy + float(entropy_weight) * entropy


def compute_unbalanced_transport_objective(
    plan: torch.Tensor,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float = EPS,
) -> torch.Tensor:
    """Compatibility alias for :func:`compute_unbalanced_classification_energy`."""
    return compute_unbalanced_classification_energy(
        plan,
        cost,
        a,
        b,
        tau_q=tau_q,
        tau_c=tau_c,
        eps=eps,
    )
