"""Poincare-ball operations with geoopt-first, pure-PyTorch fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import geoopt
except ImportError:  # pragma: no cover - exercised only when geoopt is unavailable
    geoopt = None


EPS = 1e-8
BALL_EPS = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
    torch.float32: 4e-3,
    torch.float64: 1e-5,
}


def _as_curvature_tensor(curvature: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(curvature):
        return curvature.to(device=reference.device, dtype=reference.dtype).clamp_min(EPS)
    return torch.tensor(float(curvature), device=reference.device, dtype=reference.dtype).clamp_min(EPS)


def _default_eps(dtype: torch.dtype) -> float:
    return BALL_EPS.get(dtype, 1e-5)


def safe_arctanh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.atanh(x.clamp(min=-1.0 + eps, max=1.0 - eps))


@dataclass
class NativePoincareBall:
    """Pure-PyTorch Poincare-ball implementation used as fallback/reference."""

    c: torch.Tensor | float = 1.0

    backend: str = "native"

    def curvature(self, reference: torch.Tensor) -> torch.Tensor:
        return _as_curvature_tensor(self.c, reference)

    def radius(self, reference: torch.Tensor) -> torch.Tensor:
        return self.curvature(reference).rsqrt()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        radius = self.radius(x)
        max_norm = (radius - _default_eps(x.dtype)).clamp_min(EPS)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        y = self.project(y)
        c = self.curvature(x + y)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        denominator = 1.0 + 2.0 * c * xy + (c * c) * x2 * y2
        return self.project(numerator / denominator.clamp_min(EPS))

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        c = self.curvature(tangent)
        sqrt_c = torch.sqrt(c)
        tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.tanh(sqrt_c * tangent_norm) / (sqrt_c * tangent_norm)
        return self.project(scale * tangent)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        c = self.curvature(x)
        sqrt_c = torch.sqrt(c)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
        factor = safe_arctanh(sqrt_c * norm, eps=_default_eps(x.dtype)) / (sqrt_c * norm)
        return factor * x

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        delta = self.mobius_add(-x, y)
        c = self.curvature(delta)
        sqrt_c = torch.sqrt(c)
        delta_norm = delta.norm(dim=-1).clamp_min(EPS)
        return 2.0 * safe_arctanh(sqrt_c * delta_norm, eps=_default_eps(delta.dtype)) / sqrt_c


class GeooptPoincareBallAdapter(nn.Module):
    """Adapter exposing the same minimal API as the native implementation."""

    backend = "geoopt"

    def __init__(self, manifold: "geoopt.PoincareBall") -> None:
        super().__init__()
        if geoopt is None:  # pragma: no cover - defensive
            raise ImportError("geoopt is not installed")
        self.manifold = manifold

    @property
    def c(self) -> torch.Tensor:
        return self.manifold.c

    def curvature(self, reference: torch.Tensor) -> torch.Tensor:
        return self.c.to(device=reference.device, dtype=reference.dtype).clamp_min(EPS)

    def radius(self, reference: torch.Tensor) -> torch.Tensor:
        return self.curvature(reference).rsqrt()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.projx(x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.mobius_add(x, y)

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap0(tangent)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap0(x)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.dist(x, y)


PoincareBall = NativePoincareBall | GeooptPoincareBallAdapter


def resolve_hyperbolic_backend(backend: str = "auto") -> str:
    backend = str(backend).lower()
    if backend == "auto":
        return "geoopt" if geoopt is not None else "native"
    if backend == "geoopt":
        if geoopt is None:
            raise ImportError("geoopt backend requested but geoopt is not installed")
        return "geoopt"
    if backend == "native":
        return "native"
    raise ValueError(f"Unsupported hyperbolic backend: {backend}")


def get_ball(
    curvature: torch.Tensor | float = 1.0,
    *,
    backend: str = "auto",
    learnable: bool = False,
) -> PoincareBall:
    resolved = resolve_hyperbolic_backend(backend)
    if resolved == "geoopt":
        curvature_value = float(curvature.detach().item()) if torch.is_tensor(curvature) else float(curvature)
        return GeooptPoincareBallAdapter(geoopt.PoincareBall(c=curvature_value, learnable=learnable))
    return NativePoincareBall(c=curvature)


def project_to_ball_coordinates(x: torch.Tensor, ball: PoincareBall) -> torch.Tensor:
    return ball.project(x)


def safe_project_to_ball(x: torch.Tensor, ball: PoincareBall) -> torch.Tensor:
    return project_to_ball_coordinates(ball.expmap0(x), ball)


def hyperbolic_distance_matrix(
    z_query: torch.Tensor,
    z_class: torch.Tensor,
    ball: PoincareBall,
) -> torch.Tensor:
    """Return squared hyperbolic distances with broadcastable leading dims."""

    z_query = project_to_ball_coordinates(z_query, ball)
    z_class = project_to_ball_coordinates(z_class, ball)
    return ball.dist(z_query.unsqueeze(-2), z_class.unsqueeze(-3)).pow(2)


def frechet_mean_poincare(
    z: torch.Tensor,
    ball: PoincareBall,
) -> torch.Tensor:
    """Fast tangent-space approximation of the Fréchet mean."""

    tangent_mean = ball.logmap0(z).mean(dim=-2)
    return safe_project_to_ball(tangent_mean, ball)


def hyperbolic_variance(
    z: torch.Tensor,
    mu: torch.Tensor,
    ball: PoincareBall,
) -> torch.Tensor:
    dists = ball.dist(project_to_ball_coordinates(z, ball), mu.unsqueeze(-2))
    return dists.pow(2).mean(dim=-1)
