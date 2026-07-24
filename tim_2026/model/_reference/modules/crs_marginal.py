"""Cross-referenced selective support marginals for J-ECOT-M2.

CRS-M2 keeps the UOT objective and score unchanged.  This module only shapes
the support-side marginal b = rho * pi, with sum_l pi_l = 1.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency in local shells.
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:  # pragma: no cover
    MAMBA_IMPORT_ERROR = None


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError("inverse sigmoid expects a value in (0, 1)")
    value = float(value)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


class SimpleBidirectionalScan(nn.Module):
    """Small deterministic 2D scan used when mamba_ssm is not available.

    The block scans H+, H-, W+, and W- with depthwise Conv1d filters and a
    gated residual, then fuses the four directional contexts.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )
        self.gate = nn.Linear(dim, dim)
        self.mix = nn.Linear(dim * 4, dim)
        self.output_norm = nn.LayerNorm(dim)

    def _scan_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        conv = self.depthwise(sequence.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(sequence))
        return sequence + gate * conv

    def _scan_width(self, grid: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        batch_size, height, width, dim = grid.shape
        sequence = grid.reshape(batch_size * height, width, dim)
        if reverse:
            sequence = torch.flip(sequence, dims=[1])
        scanned = self._scan_sequence(sequence)
        if reverse:
            scanned = torch.flip(scanned, dims=[1])
        return scanned.reshape(batch_size, height, width, dim)

    def _scan_height(self, grid: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        batch_size, height, width, dim = grid.shape
        sequence = grid.permute(0, 2, 1, 3).reshape(batch_size * width, height, dim)
        if reverse:
            sequence = torch.flip(sequence, dims=[1])
        scanned = self._scan_sequence(sequence)
        if reverse:
            scanned = torch.flip(scanned, dims=[1])
        return scanned.reshape(batch_size, width, height, dim).permute(0, 2, 1, 3)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() != 4:
            raise ValueError(f"grid must have shape (Batch, H, W, Dim), got {tuple(grid.shape)}")
        h_forward = self._scan_height(grid, reverse=False)
        h_backward = self._scan_height(grid, reverse=True)
        w_forward = self._scan_width(grid, reverse=False)
        w_backward = self._scan_width(grid, reverse=True)
        mixed = self.mix(torch.cat([h_forward, h_backward, w_forward, w_backward], dim=-1))
        return self.output_norm(grid + mixed)


class Mamba2DScan(nn.Module):
    """Four-direction 2D token scan backed by mamba_ssm.Mamba."""

    def __init__(self, dim: int, d_state: int = 8, d_conv: int = 4, expand: int = 1) -> None:
        super().__init__()
        if Mamba is None:  # pragma: no cover - depends on optional package.
            raise ImportError("Mamba2DScan requires mamba_ssm") from MAMBA_IMPORT_ERROR
        self.h_forward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.h_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.w_forward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.w_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mix = nn.Linear(dim * 4, dim)
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _run_width(module: nn.Module, grid: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        batch_size, height, width, dim = grid.shape
        sequence = grid.reshape(batch_size * height, width, dim)
        if reverse:
            sequence = torch.flip(sequence, dims=[1]).contiguous()
        out = module(sequence)
        if reverse:
            out = torch.flip(out, dims=[1]).contiguous()
        return out.reshape(batch_size, height, width, dim)

    @staticmethod
    def _run_height(module: nn.Module, grid: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        batch_size, height, width, dim = grid.shape
        sequence = grid.permute(0, 2, 1, 3).reshape(batch_size * width, height, dim)
        if reverse:
            sequence = torch.flip(sequence, dims=[1]).contiguous()
        out = module(sequence)
        if reverse:
            out = torch.flip(out, dims=[1]).contiguous()
        return out.reshape(batch_size, width, height, dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        h_forward = self._run_height(self.h_forward, grid, reverse=False)
        h_backward = self._run_height(self.h_backward, grid, reverse=True)
        w_forward = self._run_width(self.w_forward, grid, reverse=False)
        w_backward = self._run_width(self.w_backward, grid, reverse=True)
        mixed = self.mix(torch.cat([h_forward, h_backward, w_forward, w_backward], dim=-1))
        return self.output_norm(grid + mixed)


class CrossReferencedSelectiveMarginal(nn.Module):
    """Build budget-preserving CRS marginals for support tokens.

    Cross-reference branch:
        r_l = relu(cosine(support_l, mean(query))) + eps
        p_cr = r / sum_l r_l

    SSM branch:
        h = scan_2d(LN(support))
        p_ssm = softmax(Linear(LN(h)) / tau_ssm)

    Mixture:
        pi = (1 - eta) * uniform + eta * p_rel
        b = rho * pi
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        use_cross_ref: bool = True,
        use_ssm: bool = True,
        eta_init: float = 0.30,
        lambda_cr_init: float = 0.50,
        tau_ssm_init: float = 0.70,
        entropy_reg: float = 0.0,
        side: str = "support",
        ssm_type: str = "auto",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if not 0.0 < float(eta_init) < 1.0:
            raise ValueError("eta_init must be in (0, 1)")
        if not 0.0 < float(lambda_cr_init) < 1.0:
            raise ValueError("lambda_cr_init must be in (0, 1)")
        if float(tau_ssm_init) <= 0.0:
            raise ValueError("tau_ssm_init must be positive")
        if float(entropy_reg) < 0.0:
            raise ValueError("entropy_reg must be non-negative")
        side = str(side).strip().lower()
        if side not in {"support", "both"}:
            raise ValueError(f"Unsupported CRS marginal side: {side}")
        ssm_type = str(ssm_type).strip().lower()
        if ssm_type not in {"auto", "simple", "mamba"}:
            raise ValueError(f"Unsupported CRS SSM type: {ssm_type}")

        self.embed_dim = int(embed_dim)
        self.use_cross_ref = bool(use_cross_ref)
        self.use_ssm = bool(use_ssm)
        self.entropy_reg = float(entropy_reg)
        self.side = side
        self.ssm_type = ssm_type
        self.eps = float(eps)

        self.raw_eta = nn.Parameter(torch.tensor(_inverse_sigmoid(float(eta_init)), dtype=torch.float32))
        self.raw_lambda_cr = nn.Parameter(torch.tensor(_inverse_sigmoid(float(lambda_cr_init)), dtype=torch.float32))
        self.raw_tau_ssm = nn.Parameter(
            torch.tensor(_inverse_softplus(max(float(tau_ssm_init) - 1e-4, 1e-6)), dtype=torch.float32)
        )

        self.ssm_input_norm = nn.LayerNorm(embed_dim)
        self.ssm_output_norm = nn.LayerNorm(embed_dim)
        self.ssm_score = nn.Linear(embed_dim, 1)
        self.ssm_backend_name = "disabled"
        if self.use_ssm:
            self.ssm_scan = self._build_scan_backend(embed_dim, ssm_type)
        else:
            self.ssm_scan = None

    def _build_scan_backend(self, embed_dim: int, ssm_type: str) -> nn.Module:
        if ssm_type in {"auto", "mamba"} and Mamba is not None:  # pragma: no cover - optional dependency.
            try:
                self.ssm_backend_name = "mamba_ssm"
                return Mamba2DScan(embed_dim)
            except Exception:
                if ssm_type == "mamba":
                    self.ssm_backend_name = "simple"
                    return SimpleBidirectionalScan(embed_dim)
        self.ssm_backend_name = "simple"
        return SimpleBidirectionalScan(embed_dim)

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def lambda_cr(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_lambda_cr)

    @property
    def tau_ssm(self) -> torch.Tensor:
        return F.softplus(self.raw_tau_ssm) + 1e-4

    def _resolve_hw(self, token_count: int, spatial_hw: tuple[int, int] | None) -> tuple[int, int]:
        if spatial_hw is not None:
            height, width = int(spatial_hw[0]), int(spatial_hw[1])
            if height > 0 and width > 0 and height * width == token_count:
                return height, width
        return 1, int(token_count)

    def _rho_view(
        self,
        rho: float | torch.Tensor,
        *,
        batch_size: int,
        num_query: int,
        way_num: int,
        shot_num: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rho_tensor = torch.as_tensor(rho, device=reference.device, dtype=reference.dtype)
        if rho_tensor.dim() == 0:
            return rho_tensor.view(1, 1, 1, 1, 1)
        if rho_tensor.dim() == 3 and tuple(rho_tensor.shape) == (num_query, way_num, shot_num):
            rho_tensor = rho_tensor.unsqueeze(0)
        if rho_tensor.dim() == 4 and tuple(rho_tensor.shape) == (batch_size, num_query, way_num, shot_num):
            return rho_tensor.unsqueeze(-1)
        if rho_tensor.dim() == 5 and tuple(rho_tensor.shape) == (batch_size, num_query, way_num, shot_num, 1):
            return rho_tensor
        raise ValueError(
            "rho must be scalar or shaped (B, Nq, Way, Shot), "
            f"got {tuple(rho_tensor.shape)}"
        )

    def _support_cross_reference(
        self,
        support_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        support_norm = F.normalize(support_tokens, dim=-1, eps=self.eps)
        query_norm = F.normalize(query_tokens, dim=-1, eps=self.eps)
        q_bar = F.normalize(query_norm.mean(dim=2), dim=-1, eps=self.eps)
        relevance = torch.einsum("bckld,bnd->bnckl", support_norm, q_bar)
        relevance = F.relu(relevance) + self.eps
        return relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _query_cross_reference(
        self,
        support_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        support_norm = F.normalize(support_tokens, dim=-1, eps=self.eps)
        query_norm = F.normalize(query_tokens, dim=-1, eps=self.eps)
        support_bar = F.normalize(support_norm.mean(dim=3), dim=-1, eps=self.eps)
        relevance = torch.einsum("bntd,bckd->bnckt", query_norm, support_bar)
        relevance = F.relu(relevance) + self.eps
        return relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _ssm_distribution(
        self,
        tokens: torch.Tensor,
        *,
        token_count: int,
        spatial_hw: tuple[int, int] | None,
    ) -> torch.Tensor:
        if self.ssm_scan is None:
            raise RuntimeError("CRS SSM branch is disabled")
        height, width = self._resolve_hw(token_count, spatial_hw)
        batch_flat = tokens.shape[0]
        grid = tokens.reshape(batch_flat, height, width, self.embed_dim)
        grid = self.ssm_input_norm(grid)
        contextual = self.ssm_scan(grid)
        logits = self.ssm_score(self.ssm_output_norm(contextual)).squeeze(-1)
        logits = logits.reshape(batch_flat, token_count)
        return torch.softmax(logits / self.tau_ssm.to(device=logits.device, dtype=logits.dtype), dim=-1)

    def _support_ssm(
        self,
        support_tokens: torch.Tensor,
        *,
        num_query: int,
        spatial_hw: tuple[int, int] | None,
    ) -> torch.Tensor:
        batch_size, way_num, shot_num, support_len, embed_dim = support_tokens.shape
        flat = support_tokens.reshape(batch_size * way_num * shot_num, support_len, embed_dim)
        probs = self._ssm_distribution(flat, token_count=support_len, spatial_hw=spatial_hw)
        probs = probs.reshape(batch_size, way_num, shot_num, support_len)
        return probs.unsqueeze(1).expand(-1, num_query, -1, -1, -1)

    def _query_ssm(
        self,
        query_tokens: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None,
    ) -> torch.Tensor:
        batch_size, num_query, query_len, embed_dim = query_tokens.shape
        flat = query_tokens.reshape(batch_size * num_query, query_len, embed_dim)
        probs = self._ssm_distribution(flat, token_count=query_len, spatial_hw=spatial_hw)
        probs = probs.reshape(batch_size, num_query, query_len)
        return probs[:, :, None, None, :].expand(-1, -1, way_num, shot_num, -1)

    def _mix_distribution(
        self,
        uniform: torch.Tensor,
        *,
        p_cross_ref: torch.Tensor | None,
        p_ssm: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_cross_ref and p_cross_ref is not None and self.use_ssm and p_ssm is not None:
            lam = self.lambda_cr.to(device=uniform.device, dtype=uniform.dtype)
            p_rel = (1.0 - lam) * p_ssm + lam * p_cross_ref
        elif self.use_cross_ref and p_cross_ref is not None:
            p_rel = p_cross_ref
        elif self.use_ssm and p_ssm is not None:
            p_rel = p_ssm
        else:
            p_rel = uniform

        eta = self.eta.to(device=uniform.device, dtype=uniform.dtype)
        pi = (1.0 - eta) * uniform + eta * p_rel
        pi = pi.clamp_min(self.eps)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if bool((eta == 0).detach().cpu().item()):
            pi = uniform
        return pi

    def _diagnostics(self, pi: torch.Tensor, uniform: torch.Tensor) -> dict[str, torch.Tensor]:
        entropy = -(pi * pi.clamp_min(self.eps).log()).sum(dim=-1)
        peak_ratio = pi.max(dim=-1).values / pi.mean(dim=-1).clamp_min(self.eps)
        uniform_kl = (pi * (pi.clamp_min(self.eps) / uniform.clamp_min(self.eps)).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(float(max(pi.shape[-1], 2)))
        entropy_loss = torch.relu(pi.new_tensor(0.35) - normalized_entropy).pow(2).mean()
        aux_loss = pi.new_tensor(float(self.entropy_reg)) * entropy_loss
        return {
            "crs_entropy": entropy,
            "crs_peak_ratio": peak_ratio,
            "crs_uniform_kl": uniform_kl,
            "crs_entropy_loss": entropy_loss,
            "crs_aux_loss": aux_loss,
        }

    def forward(
        self,
        support_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        rho: float | torch.Tensor,
        spatial_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if support_tokens.dim() != 5:
            raise ValueError(
                "support_tokens must have shape (B, Way, Shot, Ls, D), "
                f"got {tuple(support_tokens.shape)}"
            )
        if query_tokens.dim() != 4:
            raise ValueError(
                "query_tokens must have shape (B, Nq, Lq, D), "
                f"got {tuple(query_tokens.shape)}"
            )

        batch_size, way_num, shot_num, support_len, support_dim = support_tokens.shape
        query_batch, num_query, query_len, query_dim = query_tokens.shape
        if query_batch != batch_size:
            raise ValueError(f"support/query batch mismatch: {batch_size} vs {query_batch}")
        if query_dim != support_dim or support_dim != self.embed_dim:
            raise ValueError(f"token dim mismatch: support={support_dim}, query={query_dim}, module={self.embed_dim}")

        uniform = support_tokens.new_full(
            (batch_size, num_query, way_num, shot_num, support_len),
            1.0 / float(support_len),
        )
        p_cross_ref = self._support_cross_reference(support_tokens, query_tokens) if self.use_cross_ref else None
        p_ssm = (
            self._support_ssm(support_tokens, num_query=num_query, spatial_hw=spatial_hw)
            if self.use_ssm
            else None
        )
        pi = self._mix_distribution(uniform, p_cross_ref=p_cross_ref, p_ssm=p_ssm)

        rho_view = self._rho_view(
            rho,
            batch_size=batch_size,
            num_query=num_query,
            way_num=way_num,
            shot_num=shot_num,
            reference=support_tokens,
        )
        b = rho_view * pi
        expected = rho_view.squeeze(-1).expand_as(b.sum(dim=-1))
        b = b * (rho_view / b.sum(dim=-1, keepdim=True).clamp_min(self.eps))
        torch._assert(torch.isfinite(b).all(), "CRS support marginal contains non-finite values")
        torch._assert((b > 0.0).all(), "CRS support marginal must be positive")
        torch._assert(
            torch.isclose(b.sum(dim=-1), expected, atol=1e-5, rtol=1e-4).all(),
            "CRS support marginal must preserve the rho budget",
        )

        p_cross_ref_out = p_cross_ref if p_cross_ref is not None else uniform
        p_ssm_out = p_ssm if p_ssm is not None else uniform
        aux: dict[str, torch.Tensor | Any] = {
            "crs_pi": pi,
            "crs_b": b,
            "crs_support_marginal": b,
            "crs_p_cross_ref": p_cross_ref_out,
            "crs_p_ssm": p_ssm_out,
            "crs_eta": self.eta.to(device=b.device, dtype=b.dtype),
            "crs_lambda_cr": self.lambda_cr.to(device=b.device, dtype=b.dtype),
            "crs_tau_ssm": self.tau_ssm.to(device=b.device, dtype=b.dtype),
        }
        aux.update(self._diagnostics(pi, uniform))

        if self.side == "both":
            query_uniform = query_tokens.new_full(
                (batch_size, num_query, way_num, shot_num, query_len),
                1.0 / float(query_len),
            )
            q_cross_ref = self._query_cross_reference(support_tokens, query_tokens) if self.use_cross_ref else None
            q_ssm = (
                self._query_ssm(
                    query_tokens,
                    way_num=way_num,
                    shot_num=shot_num,
                    spatial_hw=spatial_hw,
                )
                if self.use_ssm
                else None
            )
            query_pi = self._mix_distribution(query_uniform, p_cross_ref=q_cross_ref, p_ssm=q_ssm)
            query_a = rho_view * query_pi
            query_a = query_a * (rho_view / query_a.sum(dim=-1, keepdim=True).clamp_min(self.eps))
            aux["crs_query_pi"] = query_pi
            aux["crs_query_marginal"] = query_a
            aux["crs_query_p_cross_ref"] = q_cross_ref if q_cross_ref is not None else query_uniform
            aux["crs_query_p_ssm"] = q_ssm if q_ssm is not None else query_uniform

        return b, aux  # type: ignore[return-value]


__all__ = ["CrossReferencedSelectiveMarginal", "SimpleBidirectionalScan"]
