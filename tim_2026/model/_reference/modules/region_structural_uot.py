"""Region-structural UOT guidance for few-shot scalogram matching.

This module builds a coarse feature-space region plan, then uses that plan as a
soft structural prior on the fine-token UOT cost.  It does not use image masks or
brightness thresholds; all signals come from projected backbone tokens and the
token-grid geometry.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.modules.fgw_uot_solver import fgw_uot_solve


def parse_region_grid_size(value: str | int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = int(value)
    elif isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        if "x" in text:
            parts = text.split("x")
            if len(parts) != 2:
                raise ValueError(f"Invalid region_uot_grid_size: {value!r}")
            height, width = int(parts[0]), int(parts[1])
        else:
            height = width = int(text)
    else:
        parts = list(value)
        if len(parts) != 2:
            raise ValueError(f"Invalid region_uot_grid_size: {value!r}")
        height, width = int(parts[0]), int(parts[1])
    if height <= 1 or width <= 1:
        raise ValueError("region_uot_grid_size must be at least 2x2")
    return height, width


def _token_grid_assignments(
    spatial_hw: tuple[int, int],
    region_hw: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = int(spatial_hw[0]), int(spatial_hw[1])
    region_h, region_w = int(region_hw[0]), int(region_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial_hw={spatial_hw}")
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    region_y = torch.div(grid_y * region_h, height, rounding_mode="floor").clamp(max=region_h - 1)
    region_x = torch.div(grid_x * region_w, width, rounding_mode="floor").clamp(max=region_w - 1)
    region_index = (region_y * region_w + region_x).reshape(-1)
    return F.one_hot(region_index, num_classes=region_h * region_w).to(dtype=dtype)


def _normalized_region_coordinates(
    region_hw: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    region_h, region_w = int(region_hw[0]), int(region_hw[1])
    y = (torch.arange(region_h, device=device, dtype=dtype) + 0.5) / float(region_h)
    x = (torch.arange(region_w, device=device, dtype=dtype) + 0.5) / float(region_w)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)


def _entropy_confidence(prob: torch.Tensor, eps: float) -> torch.Tensor:
    count = int(prob.shape[-1])
    if count <= 1:
        return torch.ones(prob.shape[:-1], device=prob.device, dtype=prob.dtype)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
    return (1.0 - entropy / math.log(float(count))).clamp(0.0, 1.0)


class RegionStructuralUOTGuidance(nn.Module):
    """Coarse region UOT prior for fine-token transport costs."""

    def __init__(
        self,
        *,
        grid_size: str | int | Sequence[int] = "3x3",
        strength: float = 0.20,
        fgw_alpha: float = 0.35,
        tau: float = 0.5,
        sinkhorn_epsilon: float = 0.05,
        fgw_iters: int = 4,
        sinkhorn_iters: int = 40,
        sinkhorn_tol: float = 1e-5,
        ground_cost: str = "euclidean",
        topk: int = 3,
        fine_gate_quantile: float = 0.35,
        min_confidence: float = 0.10,
        importance_temperature: float = 0.50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.grid_size = parse_region_grid_size(grid_size)
        self.strength = float(strength)
        self.fgw_alpha = float(fgw_alpha)
        self.tau = float(tau)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.fgw_iters = int(fgw_iters)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.ground_cost = str(ground_cost).strip().lower().replace("-", "_")
        self.topk = int(topk)
        self.fine_gate_quantile = float(fine_gate_quantile)
        self.min_confidence = float(min_confidence)
        self.importance_temperature = float(importance_temperature)
        self.eps = float(eps)
        if not 0.0 <= self.strength < 1.0:
            raise ValueError("region_uot_strength must be in [0, 1)")
        if not 0.0 <= self.fgw_alpha <= 1.0:
            raise ValueError("region_uot_fgw_alpha must be in [0, 1]")
        if self.tau <= 0.0:
            raise ValueError("region_uot_tau must be positive")
        if self.sinkhorn_epsilon <= 0.0:
            raise ValueError("region_uot_sinkhorn_epsilon must be positive")
        if self.fgw_iters <= 0 or self.sinkhorn_iters <= 0:
            raise ValueError("region_uot solver iteration counts must be positive")
        if self.ground_cost not in {"auto", "euclidean", "cosine"}:
            raise ValueError("region_uot ground_cost must be auto/euclidean/cosine")
        if self.topk <= 0:
            raise ValueError("region_uot_topk must be positive")
        if not 0.0 < self.fine_gate_quantile < 1.0:
            raise ValueError("region_uot_fine_gate_quantile must be in (0, 1)")
        if not 0.0 <= self.min_confidence < 1.0:
            raise ValueError("region_uot_min_confidence must be in [0, 1)")
        if self.importance_temperature <= 0.0:
            raise ValueError("region_uot_importance_temperature must be positive")

    def _pool_regions(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        batch, token_count, dim = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if token_count != height * width:
            raise ValueError(f"spatial_hw={spatial_hw} does not match token count {token_count}")
        token_map = tokens.reshape(batch, height, width, dim).permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(token_map, output_size=self.grid_size)
        regions = pooled.permute(0, 2, 3, 1).reshape(batch, self.grid_size[0] * self.grid_size[1], dim)
        return F.normalize(regions, p=2, dim=-1, eps=self.eps)

    def _pairwise_feature_cost(self, query_regions: torch.Tensor, support_regions: torch.Tensor) -> torch.Tensor:
        mode = "euclidean" if self.ground_cost == "auto" else self.ground_cost
        if mode == "cosine":
            query_norm = F.normalize(query_regions, p=2, dim=-1, eps=self.eps)
            support_norm = F.normalize(support_regions, p=2, dim=-1, eps=self.eps)
            sim = torch.einsum("qrd,psd->qprs", query_norm, support_norm)
            return (1.0 - sim).clamp_min(0.0)
        query_sq = query_regions.pow(2).sum(dim=-1)
        support_sq = support_regions.pow(2).sum(dim=-1)
        dot = torch.einsum("qrd,psd->qprs", query_regions, support_regions)
        return (query_sq[:, None, :, None] + support_sq[None, :, None, :] - 2.0 * dot).clamp_min(0.0)

    def _importance_from_cost(self, pair_cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost_scale = pair_cost.detach().std(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        scaled_cost = pair_cost / (cost_scale * self.importance_temperature)
        query_logits = -scaled_cost.amin(dim=-1)
        support_logits = -scaled_cost.amin(dim=-2)
        query_prob = torch.softmax(query_logits, dim=-1).clamp_min(self.eps)
        support_prob = torch.softmax(support_logits, dim=-1).clamp_min(self.eps)
        query_prob = query_prob / query_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        support_prob = support_prob / support_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        confidence = 0.5 * (
            _entropy_confidence(query_prob, self.eps) + _entropy_confidence(support_prob, self.eps)
        )
        return query_prob, support_prob, confidence.detach()

    def _sparse_plan_and_confidence(self, plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rows, cols = plan.shape
        flat = plan.reshape(batch, rows * cols)
        topk = min(self.topk, int(flat.shape[-1]))
        _, indices = torch.topk(flat.detach(), k=topk, dim=-1)
        mask = torch.zeros_like(flat)
        mask.scatter_(dim=-1, index=indices, value=1.0)
        sparse_plan = plan * mask.reshape_as(plan)
        total_mass = plan.sum(dim=(-1, -2)).clamp_min(self.eps)
        sparse_mass_ratio = sparse_plan.sum(dim=(-1, -2)) / total_mass
        confidence = ((sparse_mass_ratio - self.min_confidence) / (1.0 - self.min_confidence)).clamp(0.0, 1.0)
        return sparse_plan, confidence.detach()

    def _fine_low_cost_gate(self, flat_cost: torch.Tensor) -> torch.Tensor:
        cost_flat = flat_cost.detach().reshape(flat_cost.shape[0], flat_cost.shape[1], -1)
        threshold = torch.quantile(cost_flat, q=self.fine_gate_quantile, dim=-1).reshape(
            flat_cost.shape[0],
            flat_cost.shape[1],
            1,
            1,
        )
        scale = flat_cost.detach().std(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        gate = torch.sigmoid((threshold - flat_cost.detach()) / (0.25 * scale))
        return gate

    def forward(
        self,
        *,
        flat_cost: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int],
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(f"query_tokens shape {tuple(query_tokens.shape)} does not match flat_cost")
        if tuple(support_tokens.shape[:3]) != (int(way_num), int(shot_num), support_len):
            raise ValueError(f"support_tokens shape {tuple(support_tokens.shape)} does not match flat_cost")

        query_regions = self._pool_regions(
            query_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype),
            spatial_hw,
        )
        support_flat = support_tokens.reshape(num_pairs, support_len, support_tokens.shape[-1])
        support_regions = self._pool_regions(
            support_flat.to(device=flat_cost.device, dtype=flat_cost.dtype),
            spatial_hw,
        )
        region_count = query_regions.shape[1]
        feature_cost = self._pairwise_feature_cost(query_regions, support_regions)
        base_mean = flat_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        feature_mean = feature_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        feature_cost = feature_cost * (base_mean / feature_mean)

        coords = _normalized_region_coordinates(self.grid_size, device=flat_cost.device, dtype=flat_cost.dtype)
        structure = torch.cdist(coords, coords, p=2).pow(2)
        structure = structure / structure.mean().clamp_min(self.eps)
        batch = num_query * num_pairs
        pair_cost = feature_cost.reshape(batch, region_count, region_count)
        structure_q = structure.unsqueeze(0).expand(batch, -1, -1)
        structure_s = structure.unsqueeze(0).expand(batch, -1, -1)
        query_prob, support_prob, importance_confidence = self._importance_from_cost(pair_cost)

        rho_tensor = torch.as_tensor(rho, device=flat_cost.device, dtype=flat_cost.dtype)
        if rho_tensor.numel() == 1:
            rho_flat = rho_tensor.reshape(1).expand(batch)
        else:
            rho_flat = rho_tensor.to(device=flat_cost.device, dtype=flat_cost.dtype).reshape(-1)
            if rho_flat.numel() == num_query:
                rho_flat = rho_flat[:, None].expand(num_query, num_pairs).reshape(batch)
            elif rho_flat.numel() == num_pairs:
                rho_flat = rho_flat[None, :].expand(num_query, num_pairs).reshape(batch)
            elif rho_flat.numel() != batch:
                raise ValueError(
                    "rho must be scalar or broadcastable to NumQuery*Way*Shot, "
                    f"got {rho_tensor.shape}"
                )
        rho_flat = rho_flat.clamp_min(self.eps)
        log_a = torch.log((rho_flat[:, None] * query_prob).clamp_min(self.eps))
        log_b = torch.log((rho_flat[:, None] * support_prob).clamp_min(self.eps))

        coarse_plan, coarse_cost = fgw_uot_solve(
            pair_cost,
            structure_q,
            structure_s,
            log_a,
            log_b,
            alpha=flat_cost.new_tensor(self.fgw_alpha),
            tau=self.tau,
            eps=self.sinkhorn_epsilon,
            fgw_iters=self.fgw_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )
        coarse_plan = torch.nan_to_num(coarse_plan, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        coarse_cost = torch.nan_to_num(coarse_cost, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        sparse_plan, sparse_confidence = self._sparse_plan_and_confidence(coarse_plan)
        plan_confidence = (sparse_confidence * importance_confidence).reshape(num_query, num_pairs, 1, 1)
        plan_norm = sparse_plan / sparse_plan.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        assignment = _token_grid_assignments(
            spatial_hw,
            self.grid_size,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if assignment.shape[0] != query_len or assignment.shape[0] != support_len:
            raise ValueError(
                "Region guidance currently expects query/support token grids with the same spatial_hw; "
                f"got assignment={assignment.shape[0]}, query_len={query_len}, support_len={support_len}"
            )
        fine_affinity = torch.matmul(torch.matmul(assignment, plan_norm), assignment.transpose(0, 1))
        fine_affinity = fine_affinity.reshape(num_query, num_pairs, query_len, support_len)
        fine_affinity = fine_affinity / fine_affinity.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        fine_gate = self._fine_low_cost_gate(flat_cost)

        discount = (self.strength * plan_confidence * fine_affinity * fine_gate).clamp(
            min=0.0,
            max=self.strength,
        )
        guided_cost = flat_cost * (1.0 - discount)

        coarse_plan_view = coarse_plan.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            region_count,
            region_count,
        )
        coarse_cost_view = coarse_cost.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            region_count,
            region_count,
        )
        coarse_transport_cost = (coarse_plan * coarse_cost).sum(dim=(-1, -2)).reshape(
            num_query,
            int(way_num),
            int(shot_num),
        )
        coarse_transport_mass = coarse_plan.sum(dim=(-1, -2)).reshape(num_query, int(way_num), int(shot_num))
        sparse_mass_ratio = (
            sparse_plan.sum(dim=(-1, -2)) / coarse_plan.sum(dim=(-1, -2)).clamp_min(self.eps)
        ).reshape(num_query, int(way_num), int(shot_num))
        effective_strength = discount.detach().amax(dim=(-1, -2)).reshape(num_query, int(way_num), int(shot_num))
        payload = {
            "region_uot_coarse_plan": coarse_plan_view.detach(),
            "region_uot_sparse_coarse_plan": sparse_plan.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                region_count,
                region_count,
            ).detach(),
            "region_uot_coarse_cost_matrix": coarse_cost_view.detach(),
            "region_uot_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ),
            "region_uot_coarse_transport_cost": coarse_transport_cost.detach(),
            "region_uot_coarse_transported_mass": coarse_transport_mass.detach(),
            "region_uot/strength": flat_cost.new_tensor(self.strength),
            "region_uot/fgw_alpha": flat_cost.new_tensor(self.fgw_alpha),
            "region_uot/topk": flat_cost.new_tensor(float(self.topk)),
            "region_uot/coarse_mass_mean": coarse_transport_mass.mean().detach(),
            "region_uot/coarse_cost_mean": coarse_transport_cost.mean().detach(),
            "region_uot/affinity_peak": fine_affinity.amax(dim=(-1, -2)).mean().detach(),
            "region_uot/sparse_mass_ratio": sparse_mass_ratio.mean().detach(),
            "region_uot/importance_confidence": importance_confidence.mean().detach(),
            "region_uot/effective_strength_mean": effective_strength.mean().detach(),
            "region_uot/fine_gate_mean": fine_gate.mean().detach(),
            "region_uot/cost_delta_ratio": (
                (guided_cost.detach() - flat_cost.detach()).abs().mean()
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ),
        }
        return guided_cost, payload
