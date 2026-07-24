"""Learned adaptive-region UOT guidance for Ours-Final.

This module discovers soft, multi-scale region slots from token features and
matches those slots with a small region-level UOT problem.  The resulting region
plan is used as a differentiable prior for the fine-token UOT cost and token
marginals.  It intentionally avoids image-brightness thresholds or connected
component rules so the region discovery remains part of the few-shot model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.modules.unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    sinkhorn_unbalanced_log,
)


def parse_adaptive_region_kernels(value: str | int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        kernels = [int(value)]
    elif isinstance(value, str):
        text = value.strip().lower().replace("x", ",")
        kernels = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        kernels = [int(part) for part in value]
    if not kernels:
        raise ValueError("adaptive_region_context_kernels must contain at least one kernel")
    normalized = []
    for kernel in kernels:
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("adaptive_region_context_kernels must be positive odd integers")
        normalized.append(kernel)
    return tuple(dict.fromkeys(normalized))


class AdaptiveRegionUOTGuidance(nn.Module):
    """Discover soft region slots and use region UOT to guide token transport."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_slots: int = 4,
        context_kernels: str | int | Sequence[int] = (1, 3, 5),
        cost_discount: float = 0.12,
        mass_mix: float = 0.60,
        sinkhorn_epsilon: float = 0.08,
        sinkhorn_iters: int = 30,
        sinkhorn_tol: float = 1e-5,
        tau: float = 0.5,
        fine_gate_quantile: float = 0.40,
        temperature_min: float = 0.35,
        temperature_max: float = 1.25,
        init_gate: float = 0.05,
        ground_cost: str = "euclidean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_slots = int(num_slots)
        self.context_kernels = parse_adaptive_region_kernels(context_kernels)
        self.cost_discount = float(cost_discount)
        self.mass_mix = float(mass_mix)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.tau = float(tau)
        self.fine_gate_quantile = float(fine_gate_quantile)
        self.init_gate = float(init_gate)
        self.ground_cost = str(ground_cost).strip().lower().replace("-", "_")
        self.eps = float(eps)

        if self.token_dim <= 0:
            raise ValueError("adaptive_region token_dim must be positive")
        if self.num_slots <= 0:
            raise ValueError("adaptive_region_num_slots must be positive")
        if not 0.0 <= self.cost_discount < 1.0:
            raise ValueError("adaptive_region_cost_discount must be in [0, 1)")
        if not 0.0 <= self.mass_mix <= 1.0:
            raise ValueError("adaptive_region_mass_mix must be in [0, 1]")
        if self.sinkhorn_epsilon <= 0.0:
            raise ValueError("adaptive_region_sinkhorn_epsilon must be positive")
        if self.sinkhorn_iters <= 0:
            raise ValueError("adaptive_region_sinkhorn_iters must be positive")
        if self.tau <= 0.0:
            raise ValueError("adaptive_region_tau must be positive")
        if not 0.0 < self.fine_gate_quantile < 1.0:
            raise ValueError("adaptive_region_fine_gate_quantile must be in (0, 1)")
        if temperature_min <= 0.0 or temperature_max <= 0.0:
            raise ValueError("adaptive_region temperatures must be positive")
        if temperature_max < temperature_min:
            raise ValueError("adaptive_region_temperature_max must be >= temperature_min")
        if not 0.0 <= self.init_gate <= 1.0:
            raise ValueError("adaptive_region_init_gate must be in [0, 1]")
        if self.ground_cost not in {"auto", "euclidean", "cosine"}:
            raise ValueError("adaptive_region ground_cost must be auto/euclidean/cosine")

        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.slot_queries = nn.Parameter(torch.empty(self.num_slots, self.token_dim))
        self.slot_gate = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
        )
        self.context_logits = nn.Parameter(torch.zeros(len(self.context_kernels)))
        if self.num_slots == 1:
            init_temps = torch.tensor([float(temperature_min)])
        else:
            init_temps = torch.linspace(float(temperature_min), float(temperature_max), self.num_slots)
        self.raw_slot_temperatures = nn.Parameter(_inverse_softplus_local(init_temps))
        init_gate_tensor = torch.tensor(float(self.init_gate))
        self.raw_cost_gate = nn.Parameter(_inverse_sigmoid_local(init_gate_tensor))
        self.raw_mass_gate = nn.Parameter(_inverse_sigmoid_local(init_gate_tensor))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_queries, mean=0.0, std=1.0 / math.sqrt(float(self.token_dim)))
        nn.init.xavier_uniform_(self.token_proj.weight)
        gate_linear = self.slot_gate[-1]
        if isinstance(gate_linear, nn.Linear):
            nn.init.zeros_(gate_linear.weight)
            nn.init.zeros_(gate_linear.bias)

    def _slot_masks(
        self,
        tokens: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        batch, token_count, dim = tokens.shape
        if dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {dim}")
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if height * width != token_count:
            raise ValueError(f"spatial_hw={spatial_hw} does not match token count {token_count}")

        input_dtype = tokens.dtype
        network_dtype = self.token_proj.weight.dtype
        network_tokens = tokens.to(dtype=network_dtype)
        projected = self.token_proj(self.token_norm(network_tokens))
        slot_queries = F.normalize(self.slot_queries.to(device=tokens.device, dtype=network_dtype), dim=-1)
        raw_logits = torch.einsum("bld,kd->bkl", projected, slot_queries) / math.sqrt(float(dim))
        raw_map = raw_logits.reshape(batch * self.num_slots, 1, height, width)
        context_weights = torch.softmax(self.context_logits.to(device=tokens.device, dtype=network_dtype), dim=0)
        smoothed = raw_map.new_zeros(raw_map.shape)
        for weight, kernel in zip(context_weights, self.context_kernels):
            smoothed = smoothed + weight * F.avg_pool2d(
                raw_map,
                kernel_size=int(kernel),
                stride=1,
                padding=int(kernel) // 2,
                count_include_pad=False,
            )
        logits = smoothed.reshape(batch, self.num_slots, token_count)
        temperatures = F.softplus(self.raw_slot_temperatures).to(device=tokens.device, dtype=network_dtype)
        masks = torch.softmax(logits / temperatures.reshape(1, self.num_slots, 1).clamp_min(self.eps), dim=-1)
        masks = masks.to(dtype=input_dtype)
        descriptors = torch.einsum("bkl,bld->bkd", masks, tokens)
        descriptors = F.normalize(descriptors, p=2, dim=-1, eps=self.eps)
        gate_logits = self.slot_gate(descriptors.to(dtype=network_dtype)).squeeze(-1).to(dtype=input_dtype)
        slot_prob = torch.softmax(gate_logits, dim=-1).clamp_min(self.eps)
        slot_prob = slot_prob / slot_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        token_prob = torch.einsum("bk,bkl->bl", slot_prob, masks)
        token_prob = token_prob / token_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return masks, descriptors, slot_prob, token_prob

    def _pairwise_region_cost(
        self,
        query_regions: torch.Tensor,
        support_regions: torch.Tensor,
    ) -> torch.Tensor:
        mode = "euclidean" if self.ground_cost == "auto" else self.ground_cost
        if mode == "cosine":
            query_norm = F.normalize(query_regions, p=2, dim=-1, eps=self.eps)
            support_norm = F.normalize(support_regions, p=2, dim=-1, eps=self.eps)
            sim = torch.einsum("qkd,pjd->qpkj", query_norm, support_norm)
            return (1.0 - sim).clamp_min(0.0)
        query_sq = query_regions.pow(2).sum(dim=-1)
        support_sq = support_regions.pow(2).sum(dim=-1)
        dot = torch.einsum("qkd,pjd->qpkj", query_regions, support_regions)
        return (query_sq[:, None, :, None] + support_sq[None, :, None, :] - 2.0 * dot).clamp_min(0.0)

    def _fine_low_cost_gate(self, flat_cost: torch.Tensor) -> torch.Tensor:
        cost_flat = flat_cost.detach().reshape(flat_cost.shape[0], flat_cost.shape[1], -1)
        threshold = torch.quantile(cost_flat, q=self.fine_gate_quantile, dim=-1).reshape(
            flat_cost.shape[0],
            flat_cost.shape[1],
            1,
            1,
        )
        scale = flat_cost.detach().std(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        return torch.sigmoid((threshold - flat_cost.detach()) / (0.25 * scale))

    def _rho_flat(
        self,
        rho: torch.Tensor | float,
        *,
        num_query: int,
        num_pairs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rho_tensor = torch.as_tensor(rho, device=device, dtype=dtype)
        if rho_tensor.numel() == 1:
            return rho_tensor.reshape(1).expand(num_query * num_pairs).clamp_min(self.eps)
        rho_flat = rho_tensor.reshape(-1)
        if rho_flat.numel() == num_query:
            rho_flat = rho_flat[:, None].expand(num_query, num_pairs).reshape(-1)
        elif rho_flat.numel() == num_pairs:
            rho_flat = rho_flat[None, :].expand(num_query, num_pairs).reshape(-1)
        elif rho_flat.numel() != num_query * num_pairs:
            raise ValueError(
                "rho must be scalar or broadcastable to NumQuery*Way*Shot, "
                f"got {tuple(rho_tensor.shape)}"
            )
        return rho_flat.clamp_min(self.eps)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(f"query_tokens shape {tuple(query_tokens.shape)} does not match flat_cost")
        if tuple(support_tokens.shape[:3]) != (int(way_num), int(shot_num), support_len):
            raise ValueError(f"support_tokens shape {tuple(support_tokens.shape)} does not match flat_cost")

        query_tokens = query_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_tokens = support_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_flat = support_tokens.reshape(num_pairs, support_len, support_tokens.shape[-1])

        query_masks, query_regions, query_slot_prob, query_token_prob = self._slot_masks(query_tokens, spatial_hw)
        support_masks_flat, support_regions_flat, support_slot_prob_flat, support_token_prob_flat = self._slot_masks(
            support_flat,
            spatial_hw,
        )
        region_cost = self._pairwise_region_cost(query_regions, support_regions_flat).to(dtype=flat_cost.dtype)
        base_mean = flat_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        region_mean = region_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        region_cost = region_cost * (base_mean / region_mean)

        batch = num_query * num_pairs
        pair_region_cost = region_cost.reshape(batch, self.num_slots, self.num_slots)
        rho_flat = self._rho_flat(
            rho,
            num_query=num_query,
            num_pairs=num_pairs,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        pair_query_slot = query_slot_prob[:, None, :].expand(num_query, num_pairs, self.num_slots).reshape(
            batch,
            self.num_slots,
        )
        pair_support_slot = support_slot_prob_flat[None, :, :].expand(num_query, num_pairs, self.num_slots).reshape(
            batch,
            self.num_slots,
        )
        region_a = pair_query_slot * rho_flat[:, None]
        region_b = pair_support_slot * rho_flat[:, None]
        region_plan = sinkhorn_unbalanced_log(
            pair_region_cost,
            region_a,
            region_b,
            tau_q=self.tau,
            tau_c=self.tau,
            eps=self.sinkhorn_epsilon,
            max_iter=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )
        region_plan = torch.nan_to_num(region_plan, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        region_plan_view = region_plan.reshape(num_query, num_pairs, self.num_slots, self.num_slots)
        plan_norm = region_plan_view / region_plan_view.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        fine_affinity = torch.einsum(
            "qkl,qpij,pjm->qplm",
            query_masks,
            plan_norm,
            support_masks_flat,
        )
        fine_affinity = fine_affinity / fine_affinity.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        fine_gate = self._fine_low_cost_gate(flat_cost)
        cost_gate = torch.sigmoid(self.raw_cost_gate).to(device=flat_cost.device, dtype=flat_cost.dtype)
        mass_gate = torch.sigmoid(self.raw_mass_gate).to(device=flat_cost.device, dtype=flat_cost.dtype)
        effective_cost_discount = (flat_cost.new_tensor(self.cost_discount) * cost_gate).clamp(
            min=0.0,
            max=self.cost_discount,
        )
        effective_mass_mix = (flat_cost.new_tensor(self.mass_mix) * mass_gate).clamp(
            min=0.0,
            max=self.mass_mix,
        )
        discount = (effective_cost_discount * fine_affinity * fine_gate).clamp(
            min=0.0,
            max=self.cost_discount,
        )
        guided_cost = flat_cost * (1.0 - discount)

        query_uniform = query_token_prob.new_full(query_token_prob.shape, 1.0 / float(query_len))
        support_uniform = support_token_prob_flat.new_full(support_token_prob_flat.shape, 1.0 / float(support_len))
        query_weight = (1.0 - effective_mass_mix) * query_uniform + effective_mass_mix * query_token_prob
        support_weight_flat = (
            (1.0 - effective_mass_mix) * support_uniform
            + effective_mass_mix * support_token_prob_flat
        )
        support_weight = support_weight_flat.reshape(int(way_num), int(shot_num), support_len)

        region_transport_cost = compute_transport_cost(region_plan, pair_region_cost).reshape(
            num_query,
            int(way_num),
            int(shot_num),
        )
        region_transported_mass = compute_transported_mass(region_plan).reshape(
            num_query,
            int(way_num),
            int(shot_num),
        )
        query_entropy = -(query_token_prob * query_token_prob.clamp_min(self.eps).log()).sum(dim=-1)
        support_entropy = -(
            support_token_prob_flat * support_token_prob_flat.clamp_min(self.eps).log()
        ).sum(dim=-1)
        payload = {
            "adaptive_region_query_masks": query_masks.detach(),
            "adaptive_region_support_masks": support_masks_flat.reshape(
                int(way_num),
                int(shot_num),
                self.num_slots,
                support_len,
            ).detach(),
            "adaptive_region_query_weight": query_weight.detach(),
            "adaptive_region_support_weight": support_weight.detach(),
            "adaptive_region_plan": region_plan_view.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                self.num_slots,
                self.num_slots,
            ).detach(),
            "adaptive_region_cost_matrix": region_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                self.num_slots,
                self.num_slots,
            ).detach(),
            "adaptive_region_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ),
            "adaptive_region_transport_cost": region_transport_cost.detach(),
            "adaptive_region_transported_mass": region_transported_mass.detach(),
            "adaptive_region/num_slots": flat_cost.new_tensor(float(self.num_slots)),
            "adaptive_region/cost_discount": flat_cost.new_tensor(self.cost_discount),
            "adaptive_region/mass_mix": flat_cost.new_tensor(self.mass_mix),
            "adaptive_region/cost_gate": cost_gate.detach(),
            "adaptive_region/mass_gate": mass_gate.detach(),
            "adaptive_region/effective_cost_discount": effective_cost_discount.detach(),
            "adaptive_region/effective_mass_mix": effective_mass_mix.detach(),
            "adaptive_region/region_cost_mean": region_transport_cost.mean().detach(),
            "adaptive_region/region_mass_mean": region_transported_mass.mean().detach(),
            "adaptive_region/fine_affinity_peak": fine_affinity.amax(dim=(-1, -2)).mean().detach(),
            "adaptive_region/fine_gate_mean": fine_gate.mean().detach(),
            "adaptive_region/cost_delta_ratio": (
                (guided_cost.detach() - flat_cost.detach()).abs().mean()
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ),
            "adaptive_region/query_weight_peak": query_weight.max(dim=-1).values.mean().detach(),
            "adaptive_region/support_weight_peak": support_weight_flat.max(dim=-1).values.mean().detach(),
            "adaptive_region/query_effective_area": (
                query_entropy.exp() / float(query_len)
            ).mean().detach(),
            "adaptive_region/support_effective_area": (
                support_entropy.exp() / float(support_len)
            ).mean().detach(),
            "adaptive_region/context_kernel_entropy": (
                -torch.softmax(self.context_logits, dim=0)
                * torch.softmax(self.context_logits, dim=0).clamp_min(self.eps).log()
            ).sum().detach(),
        }
        return guided_cost, query_weight, support_weight, payload


def _inverse_softplus_local(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value.clamp_min(1e-8)))


def _inverse_sigmoid_local(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(value) - torch.log1p(-value)


__all__ = ["AdaptiveRegionUOTGuidance", "parse_adaptive_region_kernels"]
