"""Pulse-region guidance for Ours-Final transport.

The module keeps the original local token transport path intact, but supplies
two opt-in priors:

* saliency-shaped token marginals so UOT spends more mass on bright/energetic
  pulse regions; and
* region-context costs, computed by pooling tokens over a larger spatial
  neighborhood, so local matches are guided by the surrounding pulse shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("pulse_region_kernel_size must be a positive odd integer")
    return kernel_size


def normalize_saliency(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert non-negative token saliency to a per-image probability vector."""
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = values.sum(dim=-1, keepdim=True)
    uniform = values.new_full(values.shape, 1.0 / float(values.shape[-1]))
    prob = values / total.clamp_min(float(eps))
    return torch.where(total > float(eps), prob, uniform)


def blend_uniform_with_saliency(
    saliency: torch.Tensor,
    mix: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a probability vector between uniform and saliency probability."""
    mix = float(mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("pulse_saliency_mass_mix must be in [0, 1]")
    prob = normalize_saliency(saliency, eps=eps)
    uniform = prob.new_full(prob.shape, 1.0 / float(prob.shape[-1]))
    return (1.0 - mix) * uniform + mix * prob


class PulseRegionGuidance(nn.Module):
    """Build saliency marginals and region-guided costs for token UOT."""

    def __init__(
        self,
        *,
        kernel_size: int = 5,
        region_cost_weight: float = 0.35,
        saliency_mass_mix: float = 0.50,
        saliency_cost_discount: float = 0.10,
        ground_cost: str = "euclidean",
        normalize_region_cost: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.kernel_size = _validate_odd_kernel(kernel_size)
        self.region_cost_weight = float(region_cost_weight)
        self.saliency_mass_mix = float(saliency_mass_mix)
        self.saliency_cost_discount = float(saliency_cost_discount)
        self.ground_cost = str(ground_cost).strip().lower().replace("-", "_")
        self.normalize_region_cost = bool(normalize_region_cost)
        self.eps = float(eps)
        if not 0.0 <= self.region_cost_weight <= 1.0:
            raise ValueError("pulse_region_cost_weight must be in [0, 1]")
        if not 0.0 <= self.saliency_mass_mix <= 1.0:
            raise ValueError("pulse_saliency_mass_mix must be in [0, 1]")
        if not 0.0 <= self.saliency_cost_discount < 1.0:
            raise ValueError("pulse_saliency_cost_discount must be in [0, 1)")
        if self.ground_cost not in {"auto", "euclidean", "cosine"}:
            raise ValueError("pulse_region ground_cost must be auto/euclidean/cosine")

    def _region_tokens(
        self,
        tokens: torch.Tensor,
        saliency: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        batch, token_count, dim = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if token_count != height * width:
            raise ValueError(
                f"spatial_hw={spatial_hw} does not match token count {token_count}"
            )
        if tuple(saliency.shape) != (batch, token_count):
            raise ValueError(
                f"saliency must have shape {(batch, token_count)}, got {tuple(saliency.shape)}"
            )

        token_map = tokens.reshape(batch, height, width, dim).permute(0, 3, 1, 2).contiguous()
        saliency_map = saliency.reshape(batch, 1, height, width).to(
            device=tokens.device,
            dtype=tokens.dtype,
        ).clamp_min(0.0)
        padding = self.kernel_size // 2
        weighted = F.avg_pool2d(
            token_map * saliency_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        denom = F.avg_pool2d(
            saliency_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        fallback = F.avg_pool2d(
            token_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        region_map = torch.where(denom > self.eps, weighted / denom.clamp_min(self.eps), fallback)
        region_tokens = region_map.permute(0, 2, 3, 1).reshape(batch, token_count, dim)
        return F.normalize(region_tokens, p=2, dim=-1, eps=self.eps)

    def _pairwise_cost(self, query_tokens: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
        mode = "euclidean" if self.ground_cost == "auto" else self.ground_cost
        if mode == "cosine":
            query_norm = F.normalize(query_tokens, p=2, dim=-1, eps=self.eps)
            support_norm = F.normalize(support_tokens, p=2, dim=-1, eps=self.eps)
            sim = torch.einsum("qtd,psd->qpts", query_norm, support_norm)
            return (1.0 - sim).clamp_min(0.0)
        query_sq = query_tokens.pow(2).sum(dim=-1)
        support_sq = support_tokens.pow(2).sum(dim=-1)
        dot = torch.einsum("qtd,psd->qpts", query_tokens, support_tokens)
        return (query_sq[:, None, :, None] + support_sq[None, :, None, :] - 2.0 * dot).clamp_min(0.0)

    def forward(
        self,
        *,
        flat_cost: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_saliency: torch.Tensor,
        support_saliency: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int],
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

        query_saliency = query_saliency.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_saliency = support_saliency.to(device=flat_cost.device, dtype=flat_cost.dtype)
        if tuple(query_saliency.shape) != (num_query, query_len):
            raise ValueError(f"query_saliency must have shape {(num_query, query_len)}, got {tuple(query_saliency.shape)}")
        if tuple(support_saliency.shape) != (int(way_num), int(shot_num), support_len):
            raise ValueError(
                f"support_saliency must have shape {(way_num, shot_num, support_len)}, got {tuple(support_saliency.shape)}"
            )

        support_flat = support_tokens.reshape(num_pairs, support_len, support_tokens.shape[-1])
        support_saliency_flat = support_saliency.reshape(num_pairs, support_len)
        query_region = self._region_tokens(query_tokens, query_saliency, spatial_hw)
        support_region = self._region_tokens(support_flat, support_saliency_flat, spatial_hw)
        region_cost = self._pairwise_cost(query_region, support_region).to(dtype=flat_cost.dtype)
        if self.normalize_region_cost:
            base_mean = flat_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
            region_mean = region_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
            region_cost = region_cost * (base_mean / region_mean)

        guided_cost = (1.0 - self.region_cost_weight) * flat_cost + self.region_cost_weight * region_cost

        query_weight = blend_uniform_with_saliency(
            query_saliency,
            mix=self.saliency_mass_mix,
            eps=self.eps,
        )
        support_weight = blend_uniform_with_saliency(
            support_saliency,
            mix=self.saliency_mass_mix,
            eps=self.eps,
        )
        if self.saliency_cost_discount > 0.0:
            q_prior = query_weight / query_weight.amax(dim=-1, keepdim=True).clamp_min(self.eps)
            s_prior = support_weight.reshape(num_pairs, support_len)
            s_prior = s_prior / s_prior.amax(dim=-1, keepdim=True).clamp_min(self.eps)
            saliency_pair = torch.sqrt(
                (q_prior[:, None, :, None] * s_prior[None, :, None, :]).clamp_min(0.0)
            )
            guided_cost = guided_cost * (1.0 - self.saliency_cost_discount * saliency_pair)

        payload = {
            "pulse_query_saliency": query_saliency.detach(),
            "pulse_support_saliency": support_saliency.detach(),
            "pulse_query_marginal_weight": query_weight.detach(),
            "pulse_support_marginal_weight": support_weight.detach(),
            "pulse_region_cost_matrix": region_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ).detach(),
            "pulse_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ),
            "pulse/region_cost_weight": flat_cost.new_tensor(self.region_cost_weight),
            "pulse/saliency_mass_mix": flat_cost.new_tensor(self.saliency_mass_mix),
            "pulse/saliency_cost_discount": flat_cost.new_tensor(self.saliency_cost_discount),
            "pulse/query_saliency_peak": query_saliency.max(dim=-1).values.mean().detach(),
            "pulse/support_saliency_peak": support_saliency.max(dim=-1).values.mean().detach(),
            "pulse/region_cost_delta_ratio": (
                (guided_cost.detach() - flat_cost.detach()).abs().mean()
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ),
        }
        return guided_cost, query_weight, support_weight, payload
