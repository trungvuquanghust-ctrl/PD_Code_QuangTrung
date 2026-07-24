"""Evidence-adaptive token relaxation for Ours-Final UOT."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _infer_hw(token_count: int) -> tuple[int, int]:
    height = max(1, int(math.sqrt(int(token_count))))
    while height > 1 and int(token_count) % height != 0:
        height -= 1
    return height, int(token_count) // height


class EvidenceAdaptiveRelaxationUOT(nn.Module):
    """Build class-conditional, token-wise UOT marginal penalties.

    The reliability field combines candidate-vs-rival specificity, positive
    threshold utility, and local spatial consensus. It changes the strength of
    mass conservation, not the target marginals or the ground cost.
    """

    def __init__(
        self,
        *,
        tau_min: float = 0.05,
        tau_max: float = 1.00,
        temperature: float = 0.25,
        margin: float = 0.0,
        spatial_mix: float = 0.50,
        kernel_size: int = 3,
        detach_reliability: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        if not 0.0 <= spatial_mix <= 1.0:
            raise ValueError("spatial_mix must be in [0, 1]")
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.spatial_mix = float(spatial_mix)
        self.kernel_size = int(kernel_size)
        self.detach_reliability = bool(detach_reliability)
        self.eps = float(eps)

    def _smooth(
        self,
        values: torch.Tensor,
        *,
        spatial_hw: tuple[int, int] | None,
    ) -> torch.Tensor:
        token_count = int(values.shape[-1])
        if self.spatial_mix <= 0.0 or token_count <= 1:
            return values
        height, width = (
            (int(spatial_hw[0]), int(spatial_hw[1]))
            if spatial_hw is not None and int(spatial_hw[0]) * int(spatial_hw[1]) == token_count
            else _infer_hw(token_count)
        )
        flat = values.reshape(-1, 1, height, width)
        local = F.avg_pool2d(
            flat,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        ).reshape_as(values)
        return (1.0 - self.spatial_mix) * values + self.spatial_mix * local

    @staticmethod
    def _threshold_scalar(
        threshold: torch.Tensor | float,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(threshold, device=reference.device, dtype=reference.dtype)
        if value.numel() != 1:
            raise ValueError("EAR-UOT currently requires a scalar transport threshold")
        return value.reshape(())

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        threshold: torch.Tensor | float,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (NumQuery, Way*Shot, QueryTokens, SupportTokens)"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        way_num = int(way_num)
        shot_num = int(shot_num)
        if num_pairs != way_num * shot_num:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}"
            )

        cost = flat_cost.reshape(
            num_query,
            way_num,
            shot_num,
            query_len,
            support_len,
        )
        reference_cost = cost.detach() if self.detach_reliability else cost
        episode_scale = reference_cost.std(
            dim=(1, 2, 3, 4),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        temperature = (self.temperature * episode_scale).clamp_min(self.eps)
        query_temperature = temperature.reshape(num_query, 1, 1)
        threshold_value = self._threshold_scalar(threshold, reference_cost)
        if self.detach_reliability:
            threshold_value = threshold_value.detach()

        candidate_query_cost = -query_temperature * (
            torch.logsumexp(-reference_cost / temperature, dim=(2, 4))
            - math.log(float(max(shot_num * support_len, 1)))
        )
        class_probability = torch.softmax(
            -candidate_query_cost / query_temperature,
            dim=1,
        )
        if way_num > 1:
            query_specificity = (
                (float(way_num) * class_probability - 1.0) / float(way_num - 1)
            ).clamp(0.0, 1.0)
            rival_cost = torch.stack(
                [
                    torch.cat(
                        [candidate_query_cost[:, :class_idx], candidate_query_cost[:, class_idx + 1 :]],
                        dim=1,
                    ).amin(dim=1)
                    for class_idx in range(way_num)
                ],
                dim=1,
            )
            rival_advantage = rival_cost - candidate_query_cost
            margin_value = self.margin * episode_scale.reshape(num_query, 1, 1)
            advantage_gate = torch.sigmoid(
                (rival_advantage - margin_value)
                / query_temperature
            )
            query_specificity = query_specificity * advantage_gate
        else:
            rival_advantage = torch.zeros_like(candidate_query_cost)
            query_specificity = torch.ones_like(candidate_query_cost)

        query_matchability = torch.sigmoid(
            (threshold_value - candidate_query_cost)
            / query_temperature
        )
        reliability_product = (
            query_specificity * query_matchability
        ).clamp(0.0, 1.0)
        reliability_eps = reference_cost.new_tensor(max(self.eps, 1e-6))
        reliability_floor = torch.sqrt(reliability_eps)
        reliability_scale = (
            torch.sqrt(reference_cost.new_tensor(1.0) + reliability_eps)
            - reliability_floor
        ).clamp_min(reliability_eps)
        query_reliability = (
            torch.sqrt(reliability_product + reliability_eps)
            - reliability_floor
        ) / reliability_scale
        query_reliability = self._smooth(
            query_reliability,
            spatial_hw=spatial_hw,
        ).clamp(0.0, 1.0)
        if self.detach_reliability:
            query_reliability = query_reliability.detach()

        row_min = reference_cost.amin(dim=-1, keepdim=True)
        edge_affinity = torch.exp(
            -(reference_cost - row_min) / temperature
        ).clamp(0.0, 1.0)
        edge_utility_gate = torch.sigmoid(
            (threshold_value - reference_cost) / temperature
        )
        pair_reliability = (
            query_reliability[:, :, None, :, None]
            * edge_affinity
            * edge_utility_gate
        )
        support_reliability = pair_reliability.amax(dim=-2)
        support_reliability = self._smooth(
            support_reliability,
            spatial_hw=spatial_hw,
        ).clamp(0.0, 1.0)
        if self.detach_reliability:
            support_reliability = support_reliability.detach()

        query_reliability_pair = (
            query_reliability[:, :, None, :]
            .expand(-1, -1, shot_num, -1)
            .reshape(num_query, num_pairs, query_len)
        )
        support_reliability_pair = support_reliability.reshape(
            num_query,
            num_pairs,
            support_len,
        )
        tau_span = flat_cost.new_tensor(self.tau_max - self.tau_min)
        tau_floor = flat_cost.new_tensor(self.tau_min)
        tau_q = tau_floor + tau_span * query_reliability_pair
        tau_c = tau_floor + tau_span * support_reliability_pair

        diagnostics = {
            "ear_uot_query_reliability": query_reliability_pair.detach(),
            "ear_uot_support_reliability": support_reliability_pair.detach(),
            "ear_uot_query_tau": tau_q.detach(),
            "ear_uot_support_tau": tau_c.detach(),
            "ear_uot_rival_advantage": rival_advantage.detach(),
            "ear_uot/enabled": flat_cost.new_tensor(1.0),
            "ear_uot/tau_min": flat_cost.new_tensor(self.tau_min),
            "ear_uot/tau_max": flat_cost.new_tensor(self.tau_max),
            "ear_uot/temperature": flat_cost.new_tensor(self.temperature),
            "ear_uot/margin": flat_cost.new_tensor(self.margin),
            "ear_uot/spatial_mix": flat_cost.new_tensor(self.spatial_mix),
            "ear_uot/query_reliability_mean": query_reliability_pair.mean().detach(),
            "ear_uot/support_reliability_mean": support_reliability_pair.mean().detach(),
            "ear_uot/query_tau_mean": tau_q.mean().detach(),
            "ear_uot/support_tau_mean": tau_c.mean().detach(),
            "ear_uot/query_tau_std": tau_q.std(unbiased=False).detach(),
            "ear_uot/support_tau_std": tau_c.std(unbiased=False).detach(),
            "ear_uot/high_reliability_query_share": (
                query_reliability_pair >= 0.5
            ).to(flat_cost.dtype).mean().detach(),
            "ear_uot/high_reliability_support_share": (
                support_reliability_pair >= 0.5
            ).to(flat_cost.dtype).mean().detach(),
            "ear_uot/rival_advantage_mean": rival_advantage.mean().detach(),
            "ear_uot/positive_rival_advantage_share": (
                rival_advantage > 0.0
            ).to(flat_cost.dtype).mean().detach(),
        }
        return tau_q, tau_c, diagnostics


__all__ = ["EvidenceAdaptiveRelaxationUOT"]
