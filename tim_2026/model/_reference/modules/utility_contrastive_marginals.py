from __future__ import annotations

import math

import torch
from torch import nn


class UtilityContrastiveMarginals(nn.Module):
    """Shared-query marginals aligned with the original Ours-Final T*M-C utility."""

    def __init__(
        self,
        *,
        tau: float = 0.25,
        marginal_mix: float = 0.65,
        adaptive_mix: bool = True,
        confidence_power: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 <= marginal_mix <= 1.0:
            raise ValueError("marginal_mix must be in [0, 1]")
        if confidence_power <= 0.0:
            raise ValueError("confidence_power must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.tau = float(tau)
        self.marginal_mix = float(marginal_mix)
        self.adaptive_mix = bool(adaptive_mix)
        self.confidence_power = float(confidence_power)
        self.eps = float(eps)

    def _normalize_with_uniform_fallback(self, raw: torch.Tensor) -> torch.Tensor:
        total = raw.sum(dim=-1, keepdim=True)
        normalized = raw / total.clamp_min(self.eps)
        uniform = torch.full_like(raw, 1.0 / float(max(raw.shape[-1], 1)))
        return torch.where(total > self.eps, normalized, uniform)

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        threshold: torch.Tensor | float,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (Nq, Way*Shot, Lq, Ls), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        expected_pairs = int(way_num) * int(shot_num)
        if num_pairs != expected_pairs:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={expected_pairs}"
            )

        reference_cost = flat_cost.detach()
        cost_view = reference_cost.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
        )
        cost_scale = reference_cost.std(
            dim=(1, 2, 3),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        temperature = (self.tau * cost_scale).clamp_min(self.eps)
        temperature_5d = temperature.reshape(num_query, 1, 1, 1, 1)

        class_count = max(int(shot_num) * support_len, 1)
        class_cost = -temperature.reshape(num_query, 1, 1) * (
            torch.logsumexp(-cost_view / temperature_5d, dim=(2, 4))
            - math.log(float(class_count))
        )
        if int(way_num) > 1:
            sorted_cost = class_cost.sort(dim=1).values
            best_cost = sorted_cost[:, 0]
            second_cost = sorted_cost[:, 1]
            query_advantage = second_cost - best_cost
            rival_cost = torch.stack(
                [
                    torch.cat(
                        [class_cost[:, :idx], class_cost[:, idx + 1 :]],
                        dim=1,
                    ).amin(dim=1)
                    for idx in range(int(way_num))
                ],
                dim=1,
            )
            rival_advantage = rival_cost - class_cost
        else:
            query_advantage = torch.zeros_like(class_cost[:, 0])
            rival_advantage = torch.zeros_like(class_cost)

        positive_advantage = rival_advantage.clamp_min(0.0)
        class_specificity = 1.0 - torch.exp(
            -positive_advantage / temperature.reshape(num_query, 1, 1)
        )
        query_specificity = class_specificity.amax(dim=1)

        threshold_value = torch.as_tensor(
            threshold,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if threshold_value.numel() != 1:
            raise ValueError("UtilityContrastiveMarginals currently requires a scalar threshold")
        best_edge_cost = cost_view.amin(dim=(1, 2, 4))
        query_positive_utility = torch.sigmoid(
            (threshold_value - best_edge_cost) / temperature.reshape(num_query, 1)
        )
        query_raw = query_specificity * query_positive_utility
        query_selective = self._normalize_with_uniform_fallback(query_raw)

        evidence_confidence = query_raw.amax(dim=-1, keepdim=True).pow(
            self.confidence_power
        )
        configured_mix = flat_cost.new_tensor(self.marginal_mix)
        if self.adaptive_mix:
            effective_mix = configured_mix * evidence_confidence
        else:
            effective_mix = configured_mix.expand_as(evidence_confidence)

        query_uniform = torch.full_like(
            query_selective,
            1.0 / float(max(query_len, 1)),
        )
        query_prob = (
            (1.0 - effective_mix) * query_uniform
            + effective_mix * query_selective
        )

        edge_positive_utility = torch.sigmoid(
            (threshold_value - cost_view) / temperature_5d
        )
        row_min = cost_view.amin(dim=-1, keepdim=True)
        row_affinity = torch.exp(
            -(cost_view - row_min) / temperature_5d
        ).clamp(0.0, 1.0)
        pair_evidence = (
            class_specificity[:, :, None, :, None]
            * edge_positive_utility
            * row_affinity
        ).clamp(0.0, 1.0)

        support_raw = (
            query_prob[:, None, None, :, None] * pair_evidence
        ).sum(dim=-2)
        support_selective = self._normalize_with_uniform_fallback(support_raw)
        support_uniform = torch.full_like(
            support_selective,
            1.0 / float(max(support_len, 1)),
        )
        support_mix = effective_mix[:, None, None, :]
        support_prob = (
            (1.0 - support_mix) * support_uniform
            + support_mix * support_selective
        )
        support_prob = support_prob.reshape(num_query, num_pairs, support_len)

        query_entropy = -(
            query_prob * query_prob.clamp_min(self.eps).log()
        ).sum(dim=-1)
        support_entropy = -(
            support_prob * support_prob.clamp_min(self.eps).log()
        ).sum(dim=-1)
        payload = {
            "score_marginal_pair_evidence": pair_evidence.detach(),
            "score_marginal_query_weight": query_prob.detach(),
            "score_marginal_support_weight": support_prob.detach(),
            "score_marginal_query_advantage": query_advantage.detach(),
            "score_marginal/enabled": flat_cost.new_tensor(1.0),
            "score_marginal/tau": flat_cost.new_tensor(self.tau),
            "score_marginal/mix": configured_mix.detach(),
            "score_marginal/adaptive_mix": flat_cost.new_tensor(
                float(self.adaptive_mix)
            ),
            "score_marginal/effective_mix": effective_mix.mean().detach(),
            "score_marginal/evidence_confidence": evidence_confidence.mean().detach(),
            "score_marginal/query_is_class_shared": flat_cost.new_tensor(1.0),
            "score_marginal/uniform_fallback_query_share": (
                query_raw.sum(dim=-1) <= self.eps
            ).to(flat_cost.dtype).mean().detach(),
            "score_marginal/uniform_fallback_support_share": (
                support_raw.sum(dim=-1) <= self.eps
            ).to(flat_cost.dtype).mean().detach(),
            "score_marginal/query_weight_entropy": query_entropy.mean().detach(),
            "score_marginal/support_weight_entropy": support_entropy.mean().detach(),
            "score_marginal/query_weight_peak": query_prob.amax(dim=-1).mean().detach(),
            "score_marginal/support_weight_peak": support_prob.amax(dim=-1).mean().detach(),
            "score_marginal/class_advantage_mean": rival_advantage.mean().detach(),
            "score_marginal/positive_advantage_mean": positive_advantage.mean().detach(),
            "score_marginal/query_threshold_acceptance_mean": (
                query_positive_utility.mean().detach()
            ),
            "score_marginal/edge_threshold_acceptance_mean": (
                edge_positive_utility.mean().detach()
            ),
        }
        return query_prob, support_prob, payload


__all__ = ["UtilityContrastiveMarginals"]
