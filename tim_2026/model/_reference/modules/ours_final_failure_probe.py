from __future__ import annotations

import math

import torch
from torch import nn


class OursFinalFailureProbe(nn.Module):
    """Label-free diagnostics for the original Ours-Final T*M-C transport path."""

    def __init__(self, *, common_margin: float = 0.10, eps: float = 1e-8) -> None:
        super().__init__()
        if common_margin < 0.0:
            raise ValueError("common_margin must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.common_margin = float(common_margin)
        self.eps = float(eps)

    def _threshold_like_plan(
        self,
        threshold: torch.Tensor | float,
        plan: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(threshold, device=plan.device, dtype=plan.dtype)
        if value.numel() == 1:
            return value.reshape(1, 1, 1, 1, 1)
        if tuple(value.shape) == tuple(plan.shape[:3]):
            return value[..., None, None]
        if tuple(value.shape) == tuple(plan.shape[:2]):
            return value[:, :, None, None, None]
        raise ValueError(
            "threshold must be scalar or match the query/class(/shot) axes, "
            f"got {tuple(value.shape)} for plan {tuple(plan.shape)}"
        )

    def forward(
        self,
        flat_cost: torch.Tensor,
        plan: torch.Tensor,
        *,
        threshold: torch.Tensor | float,
        way_num: int,
        shot_num: int,
        logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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

        expected_plan_shape = (
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
        )
        if tuple(plan.shape) != expected_plan_shape:
            raise ValueError(
                f"plan must have shape {expected_plan_shape}, got {tuple(plan.shape)}"
            )

        with torch.no_grad():
            cost = flat_cost.detach().reshape(expected_plan_shape)
            mass = plan.detach().clamp_min(0.0)
            threshold_plan = self._threshold_like_plan(threshold, mass)
            edge_utility = threshold_plan - cost

            shot_mass = mass.sum(dim=(-1, -2))
            safe_shot_mass = shot_mass.clamp_min(self.eps)
            negative_mask = edge_utility < 0.0
            negative_mass = (mass * negative_mask.to(mass.dtype)).sum(dim=(-1, -2))
            negative_mass_ratio = negative_mass / safe_shot_mass

            positive_utility = (mass * edge_utility.clamp_min(0.0)).sum(dim=(-1, -2))
            negative_utility = (mass * (-edge_utility).clamp_min(0.0)).sum(dim=(-1, -2))
            utility_abs = positive_utility + negative_utility
            harm_share = negative_utility / utility_abs.clamp_min(self.eps)

            class_token_cost = cost.amin(dim=(2, 4))
            if int(way_num) > 1:
                two_best = class_token_cost.topk(k=2, dim=1, largest=False).values
                class_margin = two_best[:, 1] - two_best[:, 0]
                episode_scale = flat_cost.detach().std(
                    dim=(1, 2, 3),
                    unbiased=False,
                ).clamp_min(self.eps)
                common_query = class_margin <= self.common_margin * episode_scale[:, None]
            else:
                common_query = torch.zeros(
                    num_query,
                    query_len,
                    device=mass.device,
                    dtype=torch.bool,
                )
            row_mass = mass.sum(dim=-1)
            common_mass = (
                row_mass * common_query[:, None, None, :].to(row_mass.dtype)
            ).sum(dim=-1)
            common_mass_ratio = common_mass / safe_shot_mass

            dead_query = class_token_cost >= threshold_plan[:, :, 0, 0, 0, None]
            dead_query_ratio = dead_query.to(mass.dtype).mean(dim=-1)

            row_prob = row_mass / row_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            row_entropy = -(
                row_prob * row_prob.clamp_min(self.eps).log()
            ).sum(dim=-1)
            if query_len > 1:
                row_entropy = row_entropy / math.log(float(query_len))
            else:
                row_entropy = torch.zeros_like(row_entropy)
            effective_query_fraction = row_entropy.exp()
            if query_len > 1:
                effective_query_fraction = (
                    torch.exp(row_entropy * math.log(float(query_len))) / float(query_len)
                )

            shot_cost = (mass * cost).sum(dim=(-1, -2))
            shot_utility = (mass * edge_utility).sum(dim=(-1, -2))
            class_mass = shot_mass.mean(dim=2)
            class_cost = shot_cost.mean(dim=2)
            class_utility = shot_utility.mean(dim=2)
            class_cost_per_mass = class_cost / class_mass.clamp_min(self.eps)
            threshold_scalar = threshold_plan.detach().mean()

            predicted = (
                logits.detach().argmax(dim=1)
                if torch.is_tensor(logits) and tuple(logits.shape) == (num_query, int(way_num))
                else class_utility.argmax(dim=1)
            )
            mass_winner = class_mass.argmax(dim=1)
            cost_winner = class_cost.argmin(dim=1)

            class_negative_mass_ratio = negative_mass_ratio.mean(dim=2)
            class_common_mass_ratio = common_mass_ratio.mean(dim=2)
            class_harm_share = harm_share.mean(dim=2)
            class_entropy = row_entropy.mean(dim=2)
            class_effective_query_fraction = effective_query_fraction.mean(dim=2)

            return {
                "ours_probe_negative_utility_mass_ratio": class_negative_mass_ratio,
                "ours_probe_common_mass_ratio": class_common_mass_ratio,
                "ours_probe_harm_share": class_harm_share,
                "ours_probe_dead_query_ratio": dead_query_ratio,
                "ours_probe_query_mass_entropy": class_entropy,
                "ours_probe_effective_query_fraction": class_effective_query_fraction,
                "ours_probe_utility_score": class_utility,
                "ours_probe_mass_score": class_mass,
                "ours_probe_cost_distance": class_cost,
                "ours_probe_cost_only_score": -class_cost,
                "ours_probe_cost_per_mass_score": -class_cost_per_mass,
                "ours_probe_threshold_x0_score": -class_cost,
                "ours_probe_threshold_x0p5_score": (
                    0.5 * threshold_scalar * class_mass - class_cost
                ),
                "ours_probe_threshold_x2_score": (
                    2.0 * threshold_scalar * class_mass - class_cost
                ),
                "ours_probe_threshold_x4_score": (
                    4.0 * threshold_scalar * class_mass - class_cost
                ),
                "ours_probe/enabled": flat_cost.new_tensor(1.0),
                "ours_probe/common_margin": flat_cost.new_tensor(self.common_margin),
                "ours_probe/negative_utility_mass_ratio": class_negative_mass_ratio.mean(),
                "ours_probe/common_mass_ratio": class_common_mass_ratio.mean(),
                "ours_probe/harm_share": class_harm_share.mean(),
                "ours_probe/dead_query_ratio": dead_query_ratio.mean(),
                "ours_probe/query_mass_entropy": class_entropy.mean(),
                "ours_probe/effective_query_fraction": class_effective_query_fraction.mean(),
                "ours_probe/full_mass_winner_agreement": (
                    predicted == mass_winner
                ).to(flat_cost.dtype).mean(),
                "ours_probe/full_cost_winner_agreement": (
                    predicted == cost_winner
                ).to(flat_cost.dtype).mean(),
                "ours_probe/mass_cost_winner_agreement": (
                    mass_winner == cost_winner
                ).to(flat_cost.dtype).mean(),
            }


__all__ = ["OursFinalFailureProbe"]
