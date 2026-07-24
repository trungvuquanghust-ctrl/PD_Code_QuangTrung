"""Rival-conditional ground cost for episodic few-shot transport."""

from __future__ import annotations

import math

import torch
from torch import nn


RIVAL_COST_MODES = frozenset({"class_nll", "edge_advantage"})
RIVAL_COST_SCOPES = frozenset({"always", "eval_only", "noise_test"})


def normalize_rival_cost_mode(value: str | None) -> str:
    name = str(value or "class_nll").strip().lower().replace("-", "_")
    aliases = {
        "nll": "class_nll",
        "query_class": "class_nll",
        "edge": "edge_advantage",
        "signed_edge": "edge_advantage",
        "centered_edge": "edge_advantage",
    }
    name = aliases.get(name, name)
    if name not in RIVAL_COST_MODES:
        raise ValueError(
            f"Unsupported rival cost mode: {value}. "
            f"Expected one of {sorted(RIVAL_COST_MODES)}"
        )
    return name


def normalize_rival_cost_scope(value: str | None) -> str:
    name = str(value or "always").strip().lower().replace("-", "_")
    aliases = {
        "train_and_test": "always",
        "inference": "eval_only",
        "test_only": "eval_only",
        "noise": "noise_test",
        "noise_only": "noise_test",
    }
    name = aliases.get(name, name)
    if name not in RIVAL_COST_SCOPES:
        raise ValueError(
            f"Unsupported rival cost scope: {value}. "
            f"Expected one of {sorted(RIVAL_COST_SCOPES)}"
        )
    return name


class RivalConditionalGroundCost(nn.Module):
    """Penalize locally cheap matches that are not specific to one class.

    The input remains the original query-to-support ground cost. A soft class
    posterior is computed independently for every query token by comparing its
    soft-min cost to all classes. The negative log posterior is then added to
    every edge of that token/class pair before the unchanged UOT solver.
    """

    def __init__(
        self,
        *,
        penalty_weight: float = 0.50,
        temperature: float = 0.25,
        mode: str = "class_nll",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if penalty_weight < 0.0:
            raise ValueError("penalty_weight must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.penalty_weight = float(penalty_weight)
        self.temperature = float(temperature)
        self.mode = normalize_rival_cost_mode(mode)
        self.eps = float(eps)

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (NumQuery, Way*Shot, QueryTokens, SupportTokens), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        way_num = int(way_num)
        shot_num = int(shot_num)
        if way_num <= 1:
            raise ValueError("rival-conditional ground cost requires at least two classes")
        if shot_num <= 0 or num_pairs != way_num * shot_num:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}"
            )

        if self.penalty_weight == 0.0:
            uniform_probability = flat_cost.new_full(
                (num_query, way_num, query_len),
                1.0 / float(way_num),
            )
            zero_penalty = flat_cost.new_zeros(
                num_query,
                way_num,
                query_len,
            )
            return flat_cost, self._payload(
                flat_cost,
                uniform_probability,
                zero_penalty,
                flat_cost.new_zeros(num_query, way_num, query_len),
                edge_advantage=flat_cost.new_zeros(
                    num_query,
                    way_num,
                    shot_num,
                    query_len,
                    support_len,
                ),
            )

        cost = flat_cost.reshape(
            num_query,
            way_num,
            shot_num,
            query_len,
            support_len,
        )
        episode_scale = flat_cost.std(
            dim=(1, 2, 3),
            unbiased=False,
            keepdim=True,
        ).clamp_min(self.eps)
        class_temperature = (
            episode_scale.reshape(num_query, 1, 1) * self.temperature
        ).clamp_min(self.eps)

        flattened_support = cost.permute(0, 1, 3, 2, 4).reshape(
            num_query,
            way_num,
            query_len,
            shot_num * support_len,
        )
        support_count = max(shot_num * support_len, 1)
        class_energy = -class_temperature * (
            torch.logsumexp(
                -flattened_support / class_temperature.unsqueeze(-1),
                dim=-1,
            )
            - math.log(float(support_count))
        )
        log_class_probability = torch.log_softmax(
            -class_energy / class_temperature,
            dim=1,
        )
        class_probability = log_class_probability.exp()
        edge_advantage = flat_cost.new_zeros(cost.shape)
        if self.mode == "class_nll":
            adjustment = (
                self.penalty_weight
                * episode_scale.reshape(num_query, 1, 1)
                * (-log_class_probability)
            )
            guided_cost = cost + adjustment[:, :, None, :, None]
        else:
            scale = episode_scale.reshape(num_query, 1, 1, 1, 1)
            temperature = (scale * self.temperature).clamp_min(self.eps)
            adjustments = []
            advantages = []
            all_classes = torch.arange(way_num, device=flat_cost.device)
            for class_idx in range(way_num):
                rival_cost = cost[:, all_classes != class_idx]
                rival_cost = rival_cost.permute(0, 3, 1, 2, 4).reshape(
                    num_query,
                    query_len,
                    -1,
                )
                rival_count = max(int(rival_cost.shape[-1]), 1)
                rival_reference = -temperature[:, 0, 0, :, 0] * (
                    torch.logsumexp(
                        -rival_cost / temperature[:, 0, 0, :, :],
                        dim=-1,
                    )
                    - math.log(float(rival_count))
                )
                advantage = (
                    rival_reference[:, None, :, None] - cost[:, class_idx]
                )
                signed_shift = (
                    -self.penalty_weight
                    * scale[:, 0]
                    * torch.tanh(advantage / temperature[:, 0])
                )
                advantages.append(advantage)
                adjustments.append(signed_shift)
            edge_advantage = torch.stack(advantages, dim=1)
            edge_adjustment = torch.stack(adjustments, dim=1)
            guided_cost = (cost + edge_adjustment).clamp_min(0.0)
            adjustment = edge_adjustment
        guided_cost = guided_cost.reshape_as(flat_cost)
        return guided_cost, self._payload(
            flat_cost,
            class_probability,
            adjustment,
            class_energy,
            edge_advantage=edge_advantage,
        )

    def _payload(
        self,
        flat_cost: torch.Tensor,
        class_probability: torch.Tensor,
        adjustment: torch.Tensor,
        class_energy: torch.Tensor,
        edge_advantage: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        way_num = int(class_probability.shape[1])
        entropy = -(
            class_probability
            * class_probability.clamp_min(self.eps).log()
        ).sum(dim=1)
        normalized_entropy = entropy / math.log(float(way_num))
        peak = class_probability.amax(dim=1)
        if edge_advantage is None:
            edge_advantage = flat_cost.new_zeros(())
        adjustment_abs_mean = adjustment.abs().mean()
        return {
            "rc_cost_class_probability": class_probability.detach(),
            "rc_cost_penalty": adjustment.detach(),
            "rc_cost_adjustment": adjustment.detach(),
            "rc_cost_class_energy": class_energy.detach(),
            "rc_cost_edge_advantage": edge_advantage.detach(),
            "rc_cost/enabled": flat_cost.new_tensor(1.0),
            "rc_cost/penalty_weight": flat_cost.new_tensor(self.penalty_weight),
            "rc_cost/temperature": flat_cost.new_tensor(self.temperature),
            "rc_cost/mode_id": flat_cost.new_tensor(
                0.0 if self.mode == "class_nll" else 1.0
            ),
            "rc_cost/class_entropy": normalized_entropy.mean().detach(),
            "rc_cost/class_probability_peak": peak.mean().detach(),
            "rc_cost/common_query_share": (
                normalized_entropy > 0.90
            ).to(flat_cost.dtype).mean().detach(),
            "rc_cost/penalty_mean": adjustment.mean().detach(),
            "rc_cost/adjustment_abs_mean": adjustment_abs_mean.detach(),
            "rc_cost/negative_adjustment_share": (
                adjustment < 0.0
            ).to(flat_cost.dtype).mean().detach(),
            "rc_cost/penalty_to_cost_ratio": (
                adjustment_abs_mean
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ).detach(),
        }


__all__ = [
    "RIVAL_COST_MODES",
    "RIVAL_COST_SCOPES",
    "RivalConditionalGroundCost",
    "normalize_rival_cost_mode",
    "normalize_rival_cost_scope",
]
