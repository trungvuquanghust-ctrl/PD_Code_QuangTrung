"""Dustbin-style residual calibration for Ours-Final UOT scores."""

from __future__ import annotations

import torch
import torch.nn as nn


class DustbinContrastiveResidualScore(nn.Module):
    """Suppress non-specific positive evidence without changing the UOT plan.

    Full dustbin OT changes both matched mass and the classification objective.
    This module keeps the solved transport plan intact and only subtracts the
    positive ``T-C`` evidence that also looks rejectable under a dustbin-style
    utility gate and weak under class-rival contrast.
    """

    def __init__(
        self,
        *,
        beta: float = 0.50,
        tau: float = 0.25,
        alpha: float = 0.0,
        margin: float = 0.0,
        min_gate: float = 0.05,
        detach_gate: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.min_gate = float(min_gate)
        self.detach_gate = bool(detach_gate)
        self.eps = float(eps)
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("dcr_beta must be in [0, 1]")
        if self.tau <= 0.0:
            raise ValueError("dcr_tau must be positive")
        if self.margin < 0.0:
            raise ValueError("dcr_margin must be non-negative")
        if not 0.0 <= self.min_gate <= 1.0:
            raise ValueError("dcr_min_gate must be in [0, 1]")
        if self.eps <= 0.0:
            raise ValueError("dcr eps must be positive")

    def _threshold_like(self, threshold: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
        active = torch.as_tensor(threshold, device=reference.device, dtype=reference.dtype)
        while active.dim() < reference.dim():
            active = active.unsqueeze(-1)
        return active

    def forward(
        self,
        *,
        cost: torch.Tensor,
        plan: torch.Tensor,
        threshold: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cost.dim() != 5:
            raise ValueError(f"cost must have shape (Nq, Way, Shot, Lq, Ls), got {tuple(cost.shape)}")
        if tuple(plan.shape) != tuple(cost.shape):
            raise ValueError(f"plan shape {tuple(plan.shape)} must match cost shape {tuple(cost.shape)}")

        reference_cost = cost.detach() if self.detach_gate else cost
        active_threshold = self._threshold_like(threshold, cost)
        utility = active_threshold - cost
        reference_utility = active_threshold.detach() - reference_cost
        episode_scale = reference_cost.std(
            dim=(1, 2, 3, 4),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        temperature = (float(self.tau) * episode_scale).clamp_min(self.eps)

        dustbin_gate = torch.sigmoid((reference_utility - float(self.alpha)) / temperature)
        class_best = reference_utility.amax(dim=2).amax(dim=-1)
        way_num = int(cost.shape[1])
        if way_num > 1:
            rival_best = []
            for class_idx in range(way_num):
                rivals = torch.cat(
                    [class_best[:, :class_idx], class_best[:, class_idx + 1 :]],
                    dim=1,
                )
                rival_best.append(rivals.amax(dim=1))
            rival_best = torch.stack(rival_best, dim=1)
            class_margin = class_best - rival_best
        else:
            class_margin = torch.zeros_like(class_best)

        token_temperature = temperature.squeeze(-1).squeeze(-1)
        specificity = torch.sigmoid((class_margin - float(self.margin)) / token_temperature)
        specificity_edge = specificity[:, :, None, :, None].expand_as(cost)
        evidence_gate = dustbin_gate * specificity_edge
        evidence_gate = self.min_gate + (1.0 - self.min_gate) * evidence_gate
        if self.detach_gate:
            evidence_gate = evidence_gate.detach()

        raw_terms = plan * utility
        positive_terms = plan * utility.clamp_min(0.0)
        removed_positive = ((1.0 - evidence_gate) * positive_terms).sum(dim=(-1, -2))
        raw_score = raw_terms.sum(dim=(-1, -2))
        calibrated_score = raw_score - float(self.beta) * removed_positive
        raw_positive = positive_terms.sum(dim=(-1, -2))
        retained_positive = raw_positive - removed_positive
        removed_share = removed_positive / raw_positive.clamp_min(self.eps)
        positive_mass = (plan * (utility > 0.0).to(dtype=plan.dtype)).sum(dim=(-1, -2))
        transported_mass = plan.sum(dim=(-1, -2)).clamp_min(self.eps)

        diagnostics = {
            "dcr/enabled": cost.new_tensor(1.0),
            "dcr/beta": cost.new_tensor(float(self.beta)),
            "dcr/tau": cost.new_tensor(float(self.tau)),
            "dcr/alpha": cost.new_tensor(float(self.alpha)),
            "dcr/margin": cost.new_tensor(float(self.margin)),
            "dcr/min_gate": cost.new_tensor(float(self.min_gate)),
            "dcr/detach_gate": cost.new_tensor(float(self.detach_gate)),
            "dcr/gate_mean": evidence_gate.detach().mean(),
            "dcr/dustbin_gate_mean": dustbin_gate.detach().mean(),
            "dcr/specificity_mean": specificity.detach().mean(),
            "dcr/positive_mass_fraction": (positive_mass / transported_mass).mean().detach(),
            "dcr/removed_positive_share": removed_share.mean().detach(),
            "dcr/removed_positive_mean": removed_positive.detach().mean(),
            "dcr/retained_positive_mean": retained_positive.detach().mean(),
            "dcr/raw_positive_mean": raw_positive.detach().mean(),
            "dcr/shot_score_delta": (calibrated_score - raw_score).detach().mean(),
            "dcr_removed_positive_shot": removed_positive.detach(),
            "dcr_retained_positive_shot": retained_positive.detach(),
            "dcr_removed_positive_share_shot": removed_share.detach(),
            "dcr_gate": evidence_gate.detach(),
            "dcr_specificity": specificity.detach(),
        }
        return calibrated_score, diagnostics


__all__ = ["DustbinContrastiveResidualScore"]
