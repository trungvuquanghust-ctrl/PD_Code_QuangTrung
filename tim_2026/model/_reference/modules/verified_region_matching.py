"""Verified region matching cost gate for Ours-Final token UOT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.modules.cost_evidence_marginals import (
    compute_rival_discriminative_query_evidence,
)


def _validate_odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("vrm_region_kernel_size must be a positive odd integer")
    return kernel_size


class VerifiedRegionMatchingUOT(nn.Module):
    """Suppress accidental low-cost token matches before UOT.

    The gate is parameter-free. It combines:
    - cost-row/column concentration;
    - region-level contextual similarity;
    - optional rival-aware query evidence reused from CostEvidenceMarginals.
    """

    def __init__(
        self,
        *,
        penalty_lambda: float = 0.20,
        concentration_tau: float = 0.25,
        region_tau: float = 0.50,
        region_kernel_size: int = 3,
        use_concentration: bool = True,
        use_region_patch: bool = True,
        use_rival: bool = False,
        rival_tau: float = 0.25,
        rival_threshold: float = 0.0,
        rival_evidence_tau: float = 0.10,
        rival_margin: float = 0.50,
        detach_gate: bool = True,
        ground_cost: str = "euclidean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if float(penalty_lambda) < 0.0:
            raise ValueError("vrm_lambda must be non-negative")
        if float(concentration_tau) <= 0.0:
            raise ValueError("vrm_concentration_tau must be positive")
        if float(region_tau) <= 0.0:
            raise ValueError("vrm_region_tau must be positive")
        if float(rival_tau) <= 0.0:
            raise ValueError("vrm_rival_tau must be positive")
        if float(rival_evidence_tau) <= 0.0:
            raise ValueError("vrm_rival_evidence_tau must be positive")
        if float(rival_margin) < 0.0:
            raise ValueError("vrm_rival_margin must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        ground_cost = str(ground_cost).strip().lower().replace("-", "_")
        if ground_cost not in {"auto", "euclidean", "cosine"}:
            raise ValueError("vrm_ground_cost must be auto/euclidean/cosine")
        self.penalty_lambda = float(penalty_lambda)
        self.concentration_tau = float(concentration_tau)
        self.region_tau = float(region_tau)
        self.region_kernel_size = _validate_odd_kernel(region_kernel_size)
        self.use_concentration = bool(use_concentration)
        self.use_region_patch = bool(use_region_patch)
        self.use_rival = bool(use_rival)
        self.rival_tau = float(rival_tau)
        self.rival_threshold = float(rival_threshold)
        self.rival_evidence_tau = float(rival_evidence_tau)
        self.rival_margin = float(rival_margin)
        self.detach_gate = bool(detach_gate)
        self.ground_cost = ground_cost
        self.eps = float(eps)
        if not (self.use_concentration or self.use_region_patch or self.use_rival):
            raise ValueError("At least one VRM gate must be enabled")

    def _combine_gates(self, components: list[torch.Tensor]) -> torch.Tensor:
        """Parameter-free soft AND that preserves relative evidence scale.

        A direct product collapses when any component has a naturally small
        absolute scale, which turned VRM full into near-uniform cost inflation.
        The geometric mean still requires all enabled checks to agree, but keeps
        the final gate on the same rough scale as its components.
        """
        if not components:
            raise ValueError("VRM requires at least one gate component")
        if len(components) == 1:
            return components[0].clamp(0.0, 1.0)
        stacked = torch.stack([component.clamp(0.0, 1.0) for component in components])
        return stacked.clamp_min(self.eps).log().mean(dim=0).exp().clamp(0.0, 1.0)

    def _region_tokens(
        self,
        tokens: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}"
            )
        batch, token_count, dim = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if token_count != height * width:
            raise ValueError(
                f"spatial_hw={spatial_hw} does not match token count {token_count}"
            )
        token_map = (
            tokens.reshape(batch, height, width, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        padding = self.region_kernel_size // 2
        region_map = F.avg_pool2d(
            token_map,
            kernel_size=self.region_kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        region_tokens = region_map.permute(0, 2, 3, 1).reshape(
            batch,
            token_count,
            dim,
        )
        return F.normalize(region_tokens, p=2, dim=-1, eps=self.eps)

    def _pairwise_cost(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        mode = "euclidean" if self.ground_cost == "auto" else self.ground_cost
        if mode == "cosine":
            q = F.normalize(query_tokens, p=2, dim=-1, eps=self.eps)
            s = F.normalize(support_tokens, p=2, dim=-1, eps=self.eps)
            sim = torch.einsum("qtd,psd->qpts", q, s)
            return (1.0 - sim).clamp_min(0.0)
        q_sq = query_tokens.pow(2).sum(dim=-1)
        s_sq = support_tokens.pow(2).sum(dim=-1)
        dot = torch.einsum("qtd,psd->qpts", query_tokens, support_tokens)
        return (q_sq[:, None, :, None] + s_sq[None, :, None, :] - 2.0 * dot).clamp_min(0.0)

    def _concentration_gate(self, cost: torch.Tensor) -> torch.Tensor:
        tau = max(self.concentration_tau, self.eps)
        row_prob = torch.softmax(-cost / tau, dim=-1)
        col_prob = torch.softmax(-cost / tau, dim=-2)
        support_uniform = 1.0 / float(max(cost.shape[-1], 1))
        query_uniform = 1.0 / float(max(cost.shape[-2], 1))
        row_score = (
            (row_prob - support_uniform) / max(1.0 - support_uniform, self.eps)
        ).clamp(0.0, 1.0)
        col_score = (
            (col_prob - query_uniform) / max(1.0 - query_uniform, self.eps)
        ).clamp(0.0, 1.0)
        return torch.sqrt((row_score * col_score).clamp_min(0.0))

    def _region_patch_gate(
        self,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_pairs = int(way_num) * int(shot_num)
        support_flat = support_tokens.reshape(
            num_pairs,
            support_tokens.shape[-2],
            support_tokens.shape[-1],
        )
        query_region = self._region_tokens(query_tokens, spatial_hw)
        support_region = self._region_tokens(support_flat, spatial_hw)
        region_cost = self._pairwise_cost(query_region, support_region).to(
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        gate = torch.exp(-region_cost / max(self.region_tau, self.eps)).clamp(
            0.0,
            1.0,
        )
        return gate, region_cost

    def _rival_gate(
        self,
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disc_evidence, diag = compute_rival_discriminative_query_evidence(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            tau_evidence=self.rival_evidence_tau,
            rival_margin=self.rival_margin,
            eps=self.eps,
        )
        gate = torch.sigmoid(
            (disc_evidence - self.rival_threshold) / max(self.rival_tau, self.eps)
        ).unsqueeze(-1)
        return gate.expand_as(flat_cost), diag

    def forward(
        self,
        *,
        flat_cost: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (NumQuery, Way*Shot, QueryTokens, SupportTokens), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        way_num = int(way_num)
        shot_num = int(shot_num)
        if num_pairs != way_num * shot_num:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}"
            )
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(
                f"query_tokens shape {tuple(query_tokens.shape)} does not match flat_cost"
            )
        if tuple(support_tokens.shape[:3]) != (way_num, shot_num, support_len):
            raise ValueError(
                f"support_tokens shape {tuple(support_tokens.shape)} does not match flat_cost"
            )

        gate_components: list[torch.Tensor] = []
        payload: dict[str, torch.Tensor] = {
            "verified/enabled": flat_cost.new_tensor(1.0),
            "verified/lambda": flat_cost.new_tensor(self.penalty_lambda),
            "verified/concentration_enabled": flat_cost.new_tensor(
                float(self.use_concentration)
            ),
            "verified/region_patch_enabled": flat_cost.new_tensor(
                float(self.use_region_patch)
            ),
            "verified/rival_enabled": flat_cost.new_tensor(float(self.use_rival)),
            "verified/detach_gate": flat_cost.new_tensor(float(self.detach_gate)),
        }
        concentration_gate = None
        if self.use_concentration:
            concentration_gate = self._concentration_gate(flat_cost)
            gate_components.append(concentration_gate)
            payload["verified/concentration_score_mean"] = (
                concentration_gate.mean().detach()
            )
            payload["verified/concentration_score_max"] = (
                concentration_gate.amax().detach()
            )
        else:
            payload["verified/concentration_score_mean"] = flat_cost.new_tensor(1.0)

        patch_gate = None
        region_cost = None
        if self.use_region_patch:
            patch_gate, region_cost = self._region_patch_gate(
                query_tokens=query_tokens.to(
                    device=flat_cost.device,
                    dtype=flat_cost.dtype,
                ),
                support_tokens=support_tokens.to(
                    device=flat_cost.device,
                    dtype=flat_cost.dtype,
                ),
                flat_cost=flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
            )
            gate_components.append(patch_gate)
            payload["verified/patch_consistency_mean"] = patch_gate.mean().detach()
            payload["verified/patch_consistency_max"] = patch_gate.amax().detach()
            payload["verified_region_cost_matrix"] = region_cost.reshape(
                num_query,
                way_num,
                shot_num,
                query_len,
                support_len,
            ).detach()
        else:
            payload["verified/patch_consistency_mean"] = flat_cost.new_tensor(1.0)

        if concentration_gate is not None and patch_gate is not None:
            payload["verified/region_vs_token_consistency_gap"] = (
                (concentration_gate - patch_gate).clamp_min(0.0).mean().detach()
            )
        else:
            payload["verified/region_vs_token_consistency_gap"] = flat_cost.new_tensor(0.0)

        if self.use_rival:
            rival_gate, rival_diag = self._rival_gate(
                flat_cost=flat_cost,
                way_num=way_num,
                shot_num=shot_num,
            )
            gate_components.append(rival_gate)
            payload["verified/rival_specificity_mean"] = rival_gate.mean().detach()
            for key, value in rival_diag.items():
                payload[f"verified/{key.removeprefix('evidence/')}"] = value
        else:
            payload["verified/rival_specificity_mean"] = flat_cost.new_tensor(1.0)

        gate = self._combine_gates(gate_components)
        gate_flat = gate.reshape(-1)
        payload["verified/gate_q50"] = torch.quantile(gate_flat.detach(), 0.50)
        payload["verified/gate_q90"] = torch.quantile(gate_flat.detach(), 0.90)
        payload["verified/gate_q95"] = torch.quantile(gate_flat.detach(), 0.95)
        if self.detach_gate:
            gate_for_cost = gate.detach()
        else:
            gate_for_cost = gate
        multiplier = 1.0 + self.penalty_lambda * (1.0 - gate_for_cost)
        guided_cost = flat_cost * multiplier

        class_gate = gate.reshape(
            num_query,
            way_num,
            shot_num,
            query_len,
            support_len,
        ).mean(dim=(2, 3, 4))
        if way_num > 1:
            top2 = torch.topk(class_gate, k=2, dim=1).values
            class_gap = top2[:, 0] - top2[:, 1]
        else:
            class_gap = torch.zeros_like(class_gate[:, 0])

        payload.update(
            {
                "verified_region_pair_gate": gate.reshape(
                    num_query,
                    way_num,
                    shot_num,
                    query_len,
                    support_len,
                ).detach(),
                "verified_region_guided_cost_matrix": guided_cost.reshape(
                    num_query,
                    way_num,
                    shot_num,
                    query_len,
                    support_len,
                ),
                "verified/gate_mean": gate.mean().detach(),
                "verified/gate_min": gate.amin().detach(),
                "verified/gate_max": gate.amax().detach(),
                "verified/gate_class_gap": class_gap.mean().detach(),
                "verified/cost_delta_ratio": (
                    (guided_cost.detach() - flat_cost.detach()).abs().mean()
                    / flat_cost.detach().abs().mean().clamp_min(self.eps)
                ),
                "verified/multiplier_mean": multiplier.mean().detach(),
            }
        )
        return guided_cost, payload


__all__ = ["VerifiedRegionMatchingUOT"]
