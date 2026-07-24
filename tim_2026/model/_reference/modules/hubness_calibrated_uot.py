"""Episode-local hubness calibration for Ours-Final UOT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HubnessCalibratedUOT(nn.Module):
    """Calibrate pair costs and marginals with class-excluded local density.

    Common descriptors act as hubs: they are similar to tokens from many
    classes and can therefore receive cheap UOT mass. This module applies a
    CSLS-style two-sided correction inside each episode before transport.
    """

    def __init__(
        self,
        *,
        topk: int = 3,
        temperature: float = 0.25,
        cost_weight: float = 0.35,
        marginal_mix: float = 0.50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(topk) < 1:
            raise ValueError("topk must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(cost_weight) < 0.0:
            raise ValueError("cost_weight must be non-negative")
        if not 0.0 <= float(marginal_mix) <= 1.0:
            raise ValueError("marginal_mix must be in [0, 1]")
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.cost_weight = float(cost_weight)
        self.marginal_mix = float(marginal_mix)
        self.eps = float(eps)

    @staticmethod
    def _topk_mean(values: torch.Tensor, *, k: int, dim: int) -> torch.Tensor:
        effective_k = min(int(k), int(values.shape[dim]))
        return values.topk(effective_k, dim=dim).values.mean(dim=dim)

    def _normalize_with_uniform_fallback(
        self,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = evidence.sum(dim=-1, keepdim=True)
        normalized = evidence / total.clamp_min(self.eps)
        uniform = torch.full_like(evidence, 1.0 / float(max(evidence.shape[-1], 1)))
        probability = torch.where(total > self.eps, normalized, uniform)
        fallback = (total.squeeze(-1) <= self.eps).to(evidence.dtype)
        return probability, fallback

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
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
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(
                "query_tokens must have shape (NumQuery, QueryTokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if tuple(support_tokens.shape[:3]) != (way_num, shot_num, support_len):
            raise ValueError(
                "support_tokens must have shape (Way, Shot, SupportTokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )

        query_unit = F.normalize(query_tokens.detach(), p=2, dim=-1, eps=self.eps)
        support_unit = F.normalize(support_tokens.detach(), p=2, dim=-1, eps=self.eps)
        pair_similarity = torch.einsum(
            "qid,wsjd->qwsij",
            query_unit,
            support_unit,
        )

        if way_num <= 1:
            adjusted_similarity = torch.zeros_like(pair_similarity)
        else:
            query_density_by_shot = self._topk_mean(
                pair_similarity,
                k=self.topk,
                dim=-1,
            )
            query_density_sum = query_density_by_shot.sum(dim=(1, 2), keepdim=False)
            query_class_density = query_density_by_shot.sum(dim=2)
            query_rival_density = (
                query_density_sum[:, None, :] - query_class_density
            ) / float((way_num - 1) * shot_num)

            support_similarity = torch.einsum(
                "wsid,uvjd->wsiuvj",
                support_unit,
                support_unit,
            )
            support_density_by_shot = self._topk_mean(
                support_similarity,
                k=self.topk,
                dim=-1,
            )
            support_density_sum = support_density_by_shot.sum(dim=(-1, -2))
            class_index = torch.arange(way_num, device=flat_cost.device)
            same_class_density = support_density_by_shot[
                class_index,
                :,
                :,
                class_index,
                :,
            ].sum(dim=-1)
            support_rival_density = (
                support_density_sum - same_class_density
            ) / float((way_num - 1) * shot_num)

            adjusted_similarity = (
                2.0 * pair_similarity
                - query_rival_density[:, :, None, :, None]
                - support_rival_density[None, :, :, None, :]
            )

        signed_specificity = torch.tanh(adjusted_similarity / self.temperature)
        positive_specificity = signed_specificity.clamp_min(0.0)

        cost_view = flat_cost.reshape(
            num_query,
            way_num,
            shot_num,
            query_len,
            support_len,
        )
        cost_scale = cost_view.detach().std(
            dim=(-1, -2),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        cost_delta = -self.cost_weight * cost_scale * signed_specificity
        guided_cost = (cost_view + cost_delta).clamp_min(0.0)

        query_evidence = positive_specificity.amax(dim=(-1, 2))
        support_evidence = positive_specificity.amax(dim=-2)
        query_probability, query_fallback = self._normalize_with_uniform_fallback(query_evidence)
        support_probability, support_fallback = self._normalize_with_uniform_fallback(support_evidence)

        query_uniform = torch.full_like(query_probability, 1.0 / float(max(query_len, 1)))
        support_uniform = torch.full_like(support_probability, 1.0 / float(max(support_len, 1)))
        query_probability = (
            (1.0 - self.marginal_mix) * query_uniform
            + self.marginal_mix * query_probability
        )
        support_probability = (
            (1.0 - self.marginal_mix) * support_uniform
            + self.marginal_mix * support_probability
        )
        query_weight = (
            query_probability[:, :, None, :]
            .expand(-1, -1, shot_num, -1)
            .reshape(num_query, num_pairs, query_len)
        )
        support_weight = support_probability.reshape(num_query, num_pairs, support_len)

        query_entropy = -(
            query_weight * query_weight.clamp_min(self.eps).log()
        ).sum(dim=-1)
        support_entropy = -(
            support_weight * support_weight.clamp_min(self.eps).log()
        ).sum(dim=-1)
        diagnostics = {
            "hcuot/enabled": flat_cost.new_tensor(1.0),
            "hcuot/topk": flat_cost.new_tensor(float(self.topk)),
            "hcuot/temperature": flat_cost.new_tensor(self.temperature),
            "hcuot/cost_weight": flat_cost.new_tensor(self.cost_weight),
            "hcuot/marginal_mix": flat_cost.new_tensor(self.marginal_mix),
            "hcuot/adjusted_similarity_mean": adjusted_similarity.mean().detach(),
            "hcuot/adjusted_similarity_std": adjusted_similarity.std(unbiased=False).detach(),
            "hcuot/positive_specificity_mean": positive_specificity.mean().detach(),
            "hcuot/common_pair_share": (
                adjusted_similarity.abs() <= self.temperature * 0.10
            ).to(flat_cost.dtype).mean().detach(),
            "hcuot/cost_delta_ratio": (
                cost_delta.detach().abs().mean()
                / cost_view.detach().abs().mean().clamp_min(self.eps)
            ),
            "hcuot/query_weight_entropy": query_entropy.mean().detach(),
            "hcuot/support_weight_entropy": support_entropy.mean().detach(),
            "hcuot/query_weight_peak": query_weight.amax(dim=-1).mean().detach(),
            "hcuot/support_weight_peak": support_weight.amax(dim=-1).mean().detach(),
            "hcuot/uniform_fallback_query_share": query_fallback.mean().detach(),
            "hcuot/uniform_fallback_support_share": support_fallback.mean().detach(),
        }
        return (
            guided_cost.reshape_as(flat_cost),
            query_weight,
            support_weight,
            positive_specificity,
            diagnostics,
        )


__all__ = ["HubnessCalibratedUOT"]
