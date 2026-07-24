"""Hyperbolic Relational Optimal Transport for few-shot classification."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from tim_2026.model._reference.hyperbolic.poincare_ops import (
    PoincareBall,
    get_ball,
    hyperbolic_distance_matrix,
    resolve_hyperbolic_backend,
    safe_project_to_ball,
)
from tim_2026.model._reference.modules.episode_adaptive_mass import EpisodeAdaptiveMass, summarize_hyperbolic_tokens
from tim_2026.model._reference.modules.dustbin_ot import dustbin_transport_diagnostics, dustbin_transport_from_cost
from tim_2026.model._reference.modules.elastic_ot import elastic_capacity_residuals, sinkhorn_elastic_log
from tim_2026.model._reference.modules.crs_marginal import CrossReferencedSelectiveMarginal
from tim_2026.model._reference.modules.ccdm_marginal import CrossClassDiscriminativeMarginal
from tim_2026.model._reference.modules.egsm_marginal import EpisodeGatedShrinkageMarginal
from tim_2026.model._reference.modules.cata import CATA
from tim_2026.model._reference.modules.hrot_transport_projector import HROTTransportProjector, build_hrot_transport_projector
from tim_2026.model._reference.modules.mutual_evidence_attention import MutualEvidenceAttentionMarginal
from tim_2026.model._reference.modules.token_attention_marginal import TokenAttentionMarginal
from tim_2026.model._reference.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    solve_partial_transport,
)
from tim_2026.model._reference.modules.unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    generalized_kl_divergence,
    resolve_ot_backend,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_log_adaptive,
    sinkhorn_unbalanced_pot,
)


HROT_PARTIAL_OT_NATIVE_BACKENDS = {"partial", "partial_ot", "partial_sinkhorn"}
HROT_PARTIAL_OT_POT_BACKENDS = {"partial_pot", "pot_partial"}
HROT_PARTIAL_OT_EXACT_BACKENDS = {"partial_exact", "exact_partial"}
HROT_PARTIAL_OT_BACKENDS = (
    HROT_PARTIAL_OT_NATIVE_BACKENDS
    | HROT_PARTIAL_OT_POT_BACKENDS
    | HROT_PARTIAL_OT_EXACT_BACKENDS
)
HROT_GROUND_COSTS = {"auto", "euclidean", "cosine"}
HROT_ECOT_TRANSPORT_MODES = {"unbalanced", "balanced"}


class HROTFSLResult(dict):
    """Dict-like container that still exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape

    def __getattr__(self, name: str):
        logits = self.get("logits")
        if logits is not None and hasattr(logits, name):
            return getattr(logits, name)
        raise AttributeError(name)


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


def _normalize_hrot_variant_name(variant: str) -> str:
    normalized = str(variant).strip().upper().replace("-", "").replace("_", "")
    if normalized in {"W", "NCET", "JNCET", "NUISANCEDECOUPLED", "JNUISANCE"}:
        return "J_NCET"
    if normalized in {"RL", "RLITE"}:
        return "T"
    if normalized in {"JEGTW", "JE"}:
        return "JE"
    if normalized in {"JECOTNNCS", "ECOTNNCS", "NNCSJECOT"}:
        return "J_ECOT_NNCS"
    if normalized in {"JECOTM2", "ECOTM2", "M2JECOT", "SBECOT", "JECOTSINGLE", "JECOTSINGLEBUDGET"}:
        return "J_ECOT_M2"
    if normalized in {"JECOTCARE", "CAREUOT", "CARE", "JCAREUOT"}:
        return "J_ECOT_CARE"
    if normalized in {"JECOT", "ECOT"}:
        return "J_ECOT"
    if normalized in {"CPECOT"}:
        return "CP_ECOT"
    if normalized in {"JHLM", "JHIERMASS", "JHMASS", "JTAHM"}:
        warnings.warn(
            "HROT variant J_HLM is deprecated and now maps to J_ECOT; "
            "learned hierarchical/token mass is no longer active.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "J_ECOT"
    return normalized


def _parse_ecot_rho_bank(
    rho_bank: str | list[float] | tuple[float, ...],
    base_rho: float,
    *,
    atol: float = 1e-6,
) -> tuple[tuple[float, ...], int]:
    if isinstance(rho_bank, str):
        values = [float(item.strip()) for item in rho_bank.split(",") if item.strip()]
    else:
        values = [float(item) for item in rho_bank]
    if not values:
        raise ValueError("hrot_ecot_rho_bank must contain at least one rho value")
    if not 0.0 < float(base_rho) <= 1.0:
        raise ValueError("hrot_ecot_base_rho must be in (0, 1]")

    unique: list[float] = []
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError("all hrot_ecot_rho_bank values must be in (0, 1]")
        if not any(abs(value - kept) <= atol for kept in unique):
            unique.append(value)
    if not any(abs(float(base_rho) - value) <= atol for value in unique):
        unique.append(float(base_rho))

    unique = sorted(unique)
    base_idx = min(range(len(unique)), key=lambda idx: abs(unique[idx] - float(base_rho)))
    return tuple(unique), int(base_idx)


def _normalize_hrot_ground_cost(ground_cost: str) -> str:
    normalized = str(ground_cost).strip().lower().replace("-", "_")
    aliases = {
        "l2": "euclidean",
        "sqeuclidean": "euclidean",
        "squared_euclidean": "euclidean",
        "legacy": "auto",
        "deepemd": "cosine",
        "deepemd_cosine": "cosine",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in HROT_GROUND_COSTS:
        raise ValueError(f"Unsupported HROT ground cost: {ground_cost}")
    return normalized


def _normalize_ecot_transport_mode(mode: str | None) -> str:
    normalized = "unbalanced" if mode is None else str(mode).strip().lower().replace("-", "_")
    aliases = {
        "uot": "unbalanced",
        "unbalanced_ot": "unbalanced",
        "kl_uot": "unbalanced",
        "ot": "balanced",
        "balanced_ot": "balanced",
        "full_ot": "balanced",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in HROT_ECOT_TRANSPORT_MODES:
        raise ValueError(f"Unsupported ECOT transport mode: {mode}")
    return normalized


def _compute_cross_shot_support_marginals(
    support_tokens: torch.Tensor,
    rho: float,
    beta: float,
    eta: float,
    detach: bool = True,
    distance: str = "auto",
    eps: float = 1e-8,
    min_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate deterministic cross-shot support token masses from same-class repeats.

    Args:
        support_tokens: Support tokens with shape ``[B, Way, Shot, L, D]``.
        rho: Fixed UOT mass budget.
        beta: Positive sharpness for ``exp(-beta * consistency)``.
        eta: Uniform floor mixed into each per-shot token distribution.
        detach: If true, nearest-neighbor consistency is computed from detached tokens.
        distance: ``"cosine"`` for cosine distance, otherwise squared Euclidean.
        eps: Numerical epsilon for normalization.
        min_sigma: Optional lower clamp for sigma before normalization.

    Returns:
        ``(b, consistency, sigma_norm)`` with shape ``[B, Way, Shot, L]``.
    """
    if support_tokens.dim() != 5:
        raise ValueError(
            "support_tokens must have shape [B, Way, Shot, L, D], "
            f"got {tuple(support_tokens.shape)}"
        )
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if float(beta) <= 0.0:
        raise ValueError("beta must be positive")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if float(min_sigma) < 0.0:
        raise ValueError("min_sigma must be non-negative")

    batch_size, way_num, shot_num, token_num, _ = support_tokens.shape
    if token_num <= 0:
        raise ValueError("support_tokens must contain at least one token")

    uniform = support_tokens.new_full(
        (batch_size, way_num, shot_num, token_num),
        1.0 / float(token_num),
    )
    if shot_num == 1:
        consistency = torch.zeros_like(uniform)
        b = uniform * float(rho)
        return b, consistency, uniform

    tokens = support_tokens.detach() if detach else support_tokens
    distance_name = _normalize_hrot_ground_cost(distance)
    if distance_name == "cosine":
        tokens_for_distance = F.normalize(tokens, p=2, dim=-1, eps=float(eps))
        left = tokens_for_distance[:, :, :, None, :, None, :]
        right = tokens_for_distance[:, :, None, :, None, :, :]
        pair_distance = (1.0 - (left * right).sum(dim=-1)).clamp_min(0.0)
    else:
        left = tokens[:, :, :, None, :, None, :]
        right = tokens[:, :, None, :, None, :, :]
        pair_distance = (left - right).pow(2).sum(dim=-1).clamp_min(0.0)

    nearest = pair_distance.amin(dim=-1)
    shot_mask = ~torch.eye(shot_num, device=support_tokens.device, dtype=torch.bool)
    nearest = nearest.masked_fill(~shot_mask.view(1, 1, shot_num, shot_num, 1), 0.0)
    consistency = nearest.sum(dim=3) / float(shot_num - 1)

    sigma = torch.exp(-float(beta) * consistency)
    if min_sigma > 0.0:
        sigma = sigma.clamp_min(float(min_sigma))
    sigma_norm = sigma / sigma.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    mixed = (1.0 - float(eta)) * sigma_norm + float(eta) * uniform
    b = mixed * float(rho)
    return b, consistency.to(dtype=support_tokens.dtype), sigma_norm.to(dtype=support_tokens.dtype)


def compute_fisher_weights(
    support_tokens: torch.Tensor,
    *,
    eps: float = 1e-6,
    clamp: tuple[float, float] = (0.1, 10.0),
) -> torch.Tensor:
    """Compute detached per-episode Fisher feature weights for CARE-UOT."""
    if support_tokens.dim() != 4:
        raise ValueError(
            "support_tokens must have shape (Way, Shot, Tokens, Dim), "
            f"got {tuple(support_tokens.shape)}"
        )
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    clamp_min, clamp_max = float(clamp[0]), float(clamp[1])
    if clamp_min <= 0.0 or clamp_max < clamp_min:
        raise ValueError("clamp must satisfy 0 < min <= max")

    way_num, _, _, embed_dim = support_tokens.shape
    if way_num == 1:
        return support_tokens.new_ones(embed_dim)

    mu_c = support_tokens.mean(dim=(1, 2))
    mu = mu_c.mean(dim=0)
    within = (support_tokens - mu_c[:, None, None, :]).pow(2).mean(dim=(0, 1, 2))
    between = (mu_c - mu).pow(2).mean(dim=0)
    weights = (between + float(eps)) / (within + float(eps))
    weights = weights * (float(embed_dim) / weights.sum().clamp_min(float(eps)))
    return weights.clamp(min=clamp_min, max=clamp_max).detach()


def compute_mdr_loss(
    shot_masses: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Mass discriminability regularizer for CARE-UOT."""
    if shot_masses.dim() == 4:
        batch_size, num_query, way_num, shot_num = shot_masses.shape
        shot_masses = shot_masses.reshape(batch_size * num_query, way_num, shot_num)
    elif shot_masses.dim() == 3:
        way_num = shot_masses.shape[1]
    else:
        raise ValueError(
            "shot_masses must have shape (Nq, Way, Shot) or (Batch, Nq, Way, Shot), "
            f"got {tuple(shot_masses.shape)}"
        )
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")
    labels = query_labels.reshape(-1).to(device=shot_masses.device, dtype=torch.long)
    if labels.shape[0] != shot_masses.shape[0]:
        raise ValueError(
            "query_labels must have one label per query, "
            f"got {labels.shape[0]} labels for {shot_masses.shape[0]} queries"
        )
    if way_num <= 1:
        return shot_masses.new_zeros(())
    if torch.any(labels < 0) or torch.any(labels >= way_num):
        raise ValueError("query_labels contain class indices outside [0, Way)")

    class_mass = shot_masses.mean(dim=-1)
    true_mass = class_mass.gather(dim=1, index=labels[:, None]).squeeze(1)
    true_mask = F.one_hot(labels, num_classes=way_num).bool()
    wrong_mass = class_mass.masked_fill(true_mask, -torch.inf).max(dim=1).values
    return (float(margin) - (true_mass - wrong_mass)).clamp_min(0.0).pow(2).mean()


class EpisodeBudgetController(nn.Module):
    """Small episode-level policy over deterministic fixed-budget UOT experts."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        budget_count: int,
        base_idx: int,
        enable_tau_shot: bool,
        tau_shot_min: float,
        tau_shot_max: float,
        enable_threshold_offset: bool,
        budget_prior_init: str = "base",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("ECOT controller input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("ECOT controller hidden_dim must be positive")
        if budget_count <= 0:
            raise ValueError("ECOT controller budget_count must be positive")
        if not 0 <= base_idx < budget_count:
            raise ValueError("ECOT base_idx must select an entry in the budget bank")
        if tau_shot_min <= 0.0 or tau_shot_max <= tau_shot_min:
            raise ValueError("ECOT tau_shot bounds must satisfy 0 < min < max")
        budget_prior_init = str(budget_prior_init).strip().lower().replace("-", "_")
        if budget_prior_init not in {"base", "uniform"}:
            raise ValueError("ECOT budget_prior_init must be 'base' or 'uniform'")

        self.input_dim = int(input_dim)
        self.budget_count = int(budget_count)
        self.base_idx = int(base_idx)
        self.enable_tau_shot = bool(enable_tau_shot)
        self.tau_shot_min = float(tau_shot_min)
        self.tau_shot_max = float(tau_shot_max)
        self.enable_threshold_offset = bool(enable_threshold_offset)
        self.budget_prior_init = budget_prior_init

        output_dim = self.budget_count
        self.tau_offset = None
        self.threshold_offset = None
        if self.enable_tau_shot:
            self.tau_offset = output_dim
            output_dim += 1
        if self.enable_threshold_offset:
            self.threshold_offset = output_dim
            output_dim += 1

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), output_dim),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("EpisodeBudgetController final layer must be nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            if self.budget_prior_init == "base":
                final.bias[self.base_idx] = 4.0
            if self.tau_offset is not None:
                unit = (1.0 - self.tau_shot_min) / (self.tau_shot_max - self.tau_shot_min)
                unit = min(max(unit, 1e-4), 1.0 - 1e-4)
                final.bias[self.tau_offset] = math.log(unit / (1.0 - unit))
            if self.threshold_offset is not None:
                final.bias[self.threshold_offset] = 0.0

    def forward(self, diagnostics: torch.Tensor) -> dict[str, torch.Tensor]:
        if diagnostics.dim() != 1 or diagnostics.shape[0] != self.input_dim:
            raise ValueError(
                "ECOT diagnostics must be one vector per episode with shape "
                f"({self.input_dim},), got {tuple(diagnostics.shape)}"
            )
        network_dtype = self.network[0].weight.dtype
        raw = self.network(diagnostics.to(dtype=network_dtype))
        raw = raw.to(device=diagnostics.device, dtype=diagnostics.dtype)
        outputs: dict[str, torch.Tensor] = {
            "budget_logits": raw[: self.budget_count],
            "pi_budget": torch.softmax(raw[: self.budget_count], dim=-1),
        }
        if self.tau_offset is not None:
            outputs["raw_tau_shot"] = raw[self.tau_offset]
        if self.threshold_offset is not None:
            outputs["raw_delta_threshold"] = raw[self.threshold_offset]
        return outputs


def mncr_refine(
    D_raw: torch.Tensor,
    temperature: float = 0.5,
    lam: float = 0.5,
) -> torch.Tensor:
    """Mutual Nearest-Neighbor Cost Refinement.

    Refines cost matrix by reducing costs for mutually-agreed matches.
    Tokens with high bidirectional affinity get lower transport cost;
    spurious one-sided matches are left unchanged.

    Args:
        D_raw:       cost matrix, shape [..., Lq, Ls]
        temperature: softmax temperature for soft nearest-neighbor (default 0.5)
        lam:         refinement strength in [0, 1] (default 0.5)

    Returns:
        D_refined:   refined cost matrix, same shape as D_raw
    """
    R = (-D_raw / temperature).softmax(dim=-1)
    C = (-D_raw / temperature).softmax(dim=-2)
    mutual = R * C
    return D_raw * (1.0 - lam * mutual)


class HROTFSL(BaseConv64FewShotModel):
    """Vectorized HROT few-shot model with ablation variants, ECOT routes, and J_NCET."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        use_raw_backbone_tokens: bool = False,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        variant: str = "E",
        eam_hidden_dim: int = 256,
        curvature_init: float = 1.0,
        projection_scale: float = 0.1,
        token_temperature: float = 0.1,
        score_scale: float = 16.0,
        tau_q: float = 0.5,
        tau_c: float = 0.5,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 60,
        sinkhorn_tolerance: float = 1e-5,
        fixed_mass: float = 0.8,
        egtw_tau: float = 1.0,
        egtw_lambda: float = 1.0,
        egtw_eps: float = 1e-8,
        egtw_attention_temperature: float = 0.2,
        egtw_support_similarity_weight: float = 0.5,
        egtw_uniform_mix: float = 0.25,
        egtw_detach_masses: bool = True,
        egtw_learn_tau: bool = False,
        egtw_learn_lambda: bool = False,
        pre_transport_shot_pool: bool = False,
        pre_transport_shot_pool_mode: str = "mean",
        hrot_tsw_enable: bool = False,
        hrot_tsw_share_gate: bool = True,
        hlm_min_mass: float = 0.1,
        hlm_init_mass: float = 0.8,
        hlm_budget_mode: str = "cost",
        hlm_token_mode: str = "cost",
        hlm_token_tau: float = 0.25,
        hlm_budget_hidden_dim: int = 64,
        hlm_token_hidden_dim: int = 64,
        hlm_detach_cost_features: bool = True,
        hlm_allow_mass_grad: bool = True,
        hlm_lambda_rho: float = 0.0,
        hlm_lambda_eff: float = 0.0,
        hlm_eff_margin: float = 0.05,
        hlm_lambda_shot_cov: float = 0.0,
        hlm_shot_entropy_floor: float = 0.35,
        hlm_return_diagnostics: bool = True,
        ecot_rho_bank: str | list[float] | tuple[float, ...] | None = None,
        ecot_base_rho: float | None = None,
        ecot_budget_tau: float = 1.0,
        ecot_max_lambda: float = 1.0,
        ecot_lambda_init: float = -8.0,
        ecot_controller_hidden: int = 32,
        ecot_budget_prior_init: str = "base",
        ecot_uniform_budget_policy: bool = False,
        ecot_enable_tau_shot: bool = True,
        ecot_tau_shot_min: float = 0.5,
        ecot_tau_shot_max: float = 2.0,
        ecot_enable_threshold_offset: bool = False,
        ecot_m2_per_shot_threshold: bool = False,
        ecot_m2_pst_hidden: int = 32,
        ecot_m2_ablate_threshold_mass: bool = True,
        ecot_m2_score_mode: str = "threshold_mass",
        ecot_m2_elastic_sigma: float = 1.0,
        ecot_m2_elastic_probe: bool = False,
        ecot_m2_dustbin_score_init: float = 0.0,
        ecot_m2_cost_per_mass_score: bool = False,
        ecot_m2_cost_per_mass_alpha: float = 1.0,
        ecot_m2_cost_per_mass_detach_mass: bool = True,
        ecot_m2_mass_score_mode: str = "standard",
        ecot_m2_consensus_mass_alpha: float = 1.0,
        ecot_m2_mass_reward_beta: float = 1.0,
        ecot_m2_mass_reward_shot_scaling: str = "none",
        ecot_m2_use_swts: bool = False,
        ecot_m2_swts_temp: float = 1.0,
        ecot_m2_use_aqm: bool = False,
        ecot_m2_tau_aqm: float = 1.0,
        use_mncr: bool = False,
        mncr_temperature: float = 0.5,
        mncr_lam: float = 0.5,
        ecot_identity_reg: float = 1e-4,
        ecot_policy_entropy_reg: float = 1e-3,
        ecot_consensus_tau_mode: str = "fixed",
        ecot_consensus_tau: float = 1.0,
        ecot_enable_nncs_marginal: bool | None = None,
        ecot_nncs_beta: float = 5.0,
        ecot_nncs_eta: float = 0.05,
        ecot_nncs_detach: bool = True,
        ecot_nncs_distance: str = "auto",
        ecot_nncs_min_sigma: float = 1e-6,
        ecot_enable_crs_marginal: bool = False,
        ecot_crs_use_cross_ref: bool = True,
        ecot_crs_use_ssm: bool = True,
        ecot_crs_eta_init: float = 0.30,
        ecot_crs_lambda_cr_init: float = 0.50,
        ecot_crs_tau_ssm_init: float = 0.70,
        ecot_crs_entropy_reg: float = 0.0,
        ecot_crs_side: str = "support",
        ecot_crs_ssm_type: str = "auto",
        ecot_enable_mea_marginal: bool = False,
        ecot_mea_eta_init: float = 0.35,
        ecot_mea_temperature_init: float = 0.70,
        ecot_mea_entropy_reg: float = 0.0,
        ecot_enable_ccdm_marginal: bool = False,
        ecot_ccdm_tau_q_init: float = 0.50,
        ecot_ccdm_tau_b_init: float = 0.50,
        ecot_ccdm_entropy_reg: float = 0.0,
        ecot_ccdm_entropy_shot_weight: float = 0.0,
        ecot_enable_egsm: bool = False,
        ecot_egsm_hidden_dim: int = 32,
        ecot_egsm_candidate_tau_q: float = 1.0,
        ecot_egsm_candidate_tau_b: float = 1.0,
        ecot_egsm_kappa_min: float = 0.05,
        ecot_egsm_kappa_max: float = 0.35,
        ecot_egsm_adaptive_rho: bool = False,
        ecot_egsm_rho_delta_max: float = 0.15,
        ecot_egsm_rho_grad_clip: float = 1.0,
        ecot_egsm_rho_reg_lambda: float = 0.01,
        ecot_enable_ccem_marginal: bool = False,
        ecot_ccem_uniform_mix: float = 0.05,
        ecot_ccem_tau_q: float = 0.25,
        ecot_ccem_tau_s: float = 0.25,
        enable_token_attention_marginal: bool = False,
        tam_hidden_dim: int = 64,
        tam_temperature_init: float = 1.0,
        tam_uniform_floor: float = 0.1,
        tam_detach_weights: bool = False,
        tam_share_qk: bool = True,
        ecot_transport_mode: str | None = None,
        ecot_enable_noise_sink: bool = False,
        ecot_noise_sink_cost_init: float = 1.0,
        ecot_noise_sink_score_penalty: float = 0.0,
        ecot_episode_feature_normalize: bool = False,
        ecot_episode_feature_norm_eps: float = 1e-6,
        ncet_mix_init: float = 0.25,
        ncet_real_penalty_init: float = 0.25,
        ncet_null_penalty_init: float = 0.05,
        ncet_sink_cost_init: float = 1.0,
        min_mass: float = 0.1,
        mass_bonus_init: float = 1.0,
        transport_cost_threshold_init: float | None = None,
        lambda_rho: float = 0.01,
        rho_target: float = 0.8,
        lambda_rho_rank: float = 0.05,
        rho_rank_margin: float = 0.05,
        rho_rank_temperature: float = 0.05,
        lambda_curvature: float = 0.0,
        min_curvature: float = 0.05,
        structure_cost_init: float = 0.05,
        ground_cost: str = "auto",
        cost_margin_aux_weight: float = 0.0,
        cost_margin_aux_margin: float = 0.02,
        eam_mode: str = "compact",
        compact_eam_prior_mix: float = 0.5,
        normalize_euclidean_tokens: bool = True,
        normalize_rho: bool = False,
        eval_use_float64: bool = True,
        hyperbolic_backend: str = "auto",
        ot_backend: str = "native",
        hrot_amp_marginals: bool = False,
        hrot_amp_marginals_tau: float = 1.0,
        hrot_token_center: bool = False,
        hrot_token_center_query: bool = True,
        care_enable_fwec: bool = True,
        care_enable_qesm: bool = True,
        care_enable_mdr: bool = True,
        care_fwec_eps: float = 1e-6,
        care_fwec_w_clamp_min: float = 0.1,
        care_fwec_w_clamp_max: float = 10.0,
        care_qesm_tau_e: float = 1.0,
        care_mdr_margin: float = 0.05,
        care_mdr_lambda: float = 0.1,
        use_cata: bool = False,
        cata_num_anchors: int = 8,
        cata_num_heads: int = 4,
        cata_attn_dropout: float = 0.0,
        projector_use_mlp: bool = False,
        projector_mlp_hidden_dim: int | None = None,
        projector_use_residual: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        variant = _normalize_hrot_variant_name(variant)
        if variant not in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "JE", "J_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE", "CP_ECOT", "J_NCET", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"}:
            raise ValueError(f"Unsupported HROT variant: {variant}")
        self.use_raw_backbone_tokens = bool(use_raw_backbone_tokens)
        self.projector_use_mlp = bool(projector_use_mlp)
        self.projector_use_residual = bool(projector_use_residual)
        self.projector_mlp_hidden_dim = (
            None if projector_mlp_hidden_dim is None else int(projector_mlp_hidden_dim)
        )
        if self.use_raw_backbone_tokens and (self.projector_use_mlp or self.projector_use_residual):
            raise ValueError(
                "projector_use_mlp and projector_use_residual require a learned projector; "
                "disable hrot_use_raw_backbone_tokens"
            )
        if self.projector_mlp_hidden_dim is not None and self.projector_mlp_hidden_dim <= 0:
            raise ValueError("projector_mlp_hidden_dim must be positive when set")
        if self.use_raw_backbone_tokens:
            token_dim = int(hidden_dim)
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if projection_scale <= 0.0:
            raise ValueError("projection_scale must be positive")
        if token_temperature <= 0.0:
            raise ValueError("token_temperature must be positive")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if not 0.0 < fixed_mass <= 1.0:
            raise ValueError("fixed_mass must be in (0, 1]")
        if egtw_tau <= 0.0:
            raise ValueError("egtw_tau must be positive")
        if egtw_lambda < 0.0:
            raise ValueError("egtw_lambda must be non-negative")
        if egtw_eps <= 0.0:
            raise ValueError("egtw_eps must be positive")
        if egtw_attention_temperature <= 0.0:
            raise ValueError("egtw_attention_temperature must be positive")
        if egtw_support_similarity_weight < 0.0:
            raise ValueError("egtw_support_similarity_weight must be non-negative")
        if not 0.0 <= egtw_uniform_mix <= 1.0:
            raise ValueError("egtw_uniform_mix must be in [0, 1]")
        if egtw_detach_masses and (egtw_learn_tau or egtw_learn_lambda):
            raise ValueError("EGTW learnable tau/lambda require egtw_detach_masses=False")
        hlm_budget_mode = str(hlm_budget_mode).strip().lower().replace("-", "_")
        hlm_token_mode = str(hlm_token_mode).strip().lower().replace("-", "_")
        if hlm_budget_mode not in {"geodesic", "cost", "hybrid"}:
            raise ValueError(f"Unsupported HLM budget mode: {hlm_budget_mode}")
        if hlm_token_mode not in {"uniform", "cost", "hybrid"}:
            raise ValueError(f"Unsupported HLM token mode: {hlm_token_mode}")
        if not 0.0 < hlm_min_mass < 1.0:
            raise ValueError("hlm_min_mass must be in (0, 1)")
        if not hlm_min_mass <= hlm_init_mass <= 1.0:
            raise ValueError("hlm_init_mass must be in [hlm_min_mass, 1]")
        if hlm_token_tau <= 0.0:
            raise ValueError("hlm_token_tau must be positive")
        if hlm_budget_hidden_dim <= 0:
            raise ValueError("hlm_budget_hidden_dim must be positive")
        if hlm_token_hidden_dim <= 0:
            raise ValueError("hlm_token_hidden_dim must be positive")
        if hlm_lambda_rho < 0.0:
            raise ValueError("hlm_lambda_rho must be non-negative")
        if hlm_lambda_eff < 0.0:
            raise ValueError("hlm_lambda_eff must be non-negative")
        if hlm_eff_margin < 0.0:
            raise ValueError("hlm_eff_margin must be non-negative")
        if hlm_lambda_shot_cov < 0.0:
            raise ValueError("hlm_lambda_shot_cov must be non-negative")
        if hlm_shot_entropy_floor < 0.0:
            raise ValueError("hlm_shot_entropy_floor must be non-negative")
        if ecot_rho_bank is None:
            if variant == "CP_ECOT":
                ecot_rho_bank = "0.50,0.80,0.95"
            elif variant in {"J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE"}:
                ecot_rho_bank = "0.80"
            else:
                ecot_rho_bank = "0.45,0.60,0.75,0.80,0.90"
        ecot_base_rho = 0.80 if variant in {"CP_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE"} and ecot_base_rho is None else (
            fixed_mass if ecot_base_rho is None else float(ecot_base_rho)
        )
        ecot_rho_bank_values, ecot_base_idx = _parse_ecot_rho_bank(ecot_rho_bank, ecot_base_rho)
        ecot_consensus_tau_mode = str(ecot_consensus_tau_mode).strip().lower().replace("-", "_")
        if ecot_consensus_tau_mode not in {"fixed", "sqrt"}:
            raise ValueError(f"Unsupported ECOT consensus tau mode: {ecot_consensus_tau_mode}")
        if ecot_budget_tau <= 0.0:
            raise ValueError("ecot_budget_tau must be positive")
        if ecot_max_lambda < 0.0:
            raise ValueError("ecot_max_lambda must be non-negative")
        if ecot_controller_hidden <= 0:
            raise ValueError("ecot_controller_hidden must be positive")
        if ecot_tau_shot_min <= 0.0 or ecot_tau_shot_max <= ecot_tau_shot_min:
            raise ValueError("ECOT tau shot bounds must satisfy 0 < min < max")
        if ecot_identity_reg < 0.0:
            raise ValueError("ecot_identity_reg must be non-negative")
        if ecot_policy_entropy_reg < 0.0:
            raise ValueError("ecot_policy_entropy_reg must be non-negative")
        ecot_budget_prior_init = str(ecot_budget_prior_init).strip().lower().replace("-", "_")
        if ecot_budget_prior_init not in {"base", "uniform"}:
            raise ValueError("ecot_budget_prior_init must be 'base' or 'uniform'")
        if ecot_consensus_tau <= 0.0:
            raise ValueError("ecot_consensus_tau must be positive")
        if ecot_nncs_beta <= 0.0:
            raise ValueError("ecot_nncs_beta must be positive")
        if not 0.0 <= ecot_nncs_eta <= 1.0:
            raise ValueError("ecot_nncs_eta must be in [0, 1]")
        ecot_nncs_distance = _normalize_hrot_ground_cost(ecot_nncs_distance)
        if ecot_nncs_min_sigma <= 0.0:
            raise ValueError("ecot_nncs_min_sigma must be positive")
        if not 0.0 < float(ecot_crs_eta_init) < 1.0:
            raise ValueError("ecot_crs_eta_init must be in (0, 1)")
        if not 0.0 < float(ecot_crs_lambda_cr_init) < 1.0:
            raise ValueError("ecot_crs_lambda_cr_init must be in (0, 1)")
        if float(ecot_crs_tau_ssm_init) <= 0.0:
            raise ValueError("ecot_crs_tau_ssm_init must be positive")
        if float(ecot_crs_entropy_reg) < 0.0:
            raise ValueError("ecot_crs_entropy_reg must be non-negative")
        ecot_crs_side = str(ecot_crs_side).strip().lower()
        if ecot_crs_side not in {"support", "both"}:
            raise ValueError(f"Unsupported CRS marginal side: {ecot_crs_side}")
        ecot_crs_ssm_type = str(ecot_crs_ssm_type).strip().lower()
        if ecot_crs_ssm_type not in {"auto", "simple", "mamba"}:
            raise ValueError(f"Unsupported CRS SSM type: {ecot_crs_ssm_type}")
        if not 0.0 < float(ecot_mea_eta_init) < 1.0:
            raise ValueError("ecot_mea_eta_init must be in (0, 1)")
        if float(ecot_mea_temperature_init) <= 0.0:
            raise ValueError("ecot_mea_temperature_init must be positive")
        if float(ecot_mea_entropy_reg) < 0.0:
            raise ValueError("ecot_mea_entropy_reg must be non-negative")
        if float(ecot_ccdm_tau_q_init) <= 0.0:
            raise ValueError("ecot_ccdm_tau_q_init must be positive")
        if float(ecot_ccdm_tau_b_init) <= 0.0:
            raise ValueError("ecot_ccdm_tau_b_init must be positive")
        if float(ecot_ccdm_entropy_reg) < 0.0:
            raise ValueError("ecot_ccdm_entropy_reg must be non-negative")
        if float(ecot_ccdm_entropy_shot_weight) < 0.0:
            raise ValueError("ecot_ccdm_entropy_shot_weight must be non-negative")
        if not 0.0 <= float(ecot_egsm_kappa_min) < float(ecot_egsm_kappa_max) <= 1.0:
            raise ValueError("ecot_egsm_kappa_min and ecot_egsm_kappa_max must satisfy 0 <= min < max <= 1")
        if int(ecot_egsm_hidden_dim) <= 0:
            raise ValueError("ecot_egsm_hidden_dim must be positive")
        if float(ecot_egsm_candidate_tau_q) <= 0.0 or float(ecot_egsm_candidate_tau_b) <= 0.0:
            raise ValueError("ecot_egsm_candidate_tau_q and ecot_egsm_candidate_tau_b must be positive")
        if not 0.0 <= float(ecot_ccem_uniform_mix) < 1.0:
            raise ValueError("ecot_ccem_uniform_mix must satisfy 0 <= mix < 1")
        if float(ecot_ccem_tau_q) <= 0.0 or float(ecot_ccem_tau_s) <= 0.0:
            raise ValueError("ecot_ccem_tau_q and ecot_ccem_tau_s must be positive")
        if int(tam_hidden_dim) <= 0:
            raise ValueError("tam_hidden_dim must be positive")
        if float(tam_temperature_init) <= 0.0:
            raise ValueError("tam_temperature_init must be positive")
        if not 0.0 <= float(tam_uniform_floor) <= 1.0:
            raise ValueError("tam_uniform_floor must be in [0, 1]")
        ecot_transport_mode = _normalize_ecot_transport_mode(ecot_transport_mode)
        if bool(ecot_enable_noise_sink) and float(ecot_noise_sink_cost_init) <= 0.0:
            raise ValueError("ecot_noise_sink_cost_init must be positive")
        if float(ecot_noise_sink_score_penalty) < 0.0:
            raise ValueError("ecot_noise_sink_score_penalty must be non-negative")
        if float(cost_margin_aux_weight) < 0.0:
            raise ValueError("cost_margin_aux_weight must be non-negative")
        if float(cost_margin_aux_margin) < 0.0:
            raise ValueError("cost_margin_aux_margin must be non-negative")
        if bool(ecot_episode_feature_normalize) and variant != "J_ECOT_M2":
            raise ValueError("ecot_episode_feature_normalize is supported only for variant J_ECOT_M2")
        if float(ecot_episode_feature_norm_eps) <= 0.0:
            raise ValueError("ecot_episode_feature_norm_eps must be positive")
        nncs_enabled_by_default = variant == "J_ECOT_NNCS"
        ecot_enable_nncs_marginal = (
            nncs_enabled_by_default
            if ecot_enable_nncs_marginal is None
            else bool(ecot_enable_nncs_marginal)
        )
        uses_nncs_marginal = variant == "J_ECOT_NNCS" and bool(ecot_enable_nncs_marginal)
        if uses_nncs_marginal and bool(pre_transport_shot_pool):
            raise ValueError("J_ECOT_NNCS requires hrot_pre_transport_shot_pool=false")
        if not 0.0 <= ncet_mix_init <= 1.0:
            raise ValueError("ncet_mix_init must be in [0, 1]")
        if ncet_real_penalty_init < 0.0:
            raise ValueError("ncet_real_penalty_init must be non-negative")
        if ncet_null_penalty_init < 0.0:
            raise ValueError("ncet_null_penalty_init must be non-negative")
        if ncet_sink_cost_init <= 0.0:
            raise ValueError("ncet_sink_cost_init must be positive")
        if lambda_rho_rank < 0.0:
            raise ValueError("lambda_rho_rank must be non-negative")
        if rho_rank_margin < 0.0:
            raise ValueError("rho_rank_margin must be non-negative")
        if rho_rank_temperature <= 0.0:
            raise ValueError("rho_rank_temperature must be positive")
        eam_mode = str(eam_mode).strip().lower().replace("-", "_")
        if eam_mode not in {"legacy", "compact"}:
            raise ValueError(f"Unsupported HROT EAM mode: {eam_mode}")
        if not 0.0 <= compact_eam_prior_mix <= 1.0:
            raise ValueError("compact_eam_prior_mix must be in [0, 1]")
        if care_fwec_eps <= 0.0:
            raise ValueError("care_fwec_eps must be positive")
        if care_fwec_w_clamp_min <= 0.0 or care_fwec_w_clamp_max < care_fwec_w_clamp_min:
            raise ValueError("CARE FWEC clamp bounds must satisfy 0 < min <= max")
        if care_qesm_tau_e <= 0.0:
            raise ValueError("care_qesm_tau_e must be positive")
        if care_mdr_margin < 0.0:
            raise ValueError("care_mdr_margin must be non-negative")
        if care_mdr_lambda < 0.0:
            raise ValueError("care_mdr_lambda must be non-negative")

        self.variant = variant
        self.uses_hrot_v = variant == "V"
        self.uses_egtw_mass = variant == "JE"
        self.uses_cp_ecot = variant == "CP_ECOT"
        self.uses_care_uot = variant == "J_ECOT_CARE"
        self.uses_ecot = variant in {"J_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE", "CP_ECOT"}
        self.uses_ecot_nncs_marginal = uses_nncs_marginal
        self.uses_ecot_crs_marginal = bool(ecot_enable_crs_marginal)
        self.uses_ecot_mea_marginal = self.uses_ecot and bool(ecot_enable_mea_marginal)
        self.uses_ecot_ccdm_marginal = self.uses_ecot and bool(ecot_enable_ccdm_marginal)
        self.uses_ecot_egsm_marginal = self.uses_ecot and bool(ecot_enable_egsm)
        self.uses_ecot_ccem_marginal = self.uses_ecot and bool(ecot_enable_ccem_marginal)
        self.uses_token_attention_marginal = self.uses_ecot and bool(
            enable_token_attention_marginal
        )
        self.uses_ecot_noise_sink = self.uses_ecot and bool(ecot_enable_noise_sink)
        if self.uses_ecot_crs_marginal and variant not in {"J_ECOT", "J_ECOT_M2"}:
            raise ValueError("CRS marginal is supported only on the J_ECOT/J_ECOT_M2 fixed-budget path")
        if bool(ecot_enable_mea_marginal) and variant not in {"J_ECOT", "J_ECOT_M2"}:
            raise ValueError("MEA marginal is supported only on the J_ECOT/J_ECOT_M2 fixed-budget path")
        if bool(ecot_enable_ccdm_marginal) and variant not in {"J_ECOT", "J_ECOT_M2"}:
            raise ValueError("CCDM marginal is supported only on the J_ECOT/J_ECOT_M2 fixed-budget path")
        if bool(ecot_enable_egsm) and variant not in {"J_ECOT", "J_ECOT_M2"}:
            raise ValueError("EGSM marginal is supported only on the J_ECOT/J_ECOT_M2 fixed-budget path")
        if bool(ecot_enable_ccem_marginal) and variant not in {"J_ECOT", "J_ECOT_M2"}:
            raise ValueError("CCEM marginal is supported only on the J_ECOT/J_ECOT_M2 fixed-budget path")
        if bool(enable_token_attention_marginal) and variant not in {
            "J_ECOT",
            "J_ECOT_M2",
        }:
            raise ValueError(
                "TokenAttentionMarginal is supported only on the "
                "J_ECOT/J_ECOT_M2 fixed-budget path"
            )
        if self.uses_ecot_crs_marginal and self.uses_ecot_nncs_marginal:
            raise ValueError("CRS and NNCS support marginals are mutually exclusive")
        if self.uses_ecot_mea_marginal and (self.uses_ecot_crs_marginal or self.uses_ecot_nncs_marginal):
            raise ValueError("MEA, CRS, and NNCS marginals are mutually exclusive")
        if self.uses_ecot_ccdm_marginal and (
            self.uses_ecot_mea_marginal
            or self.uses_ecot_crs_marginal
            or self.uses_ecot_nncs_marginal
            or self.uses_ecot_egsm_marginal
        ):
            raise ValueError("CCDM is mutually exclusive with MEA, CRS, NNCS, and EGSM marginals")
        if self.uses_ecot_egsm_marginal and (
            self.uses_ecot_mea_marginal
            or self.uses_ecot_crs_marginal
            or self.uses_ecot_nncs_marginal
            or self.uses_ecot_ccdm_marginal
        ):
            raise ValueError("EGSM is mutually exclusive with MEA, CRS, NNCS, and CCDM marginals")
        if self.uses_ecot_ccem_marginal and (
            self.uses_ecot_mea_marginal
            or self.uses_ecot_crs_marginal
            or self.uses_ecot_nncs_marginal
            or self.uses_ecot_ccdm_marginal
            or self.uses_ecot_egsm_marginal
        ):
            raise ValueError("CCEM marginal is mutually exclusive with MEA, CRS, NNCS, CCDM, and EGSM")
        if self.uses_token_attention_marginal and any(
            (
                self.uses_ecot_mea_marginal,
                self.uses_ecot_crs_marginal,
                self.uses_ecot_nncs_marginal,
                self.uses_ecot_ccdm_marginal,
                self.uses_ecot_egsm_marginal,
                self.uses_ecot_ccem_marginal,
            )
        ):
            raise ValueError(
                "TokenAttentionMarginal is mutually exclusive with MEA, CRS, "
                "NNCS, CCDM, EGSM, and CCEM marginals"
            )
        # Old J_HLM configuration knobs are accepted only for compatibility;
        # the learned hierarchical/token-mass path is no longer activated.
        self.uses_hlm_mass = False
        self.uses_episode_controller = self.uses_ecot
        self.uses_budget_bank = self.uses_ecot
        self.uses_hyperbolic_geometry = variant in {"C", "D", "E"}
        self.uses_unbalanced_transport = variant in {"B", "D", "E", "F", "G", "H", "I", "J", "JE", "J_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE", "CP_ECOT", "J_NCET", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"}
        if self.uses_ecot and ecot_transport_mode == "balanced":
            if any(abs(float(value) - 1.0) > 1e-6 for value in ecot_rho_bank_values):
                raise ValueError("balanced/full OT ECOT mode requires hrot_ecot_rho_bank=1.0")
            self.uses_unbalanced_transport = False
        self.uses_learned_mass = variant in {"E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"}
        self.uses_shot_decomposed_transport = variant in {"G", "H", "I", "J", "JE", "J_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE", "CP_ECOT", "J_NCET", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"}
        self.uses_geodesic_eam = variant in {"H", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "V"}
        self.uses_euclidean_geometric_eam = variant == "I"
        self.uses_reduced_geodesic_eam = variant == "L"
        self.uses_hybrid_ablation_eam = variant == "M"
        self.uses_transport_aware_eam = variant == "O"
        self.uses_cost_threshold_score = variant in {"H", "I", "J", "JE", "J_ECOT", "J_ECOT_M2", "J_ECOT_NNCS", "J_ECOT_CARE", "CP_ECOT", "J_NCET", "L", "M", "N", "O", "P", "Q", "R", "S"}
        self.uses_hybrid_mass_reward = variant == "M"
        self.uses_rho_rank_loss = variant == "N"
        self.uses_hyperbolic_token_attention = variant in {"P", "S"}
        self.uses_geodesic_shot_pooling = variant == "S"
        self.uses_nuisance_decoupled_transport = variant == "J_NCET"
        self.uses_noise_calibrated_transport = variant in {"Q", "R", "T", "U"}
        self.uses_structure_consistent_transport = variant in {"R", "T", "U"}
        self.uses_structural_cover_objective = variant in {"T", "U"}
        self.uses_coverage_regularized_cover = variant == "U"
        self.coverage_penalty_weight = 0.5 if self.uses_coverage_regularized_cover else 0.0
        self.ground_cost = _normalize_hrot_ground_cost(ground_cost)
        self.cost_margin_aux_weight = float(cost_margin_aux_weight)
        self.cost_margin_aux_margin = float(cost_margin_aux_margin)
        self.eam_mode = eam_mode
        self.uses_compact_geodesic_eam = self.eam_mode == "compact" and variant == "H"
        self.compact_eam_prior_mix = float(compact_eam_prior_mix)
        self.normalize_euclidean_tokens = bool(normalize_euclidean_tokens)
        self.normalize_rho = bool(normalize_rho)
        self.eval_use_float64 = bool(eval_use_float64)
        self.hrot_amp_marginals = bool(hrot_amp_marginals)
        self.hrot_amp_marginals_tau = float(hrot_amp_marginals_tau)
        self.hrot_token_center = bool(hrot_token_center)
        self.hrot_token_center_query = bool(hrot_token_center_query)
        self.care_enable_fwec = bool(care_enable_fwec)
        self.care_enable_qesm = bool(care_enable_qesm)
        self.care_enable_mdr = bool(care_enable_mdr)
        self.care_fwec_eps = float(care_fwec_eps)
        self.care_fwec_w_clamp_min = float(care_fwec_w_clamp_min)
        self.care_fwec_w_clamp_max = float(care_fwec_w_clamp_max)
        self.care_qesm_tau_e = float(care_qesm_tau_e)
        self.care_mdr_margin = float(care_mdr_margin)
        self.care_mdr_lambda = float(care_mdr_lambda)
        self.hyperbolic_backend = resolve_hyperbolic_backend(hyperbolic_backend)
        ot_backend_name = str(ot_backend).lower()
        self.uses_partial_transport = ot_backend_name in HROT_PARTIAL_OT_BACKENDS
        if self.uses_partial_transport:
            self.ot_backend = ot_backend_name
            partial_pot_backends = HROT_PARTIAL_OT_POT_BACKENDS | HROT_PARTIAL_OT_EXACT_BACKENDS
            self.partial_backend = "pot" if ot_backend_name in partial_pot_backends else "fast"
            self.partial_exact = ot_backend_name in HROT_PARTIAL_OT_EXACT_BACKENDS
        else:
            self.ot_backend = resolve_ot_backend(ot_backend_name)
            self.partial_backend = "fast"
            self.partial_exact = False
        if self.uses_nuisance_decoupled_transport and self.uses_partial_transport:
            raise ValueError("NCET/W uses KL-relaxed UOT and does not support partial OT backends")
        self.projection_scale = float(projection_scale)
        self.score_scale = float(score_scale)
        self.tau_q = float(tau_q)
        self.tau_c = float(tau_c)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.fixed_mass = float(fixed_mass)
        self.default_egtw_tau = float(egtw_tau)
        self.default_egtw_lambda = float(egtw_lambda)
        self.egtw_eps = float(egtw_eps)
        self.default_egtw_attention_temperature = float(egtw_attention_temperature)
        self.egtw_support_similarity_weight = float(egtw_support_similarity_weight)
        self.egtw_uniform_mix = float(egtw_uniform_mix)
        self.egtw_detach_masses = bool(egtw_detach_masses)
        self.egtw_learn_tau = bool(egtw_learn_tau)
        self.egtw_learn_lambda = bool(egtw_learn_lambda)
        self.pre_transport_shot_pool = bool(pre_transport_shot_pool)
        self.pre_transport_shot_pool_mode = str(pre_transport_shot_pool_mode).strip().lower().replace("-", "_")
        if self.pre_transport_shot_pool_mode not in {"mean", "concat"}:
            raise ValueError("pre_transport_shot_pool_mode must be one of {'mean', 'concat'}")
        self.hrot_tsw_enable = bool(hrot_tsw_enable)
        self.hrot_tsw_share_gate = bool(hrot_tsw_share_gate)
        self.hlm_min_mass = float(hlm_min_mass)
        self.hlm_init_mass = float(hlm_init_mass)
        self.hlm_budget_mode = hlm_budget_mode
        self.hlm_token_mode = hlm_token_mode
        self.hlm_token_tau_value = float(hlm_token_tau)
        self.hlm_detach_cost_features = bool(hlm_detach_cost_features)
        self.hlm_allow_mass_grad = bool(hlm_allow_mass_grad)
        self.hlm_lambda_rho = float(hlm_lambda_rho)
        self.hlm_lambda_eff = float(hlm_lambda_eff)
        self.hlm_eff_margin = float(hlm_eff_margin)
        self.hlm_lambda_shot_cov = float(hlm_lambda_shot_cov)
        self.hlm_shot_entropy_floor = float(hlm_shot_entropy_floor)
        self.hlm_return_diagnostics = bool(hlm_return_diagnostics)
        self.ecot_rho_bank = ecot_rho_bank_values
        self.ecot_base_rho = float(ecot_base_rho)
        self.ecot_base_idx = int(ecot_base_idx)
        self.ecot_budget_tau = float(ecot_budget_tau)
        self.ecot_max_lambda = float(ecot_max_lambda)
        self.ecot_controller_hidden = int(ecot_controller_hidden)
        self.ecot_budget_prior_init = ecot_budget_prior_init
        self.ecot_uniform_budget_policy = bool(ecot_uniform_budget_policy)
        self.ecot_enable_tau_shot = bool(ecot_enable_tau_shot)
        self.ecot_tau_shot_min = float(ecot_tau_shot_min)
        self.ecot_tau_shot_max = float(ecot_tau_shot_max)
        self.ecot_enable_threshold_offset = bool(ecot_enable_threshold_offset)
        self.ecot_m2_per_shot_threshold = bool(ecot_m2_per_shot_threshold) and variant == "J_ECOT_M2"
        self.ecot_m2_pst_hidden = int(ecot_m2_pst_hidden)
        self.ecot_m2_ablate_threshold_mass = bool(ecot_m2_ablate_threshold_mass) and variant == "J_ECOT_M2"
        score_mode = str(ecot_m2_score_mode).strip().lower().replace("-", "_")
        if score_mode not in {"threshold_mass", "uot_energy", "elastic_ot", "dustbin_ot"}:
            raise ValueError(
                "ecot_m2_score_mode must be 'threshold_mass', 'uot_energy', "
                "'elastic_ot', or 'dustbin_ot'"
            )
        self.ecot_m2_score_mode = score_mode if variant == "J_ECOT_M2" else "threshold_mass"
        self.ecot_m2_elastic_sigma = float(ecot_m2_elastic_sigma)
        self.ecot_m2_elastic_probe = bool(ecot_m2_elastic_probe) and variant == "J_ECOT_M2"
        self.ecot_m2_dustbin_score_init = float(ecot_m2_dustbin_score_init)
        if self.ecot_m2_elastic_sigma <= 0.0:
            raise ValueError("ecot_m2_elastic_sigma must be positive")
        self.ecot_m2_cost_per_mass_score = bool(ecot_m2_cost_per_mass_score) and variant == "J_ECOT_M2"
        self.ecot_m2_cost_per_mass_alpha = float(ecot_m2_cost_per_mass_alpha)
        if self.ecot_m2_score_mode == "uot_energy":
            if not self.uses_unbalanced_transport or self.uses_partial_transport:
                raise ValueError(
                    "ecot_m2_score_mode='uot_energy' requires KL-relaxed unbalanced transport"
                )
            if self.ecot_m2_ablate_threshold_mass or self.ecot_m2_cost_per_mass_score:
                raise ValueError(
                    "ecot_m2_score_mode='uot_energy' cannot be combined with cost-only "
                    "or cost-per-mass scoring"
                )
        if self.ecot_m2_score_mode == "elastic_ot":
            elastic_conflicts = {
                "ecot_m2_ablate_threshold_mass": self.ecot_m2_ablate_threshold_mass,
                "ecot_m2_cost_per_mass_score": self.ecot_m2_cost_per_mass_score,
                "ecot_enable_threshold_offset": bool(ecot_enable_threshold_offset),
                "ecot_m2_per_shot_threshold": bool(ecot_m2_per_shot_threshold),
                "ecot_enable_noise_sink": bool(ecot_enable_noise_sink),
                "ecot_m2_use_aqm": bool(ecot_m2_use_aqm),
                "ecot_m2_use_swts": bool(ecot_m2_use_swts),
                "ecot_m2_mass_score_mode": (
                    str(ecot_m2_mass_score_mode).strip().lower().replace("-", "_")
                    != "standard"
                ),
                "ecot_m2_mass_reward_beta": abs(float(ecot_m2_mass_reward_beta) - 1.0) > 1e-8,
                "ecot_m2_mass_reward_shot_scaling": (
                    str(ecot_m2_mass_reward_shot_scaling).strip().lower().replace("-", "_")
                    != "none"
                ),
                "ecot_enable_crs_marginal": bool(ecot_enable_crs_marginal),
                "ecot_enable_mea_marginal": bool(ecot_enable_mea_marginal),
                "ecot_enable_ccdm_marginal": bool(ecot_enable_ccdm_marginal),
                "ecot_enable_egsm": bool(ecot_enable_egsm),
                "ecot_enable_ccem_marginal": bool(ecot_enable_ccem_marginal),
                "enable_token_attention_marginal": bool(
                    enable_token_attention_marginal
                ),
            }
            active_conflicts = [name for name, active in elastic_conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "ecot_m2_score_mode='elastic_ot' defines the complete transport-score "
                    "objective and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if self.ecot_m2_score_mode == "dustbin_ot":
            dustbin_conflicts = {
                "ecot_m2_ablate_threshold_mass": self.ecot_m2_ablate_threshold_mass,
                "ecot_m2_cost_per_mass_score": self.ecot_m2_cost_per_mass_score,
                "ecot_enable_threshold_offset": bool(ecot_enable_threshold_offset),
                "ecot_m2_per_shot_threshold": bool(ecot_m2_per_shot_threshold),
                "ecot_enable_noise_sink": bool(ecot_enable_noise_sink),
                "ecot_m2_use_aqm": bool(ecot_m2_use_aqm),
                "ecot_m2_use_swts": bool(ecot_m2_use_swts),
                "ecot_m2_mass_score_mode": (
                    str(ecot_m2_mass_score_mode).strip().lower().replace("-", "_")
                    != "standard"
                ),
                "ecot_m2_mass_reward_beta": abs(float(ecot_m2_mass_reward_beta) - 1.0) > 1e-8,
                "ecot_m2_mass_reward_shot_scaling": (
                    str(ecot_m2_mass_reward_shot_scaling).strip().lower().replace("-", "_")
                    != "none"
                ),
                "ecot_enable_crs_marginal": bool(ecot_enable_crs_marginal),
                "ecot_enable_mea_marginal": bool(ecot_enable_mea_marginal),
                "ecot_enable_ccdm_marginal": bool(ecot_enable_ccdm_marginal),
                "ecot_enable_egsm": bool(ecot_enable_egsm),
                "ecot_enable_ccem_marginal": bool(ecot_enable_ccem_marginal),
                "enable_token_attention_marginal": bool(
                    enable_token_attention_marginal
                ),
            }
            active_conflicts = [name for name, active in dustbin_conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "ecot_m2_score_mode='dustbin_ot' defines the complete "
                    "transport-score objective and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if self.ecot_m2_ablate_threshold_mass and self.ecot_m2_cost_per_mass_score:
            raise ValueError(
                "ecot_m2_ablate_threshold_mass and ecot_m2_cost_per_mass_score are mutually exclusive on J_ECOT_M2"
            )
        if self.ecot_m2_cost_per_mass_score and self.ecot_m2_cost_per_mass_alpha <= 0.0:
            raise ValueError("ecot_m2_cost_per_mass_alpha must be positive when ecot_m2_cost_per_mass_score is enabled")
        self.ecot_m2_cost_per_mass_detach_mass = (
            bool(ecot_m2_cost_per_mass_detach_mass) and variant == "J_ECOT_M2"
        )
        mass_score_mode = str(ecot_m2_mass_score_mode).strip().lower().replace("-", "_")
        if mass_score_mode not in {"standard", "shot_consensus"}:
            raise ValueError("ecot_m2_mass_score_mode must be 'standard' or 'shot_consensus'")
        self.ecot_m2_mass_score_mode = mass_score_mode if variant == "J_ECOT_M2" else "standard"
        self.ecot_m2_consensus_mass_alpha = float(ecot_m2_consensus_mass_alpha)
        if not 0.0 <= self.ecot_m2_consensus_mass_alpha <= 1.0:
            raise ValueError("ecot_m2_consensus_mass_alpha must be in [0, 1]")
        self.ecot_m2_mass_reward_beta = float(ecot_m2_mass_reward_beta)
        if not 0.0 <= self.ecot_m2_mass_reward_beta <= 1.0:
            raise ValueError("ecot_m2_mass_reward_beta must be in [0, 1]")
        mass_reward_shot_scaling = str(ecot_m2_mass_reward_shot_scaling).strip().lower().replace("-", "_")
        if mass_reward_shot_scaling not in {"none", "multi_shot_beta"}:
            raise ValueError("ecot_m2_mass_reward_shot_scaling must be 'none' or 'multi_shot_beta'")
        self.ecot_m2_mass_reward_shot_scaling = (
            mass_reward_shot_scaling if variant == "J_ECOT_M2" else "none"
        )
        if (
            self.ecot_m2_mass_score_mode != "standard"
            and (self.ecot_m2_ablate_threshold_mass or self.ecot_m2_cost_per_mass_score)
        ):
            raise ValueError(
                "ecot_m2_mass_score_mode is only supported for active threshold-mass J_ECOT_M2 scoring"
            )
        if bool(ecot_m2_use_swts) and variant == "J_ECOT_M2":
            if float(ecot_m2_swts_temp) <= 0.0:
                raise ValueError("ecot_m2_swts_temp must be positive when ecot_m2_use_swts is enabled.")
            if not self.ecot_m2_ablate_threshold_mass or self.ecot_m2_cost_per_mass_score:
                raise ValueError(
                    "ecot_m2_use_swts is only supported on J_ECOT_M2 with ecot_m2_ablate_threshold_mass=True "
                    "and ecot_m2_cost_per_mass_score=False."
                )
        self.ecot_m2_use_swts = (
            bool(ecot_m2_use_swts)
            and variant == "J_ECOT_M2"
            and self.ecot_m2_ablate_threshold_mass
            and not self.ecot_m2_cost_per_mass_score
        )
        self.ecot_m2_swts_temp = float(ecot_m2_swts_temp)
        if bool(ecot_m2_use_aqm) and variant == "J_ECOT_M2":
            if float(ecot_m2_tau_aqm) <= 0.0:
                raise ValueError("ecot_m2_tau_aqm must be positive when ecot_m2_use_aqm is enabled.")
        self.ecot_m2_use_aqm = bool(ecot_m2_use_aqm) and variant == "J_ECOT_M2"
        self.ecot_m2_tau_aqm = float(ecot_m2_tau_aqm)
        self.use_mncr = bool(use_mncr) and variant == "J_ECOT_M2"
        self.mncr_temperature = float(mncr_temperature)
        self.mncr_lam = float(mncr_lam)
        self.ecot_identity_reg = float(ecot_identity_reg)
        self.ecot_policy_entropy_reg = float(ecot_policy_entropy_reg)
        self.ecot_consensus_tau_mode = ecot_consensus_tau_mode
        self.ecot_consensus_tau = float(ecot_consensus_tau)
        self.ecot_enable_nncs_marginal = bool(ecot_enable_nncs_marginal)
        self.ecot_nncs_beta = float(ecot_nncs_beta)
        self.ecot_nncs_eta = float(ecot_nncs_eta)
        self.ecot_nncs_detach = bool(ecot_nncs_detach)
        self.ecot_nncs_distance = ecot_nncs_distance
        self.ecot_nncs_min_sigma = float(ecot_nncs_min_sigma)
        self.ecot_enable_crs_marginal = bool(ecot_enable_crs_marginal)
        self.ecot_crs_use_cross_ref = bool(ecot_crs_use_cross_ref)
        self.ecot_crs_use_ssm = bool(ecot_crs_use_ssm)
        self.ecot_crs_eta_init = float(ecot_crs_eta_init)
        self.ecot_crs_lambda_cr_init = float(ecot_crs_lambda_cr_init)
        self.ecot_crs_tau_ssm_init = float(ecot_crs_tau_ssm_init)
        self.ecot_crs_entropy_reg = float(ecot_crs_entropy_reg)
        self.ecot_crs_side = ecot_crs_side
        self.ecot_crs_ssm_type = ecot_crs_ssm_type
        self.ecot_enable_mea_marginal = bool(ecot_enable_mea_marginal)
        self.ecot_mea_eta_init = float(ecot_mea_eta_init)
        self.ecot_mea_temperature_init = float(ecot_mea_temperature_init)
        self.ecot_mea_entropy_reg = float(ecot_mea_entropy_reg)
        self.ecot_enable_ccdm_marginal = bool(ecot_enable_ccdm_marginal)
        self.ecot_ccdm_tau_q_init = float(ecot_ccdm_tau_q_init)
        self.ecot_ccdm_tau_b_init = float(ecot_ccdm_tau_b_init)
        self.ecot_ccdm_entropy_reg = float(ecot_ccdm_entropy_reg)
        self.ecot_ccdm_entropy_shot_weight = float(ecot_ccdm_entropy_shot_weight)
        self.ecot_enable_egsm = bool(ecot_enable_egsm)
        self.ecot_egsm_hidden_dim = int(ecot_egsm_hidden_dim)
        self.ecot_egsm_candidate_tau_q = float(ecot_egsm_candidate_tau_q)
        self.ecot_egsm_candidate_tau_b = float(ecot_egsm_candidate_tau_b)
        self.ecot_egsm_kappa_min = float(ecot_egsm_kappa_min)
        self.ecot_egsm_kappa_max = float(ecot_egsm_kappa_max)
        self.ecot_egsm_adaptive_rho = bool(ecot_egsm_adaptive_rho) and self.uses_ecot_egsm_marginal
        self.ecot_egsm_rho_delta_max = float(ecot_egsm_rho_delta_max)
        self.ecot_egsm_rho_grad_clip = float(ecot_egsm_rho_grad_clip)
        self.ecot_egsm_rho_reg_lambda = float(ecot_egsm_rho_reg_lambda)
        self.ecot_enable_ccem_marginal = bool(ecot_enable_ccem_marginal)
        self.ecot_ccem_uniform_mix = float(ecot_ccem_uniform_mix)
        self.ecot_ccem_tau_q = float(ecot_ccem_tau_q)
        self.ecot_ccem_tau_s = float(ecot_ccem_tau_s)
        self.enable_token_attention_marginal = bool(
            enable_token_attention_marginal
        )
        self.tam_hidden_dim = int(tam_hidden_dim)
        self.tam_temperature_init = float(tam_temperature_init)
        self.tam_uniform_floor = float(tam_uniform_floor)
        self.tam_detach_weights = bool(tam_detach_weights)
        self.tam_share_qk = bool(tam_share_qk)
        self.ecot_transport_mode = ecot_transport_mode
        self.ecot_noise_sink_cost_init = float(ecot_noise_sink_cost_init)
        self.ecot_noise_sink_score_penalty = float(ecot_noise_sink_score_penalty)
        self.ecot_episode_feature_normalize = bool(ecot_episode_feature_normalize)
        self.ecot_episode_feature_norm_eps = float(ecot_episode_feature_norm_eps)
        self.ecot_display_name = "Episode-Conditioned Optimal Transport J-FSL"
        self.ecot_paper_name = "Episode-Conditioned Mass-Response Transport for Shot-Decomposed Few-Shot Learning"
        self.ncet_mix_init = float(ncet_mix_init)
        self.ncet_real_penalty_init = float(ncet_real_penalty_init)
        self.ncet_null_penalty_init = float(ncet_null_penalty_init)
        self.ncet_sink_cost_init = float(ncet_sink_cost_init)
        self.min_mass = float(min_mass)
        self.lambda_rho = float(lambda_rho)
        self.rho_target = float(rho_target)
        self.lambda_rho_rank = float(lambda_rho_rank)
        self.rho_rank_margin = float(rho_rank_margin)
        self.rho_rank_temperature = float(rho_rank_temperature)
        self.lambda_curvature = float(lambda_curvature)
        self.min_curvature = float(min_curvature)
        self.eps = float(eps)
        self.default_token_temperature = float(token_temperature)

        self.token_dim = int(token_dim)
        self.token_projector = (
            nn.Identity()
            if self.use_raw_backbone_tokens
            else build_hrot_transport_projector(
                int(hidden_dim),
                self.token_dim,
                use_mlp=self.projector_use_mlp,
                use_residual=self.projector_use_residual,
                mlp_hidden_dim=self.projector_mlp_hidden_dim,
            )
        )
        self.use_cata = bool(use_cata)
        self.cata = (
            CATA(
                token_dim=self.token_dim,
                num_anchors=int(cata_num_anchors),
                num_heads=int(cata_num_heads),
                attn_dropout=float(cata_attn_dropout),
            )
            if self.use_cata
            else None
        )
        if self.hrot_tsw_enable:
            if self.hrot_tsw_share_gate:
                self.tsw_gate = nn.Linear(self.token_dim, 1, bias=True)
                nn.init.zeros_(self.tsw_gate.weight)
                nn.init.zeros_(self.tsw_gate.bias)
                self.tsw_gate_q = None
                self.tsw_gate_s = None
            else:
                self.tsw_gate = None
                self.tsw_gate_q = nn.Linear(self.token_dim, 1, bias=True)
                self.tsw_gate_s = nn.Linear(self.token_dim, 1, bias=True)
                nn.init.zeros_(self.tsw_gate_q.weight)
                nn.init.zeros_(self.tsw_gate_q.bias)
                nn.init.zeros_(self.tsw_gate_s.weight)
                nn.init.zeros_(self.tsw_gate_s.bias)
        else:
            self.tsw_gate = None
            self.tsw_gate_q = None
            self.tsw_gate_s = None
        if self.hyperbolic_backend == "geoopt":
            self.manifold = get_ball(float(curvature_init), backend="geoopt", learnable=True)
            self.raw_curvature = None
        else:
            self.manifold = None
            self.raw_curvature = nn.Parameter(torch.tensor(_inverse_softplus(float(curvature_init)), dtype=torch.float32))
        eam_input_dim = None
        if self.uses_transport_aware_eam:
            eam_input_dim = 8
        elif self.uses_reduced_geodesic_eam:
            eam_input_dim = 3
        elif self.uses_geodesic_eam or self.uses_euclidean_geometric_eam:
            eam_input_dim = 4
        self.eam = EpisodeAdaptiveMass(
            embed_dim=token_dim,
            hidden_dim=eam_hidden_dim,
            min_mass=min_mass,
            default_mass=fixed_mass,
            input_dim=eam_input_dim,
        )
        if self.uses_ecot:
            self.register_buffer(
                "ecot_rho_bank_tensor",
                torch.tensor(self.ecot_rho_bank, dtype=torch.float32),
            )
            self.register_buffer(
                "ecot_base_rho_tensor",
                torch.tensor(self.ecot_base_rho, dtype=torch.float32),
            )
            self.register_buffer(
                "ecot_base_idx_tensor",
                torch.tensor(self.ecot_base_idx, dtype=torch.long),
            )
        else:
            self.ecot_rho_bank_tensor = None
            self.ecot_base_rho_tensor = None
            self.ecot_base_idx_tensor = None
        self.raw_ecot_lambda = (
            nn.Parameter(torch.tensor(float(ecot_lambda_init), dtype=torch.float32))
            if self.uses_ecot
            else None
        )
        self.ecot_diagnostic_dim = 11
        self.episode_controller = (
            EpisodeBudgetController(
                input_dim=self.ecot_diagnostic_dim,
                hidden_dim=self.ecot_controller_hidden,
                budget_count=len(self.ecot_rho_bank),
                base_idx=self.ecot_base_idx,
                enable_tau_shot=self.ecot_enable_tau_shot,
                tau_shot_min=self.ecot_tau_shot_min,
                tau_shot_max=self.ecot_tau_shot_max,
                enable_threshold_offset=self.ecot_enable_threshold_offset,
                budget_prior_init=self.ecot_budget_prior_init,
            )
            if self.uses_ecot
            else None
        )
        self.ecot_m2_pst_scorer = (
            nn.Sequential(
                nn.Linear(3, self.ecot_m2_pst_hidden),
                nn.GELU(),
                nn.Linear(self.ecot_m2_pst_hidden, 1),
            )
            if self.ecot_m2_per_shot_threshold
            else None
        )
        if self.ecot_m2_pst_scorer is not None:
            with torch.no_grad():
                self.ecot_m2_pst_scorer[-1].weight.zero_()
                self.ecot_m2_pst_scorer[-1].bias.zero_()
        self.crs_marginal = (
            CrossReferencedSelectiveMarginal(
                embed_dim=token_dim,
                use_cross_ref=self.ecot_crs_use_cross_ref,
                use_ssm=self.ecot_crs_use_ssm,
                eta_init=self.ecot_crs_eta_init,
                lambda_cr_init=self.ecot_crs_lambda_cr_init,
                tau_ssm_init=self.ecot_crs_tau_ssm_init,
                entropy_reg=self.ecot_crs_entropy_reg,
                side=self.ecot_crs_side,
                ssm_type=self.ecot_crs_ssm_type,
                eps=self.eps,
            )
            if self.uses_ecot_crs_marginal
            else None
        )
        self.mea_marginal = (
            MutualEvidenceAttentionMarginal(
                eta_init=self.ecot_mea_eta_init,
                temperature_init=self.ecot_mea_temperature_init,
                entropy_reg=self.ecot_mea_entropy_reg,
                eps=self.eps,
            )
            if self.uses_ecot_mea_marginal
            else None
        )
        self.ccdm_marginal = (
            CrossClassDiscriminativeMarginal(
                tau_q_init=self.ecot_ccdm_tau_q_init,
                tau_b_init=self.ecot_ccdm_tau_b_init,
                entropy_reg=self.ecot_ccdm_entropy_reg,
                eps=self.eps,
            )
            if self.uses_ecot_ccdm_marginal
            else None
        )
        self.egsm_marginal = (
            EpisodeGatedShrinkageMarginal(
                hidden_dim=self.ecot_egsm_hidden_dim,
                candidate_tau_q=self.ecot_egsm_candidate_tau_q,
                candidate_tau_b=self.ecot_egsm_candidate_tau_b,
                kappa_min=self.ecot_egsm_kappa_min,
                kappa_max=self.ecot_egsm_kappa_max,
                eps=self.eps,
                enable_adaptive_rho=self.ecot_egsm_adaptive_rho,
                rho_delta_max=self.ecot_egsm_rho_delta_max,
                rho_grad_clip=self.ecot_egsm_rho_grad_clip,
                rho_reg_lambda=self.ecot_egsm_rho_reg_lambda,
            )
            if self.uses_ecot_egsm_marginal
            else None
        )
        self.token_attention_marginal = (
            TokenAttentionMarginal(
                token_dim=self.token_dim,
                hidden_dim=self.tam_hidden_dim,
                temperature=self.tam_temperature_init,
                uniform_floor=self.tam_uniform_floor,
                detach_weights=self.tam_detach_weights,
                eps=self.eps,
            )
            if self.uses_token_attention_marginal
            else None
        )
        self.token_attention_support_marginal = (
            TokenAttentionMarginal(
                token_dim=self.token_dim,
                hidden_dim=self.tam_hidden_dim,
                temperature=self.tam_temperature_init,
                uniform_floor=self.tam_uniform_floor,
                detach_weights=self.tam_detach_weights,
                eps=self.eps,
            )
            if self.uses_token_attention_marginal and not self.tam_share_qk
            else None
        )
        self.ccdm_entropy_shot_weight = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if self.uses_ecot_ccdm_marginal and self.ecot_ccdm_entropy_shot_weight > 0.0
            else None
        )
        self.q_eam_feature_dim = 5 if self.uses_noise_calibrated_transport else None
        self.q_eam = (
            EpisodeAdaptiveMass(
                embed_dim=token_dim,
                hidden_dim=eam_hidden_dim,
                min_mass=min_mass,
                default_mass=fixed_mass,
                input_dim=self.q_eam_feature_dim,
            )
            if self.uses_noise_calibrated_transport
            else None
        )
        self.hierarchical_transport_mass = None
        if self.uses_hybrid_ablation_eam:
            self.euclidean_eam = EpisodeAdaptiveMass(
                embed_dim=token_dim,
                hidden_dim=eam_hidden_dim,
                min_mass=min_mass,
                default_mass=fixed_mass,
                input_dim=4,
            )
            self.reduced_geodesic_eam = EpisodeAdaptiveMass(
                embed_dim=token_dim,
                hidden_dim=eam_hidden_dim,
                min_mass=min_mass,
                default_mass=fixed_mass,
                input_dim=3,
            )
        else:
            self.euclidean_eam = None
            self.reduced_geodesic_eam = None
        self.raw_token_temperature = (
            nn.Parameter(torch.tensor(_inverse_softplus(float(token_temperature)), dtype=torch.float32))
            if self.uses_hyperbolic_token_attention or self.uses_noise_calibrated_transport
            else None
        )
        self.raw_egtw_tau = (
            nn.Parameter(torch.tensor(_inverse_softplus(float(egtw_tau)), dtype=torch.float32))
            if self.uses_egtw_mass and self.egtw_learn_tau
            else None
        )
        self.raw_egtw_lambda = (
            nn.Parameter(torch.tensor(_inverse_softplus(max(float(egtw_lambda), float(egtw_eps))), dtype=torch.float32))
            if self.uses_egtw_mass and self.egtw_learn_lambda
            else None
        )
        if self.uses_noise_calibrated_transport:
            if self.uses_structure_consistent_transport and structure_cost_init <= 0.0:
                raise ValueError("structure_cost_init must be positive for structure-consistent transport")
            self.query_reliability_weights = nn.Parameter(
                torch.tensor([1.0, -1.0, -0.5, -0.5], dtype=torch.float32)
            )
            self.support_reliability_weights = nn.Parameter(
                torch.tensor([1.0, -1.0, -0.5, -0.5], dtype=torch.float32)
            )
            self.query_token_attention_vector = nn.Parameter(torch.zeros(token_dim, dtype=torch.float32))
            self.support_token_attention_vector = nn.Parameter(torch.zeros(token_dim, dtype=torch.float32))
            self.raw_token_reliability_mix = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
            self.raw_support_consensus_mix = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
            self.raw_hyperbolic_token_prior_mix = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
            self.raw_eam_cross_attention_temperature = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
            )
            self.raw_consensus_temperature = nn.Parameter(
                torch.tensor(_inverse_softplus(float(token_temperature)), dtype=torch.float32)
            )
            self.raw_noise_sink_cost = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_shot_pool_temperature = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_shot_pool_mix = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
            self.raw_q_enhancement_mix = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
            self.raw_structure_cost_weight = (
                nn.Parameter(torch.tensor(_inverse_softplus(float(structure_cost_init)), dtype=torch.float32))
                if self.uses_structure_consistent_transport
                else None
            )
            self.q_shot_pool_scorer = nn.Linear(self.q_eam_feature_dim, 1)
            self.q_threshold_scorer = nn.Linear(self.q_eam_feature_dim, 1)
            with torch.no_grad():
                pool_init = torch.tensor([-1.0, -0.75, -0.10, -0.10, 0.75], dtype=torch.float32)
                self.q_shot_pool_scorer.weight.copy_(pool_init.unsqueeze(0))
                self.q_shot_pool_scorer.bias.zero_()
                self.q_threshold_scorer.weight.zero_()
                self.q_threshold_scorer.bias.zero_()
        else:
            self.query_reliability_weights = None
            self.support_reliability_weights = None
            self.query_token_attention_vector = None
            self.support_token_attention_vector = None
            self.raw_token_reliability_mix = None
            self.raw_support_consensus_mix = None
            self.raw_hyperbolic_token_prior_mix = None
            self.raw_eam_cross_attention_temperature = None
            self.raw_consensus_temperature = None
            self.raw_noise_sink_cost = None
            self.raw_shot_pool_temperature = None
            self.raw_shot_pool_mix = None
            self.raw_q_enhancement_mix = None
            self.raw_structure_cost_weight = None
            self.q_shot_pool_scorer = None
            self.q_threshold_scorer = None
        if self.uses_ecot_noise_sink and self.raw_noise_sink_cost is None:
            self.raw_noise_sink_cost = nn.Parameter(
                torch.tensor(_inverse_softplus(self.ecot_noise_sink_cost_init), dtype=torch.float32)
            )
        if self.uses_nuisance_decoupled_transport:
            ncet_mix = min(max(self.ncet_mix_init, 1e-4), 1.0 - 1e-4)
            self.raw_ncet_mix = nn.Parameter(torch.tensor(math.log(ncet_mix / (1.0 - ncet_mix)), dtype=torch.float32))
            self.raw_ncet_real_penalty = nn.Parameter(
                torch.tensor(_inverse_softplus(max(self.ncet_real_penalty_init, self.eps)), dtype=torch.float32)
            )
            self.raw_ncet_null_penalty = nn.Parameter(
                torch.tensor(_inverse_softplus(max(self.ncet_null_penalty_init, self.eps)), dtype=torch.float32)
            )
            self.raw_noise_sink_cost = nn.Parameter(
                torch.tensor(_inverse_softplus(max(self.ncet_sink_cost_init, self.eps)), dtype=torch.float32)
            )
        else:
            self.raw_ncet_mix = None
            self.raw_ncet_real_penalty = None
            self.raw_ncet_null_penalty = None
        if self.uses_geodesic_shot_pooling:
            self.raw_shot_pool_temperature = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_shot_pool_mix = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
            self.q_shot_pool_scorer = nn.Linear(4, 1)
            with torch.no_grad():
                pool_init = torch.tensor([-1.0, -0.75, -0.10, -0.10], dtype=torch.float32)
                self.q_shot_pool_scorer.weight.copy_(pool_init.unsqueeze(0))
                self.q_shot_pool_scorer.bias.zero_()
        if self.uses_cost_threshold_score:
            threshold_init = (
                float(transport_cost_threshold_init)
                if transport_cost_threshold_init is not None
                else float(mass_bonus_init) / self.score_scale
            )
            if threshold_init <= 0.0:
                raise ValueError("transport_cost_threshold_init must be positive for cost-threshold scoring")
            self.raw_transport_cost_threshold = nn.Parameter(
                torch.tensor(_inverse_softplus(threshold_init), dtype=torch.float32)
            )
            self.mass_bonus = (
                nn.Parameter(torch.tensor(float(mass_bonus_init), dtype=torch.float32))
                if self.uses_hybrid_mass_reward
                else None
            )
        else:
            self.raw_transport_cost_threshold = None
            self.mass_bonus = nn.Parameter(torch.tensor(float(mass_bonus_init), dtype=torch.float32))
        self.ecot_m2_dustbin_score = (
            nn.Parameter(
                torch.tensor(
                    float(self.ecot_m2_dustbin_score_init),
                    dtype=torch.float32,
                )
            )
            if self.ecot_m2_score_mode == "dustbin_ot"
            else None
        )

        if self.uses_hrot_v:
            self.w_q = nn.Parameter(torch.randn(token_dim, dtype=torch.float32) * 0.01)
            self.w_s = nn.Parameter(torch.randn(token_dim, dtype=torch.float32) * 0.01)
            self.raw_tau_token = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.raw_structure_weight = nn.Parameter(
                torch.tensor(_inverse_softplus(max(float(structure_cost_init), 1e-4)), dtype=torch.float32)
            )
            self.raw_threshold_offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.raw_threshold_scale = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_noise_sink_cost = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_transport_cost_threshold = None
            self.mass_bonus = None
        else:
            self.w_q = None
            self.w_s = None
            self.raw_tau_token = None
            self.raw_structure_weight = None
            self.raw_threshold_offset = None
            self.raw_threshold_scale = None

    @property
    def curvature(self) -> torch.Tensor:
        if self.manifold is not None:
            return self.manifold.c.clamp_min(self.eps)
        return F.softplus(self.raw_curvature) + self.eps

    @property
    def curvature_parameter(self) -> torch.nn.Parameter:
        if self.manifold is not None:
            return self.manifold.manifold.isp_c
        return self.raw_curvature

    def _build_ball(self, reference: torch.Tensor) -> PoincareBall:
        if self.manifold is not None:
            return self.manifold
        return get_ball(
            self.curvature.to(device=reference.device, dtype=reference.dtype),
            backend="native",
        )

    @property
    def transport_cost_threshold(self) -> torch.Tensor:
        if self.raw_transport_cost_threshold is not None:
            return F.softplus(self.raw_transport_cost_threshold).clamp_min(self.eps)
        return self.mass_bonus / self.score_scale

    @property
    def token_temperature(self) -> torch.Tensor:
        if self.raw_token_temperature is not None:
            return F.softplus(self.raw_token_temperature).clamp_min(self.eps)
        return self.curvature.new_tensor(self.default_token_temperature)

    @property
    def egtw_tau(self) -> torch.Tensor:
        if self.raw_egtw_tau is not None:
            return F.softplus(self.raw_egtw_tau) + self.egtw_eps
        return self.curvature.new_tensor(self.default_egtw_tau)

    @property
    def egtw_lambda(self) -> torch.Tensor:
        if self.raw_egtw_lambda is not None:
            return F.softplus(self.raw_egtw_lambda)
        return self.curvature.new_tensor(self.default_egtw_lambda)

    @property
    def ecot_lambda(self) -> torch.Tensor:
        if self.raw_ecot_lambda is None:
            return self.curvature.new_tensor(0.0)
        return self.ecot_max_lambda * torch.sigmoid(self.raw_ecot_lambda)

    def _ecot_base_idx_value(self) -> int:
        base_idx_tensor = getattr(self, "ecot_base_idx_tensor", None)
        if torch.is_tensor(base_idx_tensor):
            return int(base_idx_tensor.detach().cpu().item())
        return int(self.ecot_base_idx)

    def _ecot_base_rho_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        base_rho_tensor = getattr(self, "ecot_base_rho_tensor", None)
        if torch.is_tensor(base_rho_tensor):
            return base_rho_tensor.to(device=reference.device, dtype=reference.dtype)
        return reference.new_tensor(float(self.ecot_base_rho))

    @property
    def egtw_attention_temperature(self) -> torch.Tensor:
        return self.curvature.new_tensor(self.default_egtw_attention_temperature)

    @property
    def tau_token(self) -> torch.Tensor:
        if self.raw_tau_token is None:
            return self.token_temperature
        return F.softplus(self.raw_tau_token).clamp_min(self.eps)

    @property
    def consensus_temperature(self) -> torch.Tensor:
        if self.raw_consensus_temperature is not None:
            return F.softplus(self.raw_consensus_temperature).clamp_min(self.eps)
        return self.token_temperature

    @property
    def token_reliability_mix(self) -> torch.Tensor:
        if self.raw_token_reliability_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_token_reliability_mix)

    @property
    def support_consensus_mix(self) -> torch.Tensor:
        if self.raw_support_consensus_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_support_consensus_mix)

    @property
    def hyperbolic_token_prior_mix(self) -> torch.Tensor:
        if self.raw_hyperbolic_token_prior_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_hyperbolic_token_prior_mix)

    @property
    def eam_cross_attention_temperature(self) -> torch.Tensor:
        if self.raw_eam_cross_attention_temperature is None:
            return self.curvature.new_tensor(1.0)
        return F.softplus(self.raw_eam_cross_attention_temperature).clamp_min(self.eps)

    @property
    def noise_sink_cost(self) -> torch.Tensor:
        if self.raw_noise_sink_cost is None:
            return self.curvature.new_tensor(1.0)
        return F.softplus(self.raw_noise_sink_cost).clamp_min(self.eps)

    @staticmethod
    def _standardize_token_scores(scores: torch.Tensor, eps: float) -> torch.Tensor:
        return (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(float(eps))

    def _compute_ccem_marginal_probs(
        self,
        flat_cost: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Class-Contrastive Evidence Marginals from learned episode tokens.

        CCEM does not assume any hand-crafted scalogram morphology. Query tokens
        receive mass only when they match the candidate class more strongly than
        competing classes. Support tokens receive mass when they are consistent
        within the class and/or uncommon across other support classes.
        """
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must be 4D, got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"CCEM expected Way*Shot={int(way_num) * int(shot_num)}, got {num_pairs}")
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(
                "query_tokens must have shape (NumQuery, QueryTokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if tuple(support_tokens.shape[:3]) != (int(way_num), int(shot_num), support_len):
            raise ValueError(
                "support_tokens must have shape (Way, Shot, SupportTokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )

        w = int(way_num)
        k = int(shot_num)
        mix = float(self.ecot_ccem_uniform_mix)
        tau_q = max(float(self.ecot_ccem_tau_q), float(self.eps))
        tau_s = max(float(self.ecot_ccem_tau_s), float(self.eps))

        pair_match = (-flat_cost.detach()).amax(dim=-1).reshape(num_query, w, k, query_len)
        class_match = pair_match.amax(dim=2)
        if w > 1:
            other_best = []
            for class_idx in range(w):
                mask = torch.ones(w, device=flat_cost.device, dtype=torch.bool)
                mask[class_idx] = False
                other_best.append(class_match[:, mask].amax(dim=1))
            other_best = torch.stack(other_best, dim=1)
            class_margin = class_match - other_best
        else:
            class_margin = torch.zeros_like(class_match)
        query_score = class_margin.unsqueeze(2).expand(-1, -1, k, -1).reshape(num_query, num_pairs, query_len)
        query_score = self._standardize_token_scores(query_score, self.eps)
        query_reliability = torch.sigmoid(query_score / tau_q)
        query_weight = mix + (1.0 - mix) * query_reliability
        query_prob = query_weight / query_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        support_norm = F.normalize(support_tokens.detach(), p=2, dim=-1, eps=float(self.eps))
        support_reliability_rows = []
        for class_idx in range(w):
            class_rows = []
            other_pool = None
            if w > 1:
                other_pool = support_norm[
                    torch.arange(w, device=support_norm.device) != class_idx
                ].reshape(-1, support_norm.shape[-1])
            for shot_idx in range(k):
                tokens = support_norm[class_idx, shot_idx]
                if k > 1:
                    same_pool = support_norm[
                        class_idx,
                        torch.arange(k, device=support_norm.device) != shot_idx,
                    ].reshape(-1, support_norm.shape[-1])
                    same_score = tokens.matmul(same_pool.transpose(0, 1)).amax(dim=-1)
                else:
                    same_score = torch.zeros(support_len, device=flat_cost.device, dtype=flat_cost.dtype)
                if other_pool is not None and other_pool.numel() > 0:
                    cross_score = tokens.matmul(other_pool.transpose(0, 1)).amax(dim=-1)
                else:
                    cross_score = torch.zeros_like(same_score)
                raw_score = same_score - cross_score if k > 1 else -cross_score
                raw_score = self._standardize_token_scores(raw_score.unsqueeze(0), self.eps).squeeze(0)
                class_rows.append(torch.sigmoid(raw_score / tau_s))
            support_reliability_rows.append(torch.stack(class_rows, dim=0))
        support_reliability = torch.stack(support_reliability_rows, dim=0).to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_weight = mix + (1.0 - mix) * support_reliability
        support_prob_base = support_weight / support_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        support_prob = support_prob_base.reshape(1, num_pairs, support_len).expand(num_query, -1, -1)

        query_prob_view = query_prob.reshape(num_query, w, k, query_len)
        support_prob_view = support_prob.reshape(num_query, w, k, support_len)
        query_entropy = -(query_prob * query_prob.clamp_min(self.eps).log()).sum(dim=-1).mean()
        support_entropy = -(
            support_prob_base * support_prob_base.clamp_min(self.eps).log()
        ).sum(dim=-1).mean()
        payload = {
            "ecot_ccem_query_pi": query_prob_view,
            "ecot_ccem_support_pi": support_prob_view,
            "ecot_ccem_query_reliability": query_reliability.reshape(num_query, w, k, query_len),
            "ecot_ccem_support_reliability": support_reliability,
            "ecot_ccem_query_entropy": query_entropy,
            "ecot_ccem_support_entropy": support_entropy,
            "ecot_ccem_uniform_mix": flat_cost.new_tensor(mix),
            "ecot_ccem_tau_q": flat_cost.new_tensor(tau_q),
            "ecot_ccem_tau_s": flat_cost.new_tensor(tau_s),
        }
        return query_prob, support_prob, payload

    @property
    def ncet_mix(self) -> torch.Tensor:
        if self.raw_ncet_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_ncet_mix)

    @property
    def ncet_real_penalty(self) -> torch.Tensor:
        if self.raw_ncet_real_penalty is None:
            return self.curvature.new_tensor(0.0)
        return F.softplus(self.raw_ncet_real_penalty)

    @property
    def ncet_null_penalty(self) -> torch.Tensor:
        if self.raw_ncet_null_penalty is None:
            return self.curvature.new_tensor(0.0)
        return F.softplus(self.raw_ncet_null_penalty)

    @property
    def shot_pool_temperature(self) -> torch.Tensor:
        if self.raw_shot_pool_temperature is None:
            return self.curvature.new_tensor(1.0)
        return F.softplus(self.raw_shot_pool_temperature).clamp_min(self.eps)

    @property
    def shot_pool_mix(self) -> torch.Tensor:
        if self.raw_shot_pool_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_shot_pool_mix)

    @property
    def q_enhancement_mix(self) -> torch.Tensor:
        if self.raw_q_enhancement_mix is None:
            return self.curvature.new_tensor(1.0)
        return torch.sigmoid(self.raw_q_enhancement_mix)

    @property
    def structure_cost_weight(self) -> torch.Tensor:
        if self.raw_structure_weight is not None:
            return F.softplus(self.raw_structure_weight).clamp_min(self.eps)
        if self.raw_structure_cost_weight is None:
            return self.curvature.new_tensor(0.0)
        return F.softplus(self.raw_structure_cost_weight).clamp_min(self.eps)

    def _mass_reward_weight(self, reference: torch.Tensor) -> torch.Tensor:
        if self.raw_transport_cost_threshold is not None:
            threshold = self.transport_cost_threshold.to(device=reference.device, dtype=reference.dtype)
            reward = self.score_scale * threshold
            if self.uses_hybrid_mass_reward:
                reward = reward + self.mass_bonus.to(device=reference.device, dtype=reference.dtype)
            return reward
        return self.mass_bonus.to(device=reference.device, dtype=reference.dtype)

    def _ot_backend_code(self, reference: torch.Tensor) -> torch.Tensor:
        if self.uses_partial_transport:
            code = 2.0
        elif self.ot_backend == "pot":
            code = 1.0
        else:
            code = 0.0
        return reference.new_tensor(code)

    def _normalize_rho_budget(self, rho: torch.Tensor) -> torch.Tensor:
        if not self.normalize_rho:
            return rho
        if rho.dim() == 2:
            reduce_dims = (1,)
        elif rho.dim() == 3:
            reduce_dims = (1, 2)
        else:
            raise ValueError(f"rho normalization expects 2D or 3D rho, got {tuple(rho.shape)}")
        rho_mean = rho.mean(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        rho = (rho / rho_mean) * self.fixed_mass
        return rho.clamp(min=self.min_mass, max=1.0)

    def _reshape_query_targets(
        self,
        query_targets: torch.Tensor | None,
        *,
        batch_size: int,
        num_query: int,
    ) -> torch.Tensor | None:
        if query_targets is None:
            return None
        if query_targets.dim() == 1:
            if query_targets.shape[0] != batch_size * num_query:
                raise ValueError(
                    "Flat query_targets must have length batch_size * num_query, "
                    f"got {query_targets.shape[0]} vs {batch_size * num_query}"
                )
            return query_targets.reshape(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError(
            "query_targets must be shaped either (Batch * NumQuery,) or (Batch, NumQuery), "
            f"got {tuple(query_targets.shape)}"
        )

    def _apply_cata(self, projected: torch.Tensor) -> torch.Tensor:
        if self.cata is None:
            return projected
        lead_shape = projected.shape[:-2]
        flat = projected.reshape(-1, projected.shape[-2], projected.shape[-1])
        aggregated = self.cata(flat)
        return aggregated.reshape(*lead_shape, aggregated.shape[-2], aggregated.shape[-1])

    def _project_backbone_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.token_projector, HROTTransportProjector):
            return self.token_projector(tokens)
        return self.token_projector(tokens)

    def _project_backbone_tokens_with_prenorm(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.token_projector, HROTTransportProjector):
            return self.token_projector.forward_with_prenorm(tokens)
        if isinstance(self.token_projector, nn.Identity):
            pre_proj_norms = tokens.norm(p=2, dim=-1).clamp(min=1e-6)
            return self.token_projector(tokens), pre_proj_norms
        ln_out = self.token_projector[0](tokens)
        pre_proj_norms = ln_out.norm(p=2, dim=-1).clamp(min=1e-6)
        projected = self.token_projector[1](ln_out)
        return projected, pre_proj_norms

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        projected = self._apply_cata(projected)
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, spatial_hw

    def _encode_images_with_prenorm(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        prenorm_tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(prenorm_tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            prenorm_tokens = projected
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, prenorm_tokens, spatial_hw

    def _encode_images_with_amp_norms(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Encode images and additionally return per-token pre-projection L2 norms.

        The norms are computed on the LayerNorm output (before the Linear
        projection) and are used by Module A (amplitude-weighted marginals).
        """
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected, pre_proj_norms = self._project_backbone_tokens_with_prenorm(tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            pre_proj_norms = projected.norm(p=2, dim=-1).clamp(min=1e-6)
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, pre_proj_norms, spatial_hw

    @staticmethod
    def _episode_joint_standardize_projected(
        query_proj: torch.Tensor,
        support_proj: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension z-score over all episode tokens (query + support), shared mu/sigma."""
        feat_dim = query_proj.shape[-1]
        pooled = torch.cat(
            [query_proj.reshape(-1, feat_dim), support_proj.reshape(-1, feat_dim)],
            dim=0,
        )
        mu = pooled.mean(dim=0)
        sig = pooled.std(dim=0, unbiased=False) + float(eps)
        return (query_proj - mu) / sig, (support_proj - mu) / sig

    def _euclidean_and_hyperbolic_from_projected(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens

    def _project_tokens_from_images(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        projected = self._apply_cata(projected)
        return projected, spatial_hw

    def _project_and_amp_norms_from_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected, pre_proj_norms = self._project_backbone_tokens_with_prenorm(tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            pre_proj_norms = projected.norm(p=2, dim=-1).clamp(min=1e-6)
        return projected, pre_proj_norms, spatial_hw

    @staticmethod
    def _build_amp_marginals(
        pre_proj_norms: torch.Tensor,
        rho: float,
        tau_w: float,
    ) -> torch.Tensor:
        """Build amplitude-weighted marginals from pre-projection token norms.

        Args:
            pre_proj_norms: ``(N, L)`` per-token L2 norms.
            rho: total mass budget (sum of returned marginal equals rho).
            tau_w: softmax temperature.

        Returns:
            Marginal weights ``(N, L)`` summing to ``rho`` per row.
        """
        norm_sq = pre_proj_norms.pow(2)
        weights = torch.softmax(norm_sq / float(tau_w), dim=-1)
        return rho * weights

    def _euclidean_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = query_tokens.pow(2).sum(dim=-1)
        class_norm = class_tokens.pow(2).sum(dim=-1)
        dot = torch.einsum("qtd,wkd->qwtk", query_tokens, class_tokens)
        cost = query_norm[:, None, :, None] + class_norm[None, :, None, :] - 2.0 * dot
        return cost.clamp_min(0.0)

    def _fisher_weighted_cost(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
        fisher_weight: torch.Tensor,
    ) -> torch.Tensor:
        fisher_weight = fisher_weight.to(device=query_tokens.device, dtype=query_tokens.dtype)
        if fisher_weight.dim() != 1 or fisher_weight.shape[0] != query_tokens.shape[-1]:
            raise ValueError(
                "fisher_weight must have shape (TokenDim,), "
                f"got {tuple(fisher_weight.shape)} for token dim {query_tokens.shape[-1]}"
            )
        query_norm = (query_tokens.pow(2) * fisher_weight).sum(dim=-1)
        class_norm = (class_tokens.pow(2) * fisher_weight).sum(dim=-1)
        dot = torch.einsum("qtd,wkd,d->qwtk", query_tokens, class_tokens, fisher_weight)
        cost = query_norm[:, None, :, None] + class_norm[None, :, None, :] - 2.0 * dot
        return cost.clamp_min(0.0)

    def _cosine_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = F.normalize(query_tokens, p=2, dim=-1, eps=self.eps)
        class_norm = F.normalize(class_tokens, p=2, dim=-1, eps=self.eps)
        similarity = torch.einsum("qtd,wkd->qwtk", query_norm, class_norm)
        return (1.0 - similarity).clamp_min(0.0)

    def _ground_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        if self.ground_cost == "cosine":
            return self._cosine_cost(query_tokens, class_tokens)
        return self._euclidean_cost(query_tokens, class_tokens)

    def _supervised_transport_cost_margin_loss(
        self,
        transport_cost: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Train-only margin on the baseline UOT transported cost.

        This regularizes the learned token geometry without changing the cost
        matrix, Sinkhorn plan, threshold-mass score, or shot pooling at
        inference.
        """
        zero = transport_cost.new_zeros(())
        if not self.training or self.cost_margin_aux_weight <= 0.0 or query_targets is None:
            return zero
        if transport_cost.dim() != 2:
            raise ValueError(
                "transport_cost margin expects class costs shaped (NumQuery, Way), "
                f"got {tuple(transport_cost.shape)}"
            )
        num_query, way_num = transport_cost.shape
        if way_num <= 1:
            return zero
        targets = query_targets.to(device=transport_cost.device, dtype=torch.long).reshape(-1)
        if targets.shape[0] != num_query:
            raise ValueError(
                "query_targets length must match NumQuery for cost-margin aux loss, "
                f"got {targets.shape[0]} vs {num_query}"
            )
        true_cost = transport_cost.gather(1, targets.unsqueeze(1)).squeeze(1)
        true_mask = F.one_hot(targets, num_classes=way_num).to(dtype=torch.bool, device=transport_cost.device)
        wrong_cost = transport_cost.masked_fill(true_mask, torch.inf).amin(dim=1)
        valid = torch.isfinite(wrong_cost)
        if not bool(valid.any()):
            return zero
        margin = transport_cost.new_tensor(float(self.cost_margin_aux_margin))
        return F.relu(margin + true_cost[valid] - wrong_cost[valid]).mean()

    def _apply_tsw(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.hrot_tsw_enable:
            raise RuntimeError("TSW is disabled")
        if query_tokens.dim() != 3 or class_tokens.dim() != 3:
            raise ValueError(
                "TSW expects query/class tokens shaped (N, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)} and {tuple(class_tokens.shape)}"
            )
        if query_tokens.shape[-1] != self.token_dim or class_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "TSW token dim mismatch, "
                f"got {query_tokens.shape[-1]} and {class_tokens.shape[-1]} vs {self.token_dim}"
            )

        if self.hrot_tsw_share_gate:
            if self.tsw_gate is None:
                raise RuntimeError("Shared TSW gate is not initialized")
            w_q = torch.sigmoid(self.tsw_gate(query_tokens))
            w_s = torch.sigmoid(self.tsw_gate(class_tokens))
        else:
            if self.tsw_gate_q is None or self.tsw_gate_s is None:
                raise RuntimeError("Split TSW gates are not initialized")
            w_q = torch.sigmoid(self.tsw_gate_q(query_tokens))
            w_s = torch.sigmoid(self.tsw_gate_s(class_tokens))

        w_outer = w_q.unsqueeze(1) * w_s.transpose(-1, -2).unsqueeze(0)
        return w_outer, w_q, w_s

    def _apply_tsw_to_cost(
        self,
        cost: torch.Tensor,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if not self.hrot_tsw_enable:
            return cost, None

        w_outer, w_q, w_s = self._apply_tsw(query_tokens, class_tokens)
        if w_outer.shape != cost.shape:
            raise ValueError(f"TSW modifier shape {tuple(w_outer.shape)} does not match cost {tuple(cost.shape)}")
        w_outer = w_outer.to(device=cost.device, dtype=cost.dtype)
        return cost * w_outer, {
            "tsw_gate_q": w_q,
            "tsw_gate_s": w_s,
            "tsw_cost_modifier": w_outer,
        }

    def _format_tsw_payload(
        self,
        tsw_payload: dict[str, torch.Tensor] | None,
        *,
        query_count: int | None = None,
        way_num: int | None = None,
        shot_num: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if tsw_payload is None:
            return {}

        gate_q = tsw_payload["tsw_gate_q"]
        gate_s = tsw_payload["tsw_gate_s"]
        cost_modifier = tsw_payload["tsw_cost_modifier"]
        if way_num is not None and shot_num is not None:
            q_count = cost_modifier.shape[0] if query_count is None else int(query_count)
            cost_modifier = cost_modifier.reshape(
                q_count,
                int(way_num),
                int(shot_num),
                cost_modifier.shape[-2],
                cost_modifier.shape[-1],
            )
            gate_s = gate_s.reshape(int(way_num), int(shot_num), gate_s.shape[-2], gate_s.shape[-1])

        return {
            "tsw_gate_q": gate_q,
            "tsw_gate_s": gate_s,
            "tsw_cost_modifier": cost_modifier,
            "tsw_mean_gate_q": gate_q.mean(),
            "tsw_mean_gate_s": gate_s.mean(),
        }

    def _class_ground_cost(
        self,
        query_euclidean: torch.Tensor,
        class_euclidean: torch.Tensor,
        query_hyperbolic: torch.Tensor,
        class_hyperbolic: torch.Tensor,
    ) -> torch.Tensor:
        if self.ground_cost == "auto" and self.uses_hyperbolic_geometry:
            return self._hyperbolic_cost(query_hyperbolic, class_hyperbolic)
        return self._ground_cost(query_euclidean, class_euclidean)

    def _hyperbolic_cost(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens.dtype
        query_cast = query_tokens.to(dtype=calc_dtype)
        class_cast = class_tokens.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast)
        cost = hyperbolic_distance_matrix(query_cast.unsqueeze(1), class_cast.unsqueeze(0), ball)
        return cost.to(dtype=query_tokens.dtype)

    def _build_pairwise_rho(
        self,
        query_tokens_hyp: torch.Tensor,
        class_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_hyp.new_ones(query_tokens_hyp.shape[0], class_tokens_hyp.shape[0])
        if not self.uses_learned_mass:
            return query_tokens_hyp.new_full((query_tokens_hyp.shape[0], class_tokens_hyp.shape[0]), self.fixed_mass)

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        class_cast = class_tokens_hyp.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast)
        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        class_stats = summarize_hyperbolic_tokens(class_cast, ball)
        rho = self.eam.forward_from_stats(query_stats, class_stats)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype)

    def _build_geodesic_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_hyp.new_ones(query_tokens_hyp.shape[0], *support_tokens_hyp.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_hyp.new_full(
                (query_tokens_hyp.shape[0], *support_tokens_hyp.shape[:2]),
                self.fixed_mass,
            )

        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        features = (
            self._build_compact_geodesic_eam_features(geodesic_features)
            if self.uses_compact_geodesic_eam
            else geodesic_features
        )
        rho = self.eam.forward_features(features)
        rho = self._apply_compact_geodesic_mass_prior(rho, features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype)

    def _build_geodesic_eam_features(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        support_cast = support_tokens_hyp.to(dtype=calc_dtype)
        way_num, shot_num = support_cast.shape[:2]
        ball = self._build_ball(query_cast)

        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        flat_support = support_cast.reshape(way_num * shot_num, support_cast.shape[-2], support_cast.shape[-1])
        shot_stats = summarize_hyperbolic_tokens(flat_support, ball)
        shot_means = shot_stats.mean_hyp.reshape(way_num, shot_num, -1)
        shot_variance = shot_stats.variance.reshape(way_num, shot_num)
        class_support = support_cast.reshape(way_num, shot_num * support_cast.shape[-2], support_cast.shape[-1])
        class_stats = summarize_hyperbolic_tokens(class_support, ball)

        mean_distance = ball.dist(
            query_stats.mean_hyp[:, None, None, :],
            shot_means[None, :, :, :],
        )
        shot_spread = ball.dist(
            shot_means,
            class_stats.mean_hyp[:, None, :],
        )[None, :, :].expand_as(mean_distance)
        query_variance = query_stats.variance[:, None, None].expand_as(mean_distance)
        support_variance = shot_variance[None, :, :].expand_as(mean_distance)
        if self.uses_reduced_geodesic_eam:
            return torch.stack([mean_distance, query_variance, support_variance], dim=-1)
        return torch.stack([mean_distance, shot_spread, query_variance, support_variance], dim=-1)

    def _build_q_geodesic_eam_features(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        distance = geodesic_features[..., 0]
        inv_distance = distance.clamp_min(self.eps).reciprocal()
        temperature = self.eam_cross_attention_temperature.to(
            device=inv_distance.device,
            dtype=inv_distance.dtype,
        )
        cross_attention = torch.softmax(inv_distance / temperature, dim=-1) * float(distance.shape[-1])
        q_features = torch.cat([geodesic_features, cross_attention.unsqueeze(-1)], dim=-1)
        return q_features, geodesic_features

    def _build_euclidean_rho_per_shot(
        self,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_euc.new_ones(query_tokens_euc.shape[0], *support_tokens_euc.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_euc.new_full(
                (query_tokens_euc.shape[0], *support_tokens_euc.shape[:2]),
                self.fixed_mass,
            )

        features = self._build_euclidean_eam_features(query_tokens_euc, support_tokens_euc)
        rho = self.eam.forward_features(features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_euc.dtype)

    def _build_hybrid_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_euc.new_ones(query_tokens_euc.shape[0], *support_tokens_euc.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_euc.new_full(
                (query_tokens_euc.shape[0], *support_tokens_euc.shape[:2]),
                self.fixed_mass,
            )
        if self.euclidean_eam is None or self.reduced_geodesic_eam is None:
            raise RuntimeError("Hybrid HROT variant requires Euclidean and reduced-geodesic EAM heads")

        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        euclidean_features = self._build_euclidean_eam_features(query_tokens_euc, support_tokens_euc)
        reduced_geodesic_features = geodesic_features[..., [0, 2, 3]]
        rho_geodesic = self.eam.forward_features(geodesic_features)
        rho_euclidean = self.euclidean_eam.forward_features(euclidean_features)
        rho_reduced = self.reduced_geodesic_eam.forward_features(reduced_geodesic_features)
        rho_fixed = torch.full_like(rho_geodesic, self.fixed_mass)
        rho = torch.stack([rho_geodesic, rho_euclidean, rho_reduced, rho_fixed], dim=0).mean(dim=0)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_euc.dtype)

    def _build_euclidean_eam_features(
        self,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        query_mean = query_tokens_euc.mean(dim=-2)
        way_num, shot_num = support_tokens_euc.shape[:2]
        flat_support = support_tokens_euc.reshape(way_num * shot_num, support_tokens_euc.shape[-2], support_tokens_euc.shape[-1])
        shot_mean = flat_support.mean(dim=-2).reshape(way_num, shot_num, -1)
        class_support = support_tokens_euc.reshape(way_num, shot_num * support_tokens_euc.shape[-2], support_tokens_euc.shape[-1])
        class_mean = class_support.mean(dim=-2)

        query_variance = (query_tokens_euc - query_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1)
        flat_shot_mean = shot_mean.reshape(way_num * shot_num, -1)
        shot_variance = (flat_support - flat_shot_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1)
        shot_variance = shot_variance.reshape(way_num, shot_num)

        mean_distance = torch.linalg.vector_norm(
            query_mean[:, None, None, :] - shot_mean[None, :, :, :],
            dim=-1,
        )
        shot_spread = torch.linalg.vector_norm(
            shot_mean - class_mean[:, None, :],
            dim=-1,
        )[None, :, :].expand_as(mean_distance)
        query_variance = query_variance[:, None, None].expand_as(mean_distance)
        support_variance = shot_variance[None, :, :].expand_as(mean_distance)
        return torch.stack([mean_distance, shot_spread, query_variance, support_variance], dim=-1)

    def _build_transport_probe_features(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_query, num_pairs, query_tokens, support_tokens = flat_cost.shape
        if num_pairs != way_num * shot_num:
            raise ValueError(
                "flat_cost pair dimension must equal way_num * shot_num, "
                f"got {num_pairs} vs {way_num * shot_num}"
            )

        fixed_rho = flat_cost.new_full((num_query, num_pairs), self.fixed_mass)
        with torch.no_grad():
            probe_plan, probe_cost, probe_mass = self._transport_match(flat_cost.detach(), fixed_rho)
            plan_mass = probe_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
            plan_prob = probe_plan / plan_mass
            probe_entropy = -(plan_prob * plan_prob.clamp_min(self.eps).log()).sum(dim=(-1, -2))
            entropy_scale = math.log(float(max(2, query_tokens * support_tokens)))
            probe_entropy = probe_entropy / entropy_scale
            min_token_cost = flat_cost.detach().amin(dim=(-1, -2))

        features = torch.stack(
            [
                probe_cost,
                probe_mass,
                probe_entropy,
                min_token_cost,
            ],
            dim=-1,
        ).reshape(num_query, way_num, shot_num, 4)
        payload = {
            "transport_probe_cost": probe_cost.reshape(num_query, way_num, shot_num),
            "transport_probe_mass": probe_mass.reshape(num_query, way_num, shot_num),
            "transport_probe_entropy": probe_entropy.reshape(num_query, way_num, shot_num),
            "transport_probe_min_cost": min_token_cost.reshape(num_query, way_num, shot_num),
        }
        return features.to(dtype=flat_cost.dtype), payload

    def _build_transport_aware_geodesic_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        flat_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        way_num, shot_num = support_tokens_hyp.shape[:2]
        transport_features, probe_payload = self._build_transport_probe_features(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
        )
        features = torch.cat(
            [
                geodesic_features,
                transport_features.to(device=geodesic_features.device, dtype=geodesic_features.dtype),
            ],
            dim=-1,
        )
        rho = self.eam.forward_features(features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype), features, probe_payload

    def _standardize_over_tokens(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-2, keepdim=True)
        std = values.std(dim=-2, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (values - mean) / std

    def _standardize_over_shots(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-2, keepdim=True)
        std = values.std(dim=-2, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (values - mean) / std

    def _standardize_over_candidates(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() < 4:
            raise ValueError(
                "candidate standardization expects values shaped "
                "(NumQuery, Way, Shot, FeatureDim)"
            )
        mean = values.mean(dim=(1, 2), keepdim=True)
        std = values.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(self.eps)
        return (values - mean) / std

    def _build_compact_geodesic_eam_features(self, geodesic_features: torch.Tensor) -> torch.Tensor:
        if geodesic_features.shape[-1] != 4:
            raise ValueError(
                "compact geodesic EAM expects raw features "
                "[mean_distance, shot_spread, query_variance, support_variance]"
            )
        mean_distance = geodesic_features[..., 0]
        shot_spread = geodesic_features[..., 1]
        query_variance = geodesic_features[..., 2].clamp_min(0.0)
        support_variance = geodesic_features[..., 3].clamp_min(0.0)

        query_log_var = torch.log1p(query_variance)
        support_log_var = torch.log1p(support_variance)
        variance_gap = (query_log_var - support_log_var).abs()
        joint_dispersion = torch.log1p(query_variance + support_variance)
        evidence_features = torch.stack(
            [
                mean_distance,
                shot_spread,
                variance_gap,
                joint_dispersion,
            ],
            dim=-1,
        )
        return self._standardize_over_candidates(evidence_features)

    def _compact_geodesic_eam_prior(self, compact_features: torch.Tensor) -> torch.Tensor:
        if compact_features.shape[-1] != 4:
            raise ValueError(f"compact_features last dim must be 4, got {compact_features.shape[-1]}")
        distance_z = compact_features[..., 0]
        spread_z = compact_features[..., 1]
        variance_gap_z = compact_features[..., 2]
        dispersion_z = compact_features[..., 3]
        evidence = -(distance_z + 0.5 * spread_z + 0.25 * variance_gap_z + 0.25 * dispersion_z)
        base_logit = math.log(self.fixed_mass / (1.0 - self.fixed_mass))
        prior = torch.sigmoid(evidence + compact_features.new_tensor(base_logit))
        return prior.clamp(min=self.min_mass, max=1.0)

    def _apply_compact_geodesic_mass_prior(
        self,
        rho: torch.Tensor,
        compact_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_compact_geodesic_eam or self.compact_eam_prior_mix == 0.0:
            return rho
        prior = self._compact_geodesic_eam_prior(compact_features).to(device=rho.device, dtype=rho.dtype)
        mix = rho.new_tensor(self.compact_eam_prior_mix)
        return ((1.0 - mix) * rho + mix * prior).clamp(min=self.min_mass, max=1.0)

    def _compute_hyperbolic_token_prior_logits(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.query_token_attention_vector is None or self.support_token_attention_vector is None:
            raise RuntimeError("Noise-calibrated transport requires token attention vectors")
        if query_tokens_hyp.dim() != 3 or support_tokens_hyp.dim() != 3:
            raise ValueError(
                "Hyperbolic token prior expects query/support tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumPairs, Tokens, Dim)"
            )

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        support_cast = support_tokens_hyp.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast)
        query_tangent = ball.logmap0(ball.project(query_cast))
        support_tangent = ball.logmap0(ball.project(support_cast))

        scale = math.sqrt(float(query_tangent.shape[-1]))
        query_vector = self.query_token_attention_vector.to(
            device=query_tangent.device,
            dtype=query_tangent.dtype,
        )
        support_vector = self.support_token_attention_vector.to(
            device=support_tangent.device,
            dtype=support_tangent.dtype,
        )
        query_direction = torch.einsum("qtd,d->qt", query_tangent, query_vector) / scale
        support_direction = torch.einsum("ptd,d->pt", support_tangent, support_vector) / scale
        query_radius = torch.linalg.vector_norm(query_tangent, dim=-1)
        support_radius = torch.linalg.vector_norm(support_tangent, dim=-1)

        query_logits = self._standardize_over_tokens((query_direction + query_radius).unsqueeze(-1)).squeeze(-1)
        support_logits = self._standardize_over_tokens((support_direction + support_radius).unsqueeze(-1)).squeeze(-1)
        query_logits = query_logits[:, None, :].expand(-1, support_tokens_hyp.shape[0], -1)
        support_logits = support_logits[None, :, :].expand(query_tokens_hyp.shape[0], -1, -1)
        return query_logits.to(dtype=query_tokens_hyp.dtype), support_logits.to(dtype=support_tokens_hyp.dtype)

    def _build_probe_token_features(
        self,
        flat_cost: torch.Tensor,
        flat_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            probe_plan, probe_cost, probe_mass = self._transport_match(flat_cost.detach(), flat_rho.detach())
            plan_mass = probe_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

            row_mass = probe_plan.sum(dim=-1)
            row_cost = (probe_plan * flat_cost.detach()).sum(dim=-1) / row_mass.clamp_min(self.eps)
            row_prob = probe_plan / row_mass.unsqueeze(-1).clamp_min(self.eps)
            row_entropy = -(row_prob * row_prob.clamp_min(self.eps).log()).sum(dim=-1)
            row_entropy = row_entropy / math.log(float(max(2, flat_cost.shape[-1])))
            row_min_cost = flat_cost.detach().amin(dim=-1)
            row_mass_share = row_mass / plan_mass.squeeze(-1) * float(flat_cost.shape[-2])

            col_mass = probe_plan.sum(dim=-2)
            col_cost = (probe_plan * flat_cost.detach()).sum(dim=-2) / col_mass.clamp_min(self.eps)
            col_prob = probe_plan / col_mass.unsqueeze(-2).clamp_min(self.eps)
            col_entropy = -(col_prob * col_prob.clamp_min(self.eps).log()).sum(dim=-2)
            col_entropy = col_entropy / math.log(float(max(2, flat_cost.shape[-2])))
            col_min_cost = flat_cost.detach().amin(dim=-2)
            col_mass_share = col_mass / plan_mass.squeeze(-2) * float(flat_cost.shape[-1])

            query_features = torch.stack(
                [
                    row_mass_share.clamp_min(self.eps).log(),
                    row_cost,
                    row_entropy,
                    row_min_cost,
                ],
                dim=-1,
            )
            support_features = torch.stack(
                [
                    col_mass_share.clamp_min(self.eps).log(),
                    col_cost,
                    col_entropy,
                    col_min_cost,
                ],
                dim=-1,
            )

            query_features = self._standardize_over_tokens(query_features)
            support_features = self._standardize_over_tokens(support_features)
            plan_prob = probe_plan / plan_mass
            probe_entropy = -(plan_prob * plan_prob.clamp_min(self.eps).log()).sum(dim=(-1, -2))
            probe_entropy = probe_entropy / math.log(float(max(2, flat_cost.shape[-2] * flat_cost.shape[-1])))
            probe_min_cost = flat_cost.detach().amin(dim=(-1, -2))

        payload = {
            "transport_probe_cost": probe_cost,
            "transport_probe_mass": probe_mass,
            "transport_probe_entropy": probe_entropy,
            "transport_probe_min_cost": probe_min_cost,
        }
        return query_features.to(dtype=flat_cost.dtype), support_features.to(dtype=flat_cost.dtype), payload

    def _build_support_consensus_scores(self, support_tokens_euc: torch.Tensor) -> torch.Tensor:
        way_num, shot_num, token_num, _ = support_tokens_euc.shape
        if shot_num <= 1:
            return support_tokens_euc.new_zeros(way_num, shot_num, token_num)

        pair_cost = (
            support_tokens_euc[:, :, None, :, None, :] - support_tokens_euc[:, None, :, None, :, :]
        ).pow(2).sum(dim=-1)
        nearest_other = pair_cost.amin(dim=-1)
        shot_mask = ~torch.eye(shot_num, device=support_tokens_euc.device, dtype=torch.bool)
        nearest_other = nearest_other.masked_fill(~shot_mask[None, :, :, None], 0.0)
        mean_nearest = nearest_other.sum(dim=2) / float(shot_num - 1)
        temperature = self.consensus_temperature.to(device=support_tokens_euc.device, dtype=support_tokens_euc.dtype)
        scores = -mean_nearest / temperature
        return self._standardize_over_tokens(scores.unsqueeze(-1)).squeeze(-1)

    def _compute_noise_calibrated_token_marginals(
        self,
        flat_cost: torch.Tensor,
        flat_rho: torch.Tensor,
        support_tokens_euc: torch.Tensor,
        query_tokens_hyp: torch.Tensor,
        flat_support_tokens_hyp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.query_reliability_weights is None or self.support_reliability_weights is None:
            raise RuntimeError("Noise-calibrated transport requires reliability weights")

        way_num, shot_num = support_tokens_euc.shape[:2]
        query_features, support_features, payload = self._build_probe_token_features(flat_cost, flat_rho)
        query_logits = torch.einsum(
            "qstf,f->qst",
            query_features,
            self.query_reliability_weights.to(device=flat_cost.device, dtype=flat_cost.dtype),
        )
        support_logits = torch.einsum(
            "qstf,f->qst",
            support_features,
            self.support_reliability_weights.to(device=flat_cost.device, dtype=flat_cost.dtype),
        )

        support_consensus = self._build_support_consensus_scores(support_tokens_euc)
        flat_consensus = support_consensus.reshape(way_num * shot_num, support_consensus.shape[-1])
        flat_consensus = flat_consensus.unsqueeze(0).expand(flat_cost.shape[0], -1, -1)
        consensus_mix = self.support_consensus_mix.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_logits = support_logits + consensus_mix * flat_consensus

        query_prior_logits, support_prior_logits = self._compute_hyperbolic_token_prior_logits(
            query_tokens_hyp,
            flat_support_tokens_hyp,
        )
        query_prior_logits = query_prior_logits.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_prior_logits = support_prior_logits.to(device=flat_cost.device, dtype=flat_cost.dtype)
        prior_mix = self.hyperbolic_token_prior_mix.to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_logits = query_logits + prior_mix * query_prior_logits
        support_logits = support_logits + prior_mix * support_prior_logits

        temperature = self.token_temperature.to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_attn = torch.softmax(query_logits / temperature, dim=-1)
        support_attn = torch.softmax(support_logits / temperature, dim=-1)
        query_uniform = flat_cost.new_full(query_attn.shape, 1.0 / float(query_attn.shape[-1]))
        support_uniform = flat_cost.new_full(support_attn.shape, 1.0 / float(support_attn.shape[-1]))
        reliability_mix = self.token_reliability_mix.to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_weights = (1.0 - reliability_mix) * query_uniform + reliability_mix * query_attn
        support_weights = (1.0 - reliability_mix) * support_uniform + reliability_mix * support_attn

        query_mass = query_weights * flat_rho.unsqueeze(-1).to(dtype=query_weights.dtype)
        support_mass = support_weights * flat_rho.unsqueeze(-1).to(dtype=support_weights.dtype)
        payload.update(
            {
                "probe_query_reliability": query_weights,
                "probe_support_reliability": support_weights,
                "support_consensus": flat_consensus,
                "query_hyperbolic_token_prior": torch.softmax(query_prior_logits / temperature, dim=-1),
                "support_hyperbolic_token_prior": torch.softmax(support_prior_logits / temperature, dim=-1),
            }
        )
        return query_mass, support_mass, payload

    def _compute_token_marginals(
        self,
        query_euc: torch.Tensor,
        support_euc: torch.Tensor,
        flat_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute HROT-V learned token marginals from dot-product attention."""
        if self.w_q is None or self.w_s is None:
            raise RuntimeError("HROT-V token marginals require w_q and w_s parameters")
        if query_euc.dim() != 3 or support_euc.dim() != 3:
            raise ValueError("HROT-V token marginals expect query/support shaped (N, Tokens, Dim)")
        if tuple(flat_rho.shape) != (query_euc.shape[0], support_euc.shape[0]):
            raise ValueError(
                "flat_rho must have shape (NumQuery, NumPairs), "
                f"got {tuple(flat_rho.shape)} vs {(query_euc.shape[0], support_euc.shape[0])}"
            )

        tau = self.tau_token.to(device=query_euc.device, dtype=query_euc.dtype)
        w_q = self.w_q.to(device=query_euc.device, dtype=query_euc.dtype)
        w_s = self.w_s.to(device=support_euc.device, dtype=support_euc.dtype)

        q_logits = torch.einsum("qtd,d->qt", query_euc, w_q)
        s_logits = torch.einsum("ptd,d->pt", support_euc, w_s)
        q_attn = F.softmax(q_logits / tau, dim=-1)
        s_attn = F.softmax(s_logits / tau, dim=-1)

        rho = flat_rho.to(device=query_euc.device, dtype=query_euc.dtype)
        query_mass = q_attn[:, None, :] * rho[:, :, None]
        support_mass = s_attn[None, :, :].to(device=query_euc.device, dtype=query_euc.dtype) * rho[:, :, None]
        return query_mass, support_mass

    def _compute_gw_structure_cost(
        self,
        query_euc: torch.Tensor,
        support_euc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute HROT-V differentiable linearized FGW neighborhood cost."""
        if self.raw_structure_weight is None:
            raise RuntimeError("HROT-V structure cost requires raw_structure_weight")
        if query_euc.dim() != 3 or support_euc.dim() != 3:
            raise ValueError("HROT-V structure cost expects query/support shaped (N, Tokens, Dim)")
        if query_euc.shape[-2] != support_euc.shape[-2]:
            raise ValueError(
                "HROT-V structure cost requires equal query/support token counts, "
                f"got {query_euc.shape[-2]} vs {support_euc.shape[-2]}"
            )

        D_q = torch.cdist(query_euc, query_euc, p=2)
        D_s = torch.cdist(support_euc, support_euc, p=2)
        D_q = D_q / D_q.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        D_s = D_s / D_s.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        term_q = D_q.pow(2).sum(dim=-1)[:, None, :, None]
        term_s = D_s.pow(2).sum(dim=-1)[None, :, None, :]
        cross = torch.einsum("qik,pjk->qpij", D_q, D_s)
        structure_cost = (term_q + term_s - 2.0 * cross).clamp_min(0.0)
        structure_cost = structure_cost / structure_cost.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        weight = self.structure_cost_weight.to(device=query_euc.device, dtype=query_euc.dtype)
        return weight * structure_cost

    def _compute_threshold(self, flat_rho: torch.Tensor) -> torch.Tensor:
        """Compute HROT-V positive adaptive threshold from pairwise rho."""
        if self.raw_threshold_offset is None or self.raw_threshold_scale is None:
            raise RuntimeError("HROT-V threshold requires raw_threshold_offset and raw_threshold_scale")
        offset = self.raw_threshold_offset.to(device=flat_rho.device, dtype=flat_rho.dtype)
        scale = F.softplus(self.raw_threshold_scale.to(device=flat_rho.device, dtype=flat_rho.dtype)) + self.eps
        return F.softplus(offset + scale * flat_rho)

    def _pairwise_token_distance(self, tokens: torch.Tensor) -> torch.Tensor:
        token_norm = tokens.pow(2).sum(dim=-1)
        pairwise = token_norm.unsqueeze(-1) + token_norm.unsqueeze(-2) - 2.0 * torch.matmul(
            tokens,
            tokens.transpose(-1, -2),
        )
        return pairwise.clamp_min(0.0)

    def _normalize_structure_distance(self, distance: torch.Tensor) -> torch.Tensor:
        scale = distance.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        return distance / scale

    def _compute_structure_consistency_cost(
        self,
        flat_cost: torch.Tensor,
        query_tokens_euc: torch.Tensor,
        flat_support_tokens_euc: torch.Tensor,
        flat_rho: torch.Tensor,
        query_mass: torch.Tensor,
        support_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_tokens_euc.dim() != 3 or flat_support_tokens_euc.dim() != 3:
            raise ValueError("Structure cost expects query/support tokens shaped (N, Tokens, Dim)")
        if flat_cost.shape[:2] != flat_rho.shape:
            raise ValueError(f"flat_rho must match flat_cost leading dims, got {tuple(flat_rho.shape)}")

        with torch.no_grad():
            probe_plan, _, _ = self._transport_match(
                flat_cost.detach(),
                flat_rho.detach(),
                a=query_mass.detach(),
                b=support_mass.detach(),
            )
            probe_plan = probe_plan.detach()
            row_mass = probe_plan.sum(dim=-1)
            col_mass = probe_plan.sum(dim=-2)
            probe_mass = probe_plan.sum(dim=(-1, -2))

        query_structure = self._normalize_structure_distance(self._pairwise_token_distance(query_tokens_euc))
        support_structure = self._normalize_structure_distance(
            self._pairwise_token_distance(flat_support_tokens_euc)
        )
        query_structure = query_structure.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_structure = support_structure.to(device=flat_cost.device, dtype=flat_cost.dtype)

        query_term = torch.einsum("qti,qpi->qpt", query_structure.pow(2), row_mass)
        support_term = torch.einsum("pkj,qpj->qpk", support_structure.pow(2), col_mass)
        cross_term = torch.einsum("qti,qpij,pkj->qptk", query_structure, probe_plan, support_structure)
        structure_cost = (query_term.unsqueeze(-1) + support_term.unsqueeze(-2) - 2.0 * cross_term).clamp_min(0.0)
        structure_cost = structure_cost / structure_cost.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        structure_cost = self.structure_cost_weight.to(device=flat_cost.device, dtype=flat_cost.dtype) * structure_cost

        payload = {
            "structure_cost": structure_cost,
            "structure_probe_mass": probe_mass,
        }
        return structure_cost.to(dtype=flat_cost.dtype), payload

    def _append_noise_sink(
        self,
        flat_cost: torch.Tensor,
        query_mass: torch.Tensor,
        support_mass: torch.Tensor,
        flat_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sink_cost = self.noise_sink_cost.to(device=flat_cost.device, dtype=flat_cost.dtype)
        cost_with_sink = sink_cost.expand(
            flat_cost.shape[0],
            flat_cost.shape[1],
            flat_cost.shape[2] + 1,
            flat_cost.shape[3] + 1,
        ).clone()
        cost_with_sink[..., :-1, :-1] = flat_cost
        cost_with_sink[..., -1, -1] = 0.0

        sink_mass = (1.0 - flat_rho).clamp_min(self.eps).to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_mass_with_sink = torch.cat([query_mass, sink_mass.unsqueeze(-1)], dim=-1)
        support_mass_with_sink = torch.cat([support_mass, sink_mass.unsqueeze(-1)], dim=-1)
        return cost_with_sink, query_mass_with_sink, support_mass_with_sink

    def _compute_ncet_nuisance_scores(
        self,
        flat_cost: torch.Tensor,
        probe_plan: torch.Tensor,
        support_tokens_euc: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate behavior-defined nuisance scores without relying on k-shot repeats."""
        with torch.no_grad():
            if flat_cost.dim() != 4:
                raise ValueError(f"NCET flat_cost must be 4D, got {tuple(flat_cost.shape)}")
            if probe_plan.shape != flat_cost.shape:
                raise ValueError(
                    "NCET probe_plan must match flat_cost shape, "
                    f"got {tuple(probe_plan.shape)} vs {tuple(flat_cost.shape)}"
                )
            num_query, num_pairs, query_len, support_len = flat_cost.shape
            if num_pairs != way_num * shot_num:
                raise ValueError(f"NCET expected Way*Shot={way_num * shot_num}, got {num_pairs}")

            cost_det = flat_cost.detach()
            plan_det = probe_plan.detach().clamp_min(0.0)

            row_min = cost_det.amin(dim=-1)
            if way_num > 1:
                class_row_min = row_min.reshape(num_query, way_num, shot_num, query_len).amin(dim=2)
                class_mean = class_row_min.mean(dim=1, keepdim=True)
                class_std = class_row_min.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
                class_logits = -((class_row_min - class_mean) / class_std)
                class_prob = torch.softmax(class_logits, dim=1)
                query_ambiguity = 1.0 - class_prob
                query_ambiguity = query_ambiguity[:, :, None, :].expand(-1, -1, shot_num, -1)
                query_ambiguity = query_ambiguity.reshape(num_query, num_pairs, query_len)
            else:
                query_ambiguity = torch.zeros_like(row_min)

            support_common = cost_det.new_zeros(way_num, shot_num, support_len)
            if way_num > 1:
                support_flat = support_tokens_euc.detach().reshape(way_num * shot_num, support_len, -1)
                token_flat = support_flat.reshape(way_num * shot_num * support_len, -1)
                token_dist = torch.cdist(token_flat, token_flat, p=2).pow(2)
                class_ids = torch.arange(way_num, device=flat_cost.device).repeat_interleave(shot_num * support_len)
                same_class = class_ids[:, None] == class_ids[None, :]
                token_dist = token_dist.masked_fill(same_class, float("inf"))
                min_other = token_dist.amin(dim=-1)
                mean_other = min_other.mean()
                std_other = min_other.std(unbiased=False).clamp_min(self.eps)
                support_common = torch.sigmoid(-((min_other - mean_other) / std_other))
                support_common = support_common.reshape(way_num, shot_num, support_len)
            support_common = support_common.reshape(1, num_pairs, support_len).expand(num_query, -1, -1)

            row_mass = plan_det.sum(dim=-1).clamp_min(self.eps)
            row_prob = plan_det / row_mass.unsqueeze(-1)
            row_entropy = -(row_prob * row_prob.clamp_min(self.eps).log()).sum(dim=-1)
            row_entropy = row_entropy / math.log(float(max(2, support_len)))

            col_mass = plan_det.sum(dim=-2).clamp_min(self.eps)
            col_prob = plan_det / col_mass.unsqueeze(-2)
            col_entropy = -(col_prob * col_prob.clamp_min(self.eps).log()).sum(dim=-2)
            col_entropy = col_entropy / math.log(float(max(2, query_len)))

            row_cost = (plan_det * cost_det).sum(dim=-1) / row_mass
            col_cost = (plan_det * cost_det).sum(dim=-2) / col_mass
            row_cost_z = (row_cost - row_cost.mean(dim=-1, keepdim=True)) / row_cost.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(self.eps)
            col_cost_z = (col_cost - col_cost.mean(dim=-1, keepdim=True)) / col_cost.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(self.eps)
            query_high_cost = torch.sigmoid(row_cost_z)
            support_high_cost = torch.sigmoid(col_cost_z)

            query_nuisance = 0.50 * query_ambiguity + 0.30 * row_entropy + 0.20 * query_high_cost
            support_nuisance = 0.50 * support_common + 0.30 * col_entropy + 0.20 * support_high_cost
            query_nuisance = torch.nan_to_num(query_nuisance, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            support_nuisance = torch.nan_to_num(support_nuisance, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            confidence = ((query_nuisance - 0.5).abs().mean(dim=-1) + (support_nuisance - 0.5).abs().mean(dim=-1))
            confidence = confidence.clamp(0.0, 1.0)

        return query_nuisance, support_nuisance, confidence

    def _append_ncet_nuisance_sink(
        self,
        flat_cost: torch.Tensor,
        query_mass: torch.Tensor,
        support_mass: torch.Tensor,
        flat_rho: torch.Tensor,
        query_nuisance: torch.Tensor,
        support_nuisance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_nuisance.shape != query_mass.shape:
            raise ValueError(
                "query_nuisance must match query_mass shape, "
                f"got {tuple(query_nuisance.shape)} vs {tuple(query_mass.shape)}"
            )
        if support_nuisance.shape != support_mass.shape:
            raise ValueError(
                "support_nuisance must match support_mass shape, "
                f"got {tuple(support_nuisance.shape)} vs {tuple(support_mass.shape)}"
            )

        real_penalty = self.ncet_real_penalty.to(device=flat_cost.device, dtype=flat_cost.dtype)
        adjusted_real_cost = flat_cost + 0.5 * real_penalty * (
            query_nuisance.unsqueeze(-1).to(dtype=flat_cost.dtype)
            + support_nuisance.unsqueeze(-2).to(dtype=flat_cost.dtype)
        )

        sink_cost = self.noise_sink_cost.to(device=flat_cost.device, dtype=flat_cost.dtype)
        cost_with_sink = sink_cost.expand(
            flat_cost.shape[0],
            flat_cost.shape[1],
            flat_cost.shape[2] + 1,
            flat_cost.shape[3] + 1,
        ).clone()
        cost_with_sink[..., :-1, :-1] = adjusted_real_cost
        cost_with_sink[..., :-1, -1] = sink_cost * (0.1 + 0.9 * (1.0 - query_nuisance.to(dtype=flat_cost.dtype)))
        cost_with_sink[..., -1, :-1] = sink_cost * (0.1 + 0.9 * (1.0 - support_nuisance.to(dtype=flat_cost.dtype)))
        cost_with_sink[..., -1, -1] = 0.0

        sink_mass = (1.0 - flat_rho).clamp_min(self.eps).to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_mass_with_sink = torch.cat([query_mass, sink_mass.unsqueeze(-1)], dim=-1)
        support_mass_with_sink = torch.cat([support_mass, sink_mass.unsqueeze(-1)], dim=-1)
        return cost_with_sink, query_mass_with_sink, support_mass_with_sink, adjusted_real_cost

    def _pool_shot_scores(
        self,
        shot_logits: torch.Tensor,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
        geodesic_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shot_num = shot_logits.shape[-1]
        if shot_num == 1:
            weights = torch.ones_like(shot_logits)
        else:
            if geodesic_features is not None and self.q_shot_pool_scorer is not None:
                if geodesic_features.shape[:-1] != shot_logits.shape:
                    raise ValueError(
                        "geodesic_features must have shape (*shot_logits.shape, FeatureDim), "
                        f"got {tuple(geodesic_features.shape)} vs {tuple(shot_logits.shape)}"
                    )
                pooled_features = self._standardize_over_shots(geodesic_features)
                network_dtype = self.q_shot_pool_scorer.weight.dtype
                evidence = self.q_shot_pool_scorer(pooled_features.to(dtype=network_dtype)).squeeze(-1)
                evidence = evidence.to(device=shot_logits.device, dtype=shot_logits.dtype)
            else:
                evidence = shot_logits / float(self.score_scale)
                evidence = (evidence - evidence.mean(dim=-1, keepdim=True)) / evidence.std(
                    dim=-1,
                    keepdim=True,
                    unbiased=False,
                ).clamp_min(self.eps)
            temperature = self.shot_pool_temperature.to(device=shot_logits.device, dtype=shot_logits.dtype)
            attentive = torch.softmax(evidence / temperature, dim=-1)
            uniform = torch.full_like(attentive, 1.0 / float(shot_num))
            mix = self.shot_pool_mix.to(device=shot_logits.device, dtype=shot_logits.dtype)
            weights = (1.0 - mix) * uniform + mix * attentive

        logits = (weights * shot_logits).sum(dim=-1)
        transport_cost = (weights * shot_transport_cost).sum(dim=-1)
        transport_mass = (weights * shot_transport_mass).sum(dim=-1)
        return logits, transport_cost, transport_mass, weights

    def _pool_j_shot_scores(
        self,
        shot_logits: torch.Tensor,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shot_num = shot_logits.shape[-1]
        if shot_num == 1:
            weights = torch.ones_like(shot_logits)
            logits = shot_logits.squeeze(-1)
        else:
            weights = torch.softmax(shot_logits, dim=-1)
            logits = torch.logsumexp(shot_logits, dim=-1) - math.log(float(shot_num))
        transport_cost = (weights * shot_transport_cost).sum(dim=-1)
        transport_mass = (weights * shot_transport_mass).sum(dim=-1)
        return logits, transport_cost, transport_mass, weights

    def _compute_ecot_diagnostics(
        self,
        budget_scores: torch.Tensor,
        shot_cost_bank: torch.Tensor,
        shot_mass_bank: torch.Tensor,
    ) -> torch.Tensor:
        scores_det = budget_scores.detach()
        cost_det = shot_cost_bank.detach()
        mass_det = shot_mass_bank.detach()
        base_idx = self._ecot_base_idx_value()
        base_scores = scores_det[..., base_idx]
        shot_num = base_scores.shape[-1]
        if shot_num == 1:
            base_class_scores = base_scores.squeeze(-1)
            shot_entropy = torch.zeros_like(base_scores.squeeze(-1))
        else:
            base_class_scores = torch.logsumexp(base_scores, dim=-1) - math.log(float(shot_num))
            shot_weights = torch.softmax(base_scores, dim=-1)
            shot_entropy = -(shot_weights.clamp_min(self.eps) * shot_weights.clamp_min(self.eps).log()).sum(dim=-1)

        if base_class_scores.shape[-1] >= 2:
            top2 = torch.topk(base_class_scores, k=2, dim=-1).values
            margins = top2[..., 0] - top2[..., 1]
        else:
            margins = torch.zeros_like(base_class_scores[..., 0])

        budget_slope = scores_det[..., -1] - scores_det[..., 0]
        diagnostics = torch.stack(
            [
                scores_det.mean(),
                scores_det.std(unbiased=False),
                cost_det.mean(),
                cost_det.std(unbiased=False),
                mass_det.mean(),
                mass_det.std(unbiased=False),
                margins.mean(),
                margins.std(unbiased=False),
                shot_entropy.mean(),
                budget_slope.mean(),
                budget_slope.std(unbiased=False),
            ]
        )
        diagnostics = torch.nan_to_num(diagnostics, nan=0.0, posinf=0.0, neginf=0.0)
        return diagnostics.to(dtype=budget_scores.dtype, device=budget_scores.device)

    def _pool_ecot_shot_scores(
        self,
        shot_logits: torch.Tensor,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
        raw_tau_shot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if raw_tau_shot is None:
            logits, transport_cost, transport_mass, weights = self._pool_j_shot_scores(
                shot_logits,
                shot_transport_cost,
                shot_transport_mass,
            )
            return logits, transport_cost, transport_mass, weights, None

        tau_min = shot_logits.new_tensor(self.ecot_tau_shot_min)
        tau_max = shot_logits.new_tensor(self.ecot_tau_shot_max)
        tau_shot = tau_min + (tau_max - tau_min) * torch.sigmoid(
            raw_tau_shot.to(device=shot_logits.device, dtype=shot_logits.dtype)
        )
        scaled = shot_logits / tau_shot.clamp_min(self.eps)
        shot_num = shot_logits.shape[-1]
        if shot_num == 1:
            weights = torch.ones_like(shot_logits)
            logits = shot_logits.squeeze(-1)
        else:
            weights = torch.softmax(scaled, dim=-1)
            logits = tau_shot * (torch.logsumexp(scaled, dim=-1) - math.log(float(shot_num)))
        transport_cost = (weights * shot_transport_cost).sum(dim=-1)
        transport_mass = (weights * shot_transport_mass).sum(dim=-1)
        return logits, transport_cost, transport_mass, weights, tau_shot

    def _cp_ecot_tau_k(self, shot_num: int, reference: torch.Tensor) -> torch.Tensor:
        if self.ecot_consensus_tau_mode == "sqrt":
            tau_value = min(1.5, math.sqrt(float(shot_num)))
        else:
            tau_value = self.ecot_consensus_tau
        return reference.new_tensor(float(tau_value))

    def _pool_cp_ecot_class_budget_scores(
        self,
        budget_scores: torch.Tensor,
        shot_cost_bank: torch.Tensor,
        shot_mass_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if budget_scores.dim() != 4:
            raise ValueError(
                "CP_ECOT budget_scores must have shape (Nq, Way, Shot, Budget), "
                f"got {tuple(budget_scores.shape)}"
            )
        shot_num = budget_scores.shape[-2]
        tau_k = self._cp_ecot_tau_k(shot_num, budget_scores)
        scaled = budget_scores / tau_k.clamp_min(self.eps)
        if shot_num == 1:
            weights = torch.ones_like(budget_scores)
            class_budget_scores = budget_scores.squeeze(-2)
        else:
            weights = torch.softmax(scaled, dim=-2)
            class_budget_scores = tau_k * (torch.logsumexp(scaled, dim=-2) - math.log(float(shot_num)))
        class_budget_cost = (weights * shot_cost_bank).sum(dim=-2)
        class_budget_mass = (weights * shot_mass_bank).sum(dim=-2)
        return class_budget_scores, class_budget_cost, class_budget_mass, weights, tau_k

    def _compute_nncs_support_payload(
        self,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if support_tokens.dim() != 4:
            raise ValueError(
                "NNCS support marginals expect support tokens shaped "
                f"(Way, Shot, Tokens, Dim), got {tuple(support_tokens.shape)}"
            )
        distance = self.ground_cost if self.ecot_nncs_distance == "auto" else self.ecot_nncs_distance
        b, consistency, sigma_norm = _compute_cross_shot_support_marginals(
            support_tokens.unsqueeze(0),
            rho=self.ecot_base_rho,
            beta=self.ecot_nncs_beta,
            eta=self.ecot_nncs_eta,
            detach=self.ecot_nncs_detach,
            distance=distance,
            eps=self.eps,
            min_sigma=self.ecot_nncs_min_sigma,
        )
        support_weight = (b / float(self.ecot_base_rho)).squeeze(0)
        prob = b / b.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        entropy = -(prob * prob.clamp_min(self.eps).log()).sum(dim=-1)
        entropy = entropy / math.log(float(max(2, b.shape[-1])))
        payload = {
            "nncs_support_marginal": b,
            "nncs_consistency": consistency,
            "nncs_sigma": sigma_norm,
            "nncs_beta": support_tokens.new_tensor(self.ecot_nncs_beta),
            "nncs_eta": support_tokens.new_tensor(self.ecot_nncs_eta),
            "mean_nncs_entropy": entropy.mean(),
            "mean_nncs_max_mass": b.max(dim=-1).values.mean(),
            "mean_nncs_min_mass": b.min(dim=-1).values.mean(),
            "mean_nncs_consistency": consistency.mean(),
            "std_nncs_consistency": consistency.std(unbiased=False),
        }
        return support_weight.to(dtype=support_tokens.dtype), payload

    @staticmethod
    def _ecot_swts_mass_removed_score_terms(
        plan: torch.Tensor,
        cost: torch.Tensor,
        swts_temp: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-Weighted Transport Score for mass-removed M2 (-C ablation).

        plan, cost: same shape with leading batch dims and last two (Lq, Ls).
        Returns ``(score_terms, w_entropy_mean)`` where ``score_terms`` has the
        same leading dimensions as ``plan`` minus the final two, and
        ``w_entropy_mean`` is a scalar tensor: mean of
        ``-(w * log(w)).sum(-1)`` over all query/class/shot/budget cells.
        """
        s = (plan * (-cost)).sum(dim=-1)
        q = plan.sum(dim=-1)
        temp = max(float(swts_temp), float(eps))
        w = torch.softmax(q / temp, dim=-1)
        lq = int(plan.shape[-2])
        score = lq * (w * s).sum(dim=-1)
        w_c = w.clamp_min(float(eps))
        w_entropy_cells = -(w * w_c.log()).sum(dim=-1)
        w_entropy_mean = w_entropy_cells.mean()
        return score, w_entropy_mean

    def _forward_ecot_budget_bank(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        support_weight: torch.Tensor | None = None,
        query_weight: torch.Tensor | None = None,
        support_tokens: torch.Tensor | None = None,
        query_tokens: torch.Tensor | None = None,
        spatial_hw: tuple[int, int] | None = None,
        score_query_evidence: torch.Tensor | None = None,
        score_support_evidence: torch.Tensor | None = None,
        score_pair_evidence: torch.Tensor | None = None,
        score_evidence_mass_weight: float = 1.0,
        score_evidence_cost_weight: float = 1.0,
        score_background_penalty: float = 0.0,
        score_evidence_mode: str = "replace",
        score_evidence_mix: float = 1.0,
        transport_tau_q: torch.Tensor | None = None,
        transport_tau_c: torch.Tensor | None = None,
        threshold_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.episode_controller is None:
            raise RuntimeError("ECOT variants require an episode_controller")
        if self.ecot_rho_bank_tensor is None:
            raise RuntimeError("ECOT variants require an ecot_rho_bank_tensor")
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        budget_count = len(self.ecot_rho_bank)
        if num_pairs != way_num * shot_num:
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={way_num * shot_num}")
        if (transport_tau_q is None) != (transport_tau_c is None):
            raise ValueError("transport_tau_q and transport_tau_c must be provided together")
        adaptive_tau_q_bank: torch.Tensor | None = None
        adaptive_tau_c_bank: torch.Tensor | None = None
        if transport_tau_q is not None and transport_tau_c is not None:
            if not self.uses_unbalanced_transport:
                raise ValueError("token-wise transport relaxation requires unbalanced OT")
            if self.ot_backend != "native":
                raise ValueError("token-wise transport relaxation currently requires the native OT backend")
            if tuple(transport_tau_q.shape) != (num_query, num_pairs, query_len):
                raise ValueError(
                    "transport_tau_q must have shape "
                    f"{(num_query, num_pairs, query_len)}, got {tuple(transport_tau_q.shape)}"
                )
            if tuple(transport_tau_c.shape) != (num_query, num_pairs, support_len):
                raise ValueError(
                    "transport_tau_c must have shape "
                    f"{(num_query, num_pairs, support_len)}, got {tuple(transport_tau_c.shape)}"
                )
            adaptive_tau_q_bank = (
                transport_tau_q.to(device=flat_cost.device, dtype=flat_cost.dtype)
                .unsqueeze(2)
                .expand(-1, -1, budget_count, -1)
                .reshape(num_query, num_pairs * budget_count, query_len)
            )
            adaptive_tau_c_bank = (
                transport_tau_c.to(device=flat_cost.device, dtype=flat_cost.dtype)
                .unsqueeze(2)
                .expand(-1, -1, budget_count, -1)
                .reshape(num_query, num_pairs * budget_count, support_len)
            )

        rho_bank = self.ecot_rho_bank_tensor.to(device=flat_cost.device, dtype=flat_cost.dtype)
        cost_bank = flat_cost.unsqueeze(2).expand(num_query, num_pairs, budget_count, query_len, support_len)
        cost_bank = cost_bank.reshape(num_query, num_pairs * budget_count, query_len, support_len)
        rho_bank_flat = rho_bank.view(1, 1, budget_count).expand(num_query, num_pairs, budget_count)
        rho_bank_flat = rho_bank_flat.reshape(num_query, num_pairs * budget_count)

        transport_kwargs: dict[str, torch.Tensor] = {}
        mean_aqm_a_entropy: torch.Tensor | None = None
        crs_payload: dict[str, torch.Tensor] = {}
        mea_payload: dict[str, torch.Tensor] = {}
        ccdm_payload: dict[str, torch.Tensor] = {}
        egsm_payload: dict[str, torch.Tensor] = {}
        ccem_payload: dict[str, torch.Tensor] = {}
        tam_payload: dict[str, torch.Tensor] = {}
        if self.token_attention_marginal is not None:
            if query_weight is not None or support_weight is not None:
                raise ValueError(
                    "TokenAttentionMarginal cannot be combined with explicit "
                    "query/support weights"
                )
            if support_tokens is None or query_tokens is None:
                raise ValueError(
                    "TokenAttentionMarginal requires support_tokens and query_tokens"
                )
            if tuple(query_tokens.shape[:2]) != (num_query, query_len):
                raise ValueError(
                    "query_tokens must have shape (NumQuery, QueryTokens, Dim), "
                    f"got {tuple(query_tokens.shape)}"
                )
            if tuple(support_tokens.shape[:3]) != (
                way_num,
                shot_num,
                support_len,
            ):
                raise ValueError(
                    "support_tokens must have shape "
                    "(Way, Shot, SupportTokens, Dim), "
                    f"got {tuple(support_tokens.shape)}"
                )
            base_rho = self._ecot_base_rho_tensor(flat_cost)
            query_rho = base_rho.expand(num_query)
            support_rho = base_rho.expand(way_num, shot_num)
            query_marginal, query_diag = self.token_attention_marginal(
                query_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype),
                query_rho,
            )
            support_module = (
                self.token_attention_marginal
                if self.token_attention_support_marginal is None
                else self.token_attention_support_marginal
            )
            support_marginal, support_diag = support_module(
                support_tokens.to(
                    device=flat_cost.device,
                    dtype=flat_cost.dtype,
                ),
                support_rho,
            )
            query_prob_shared = query_marginal / query_marginal.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
            support_prob_shared = support_marginal / support_marginal.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
            query_prob = query_prob_shared.unsqueeze(1).expand(
                -1,
                num_pairs,
                -1,
            )
            support_prob = support_prob_shared.reshape(
                1,
                num_pairs,
                support_len,
            ).expand(num_query, -1, -1)
            a_bank = query_prob.unsqueeze(2) * rho_bank.view(
                1,
                1,
                budget_count,
                1,
            )
            b_bank = support_prob.unsqueeze(2) * rho_bank.view(
                1,
                1,
                budget_count,
                1,
            )
            transport_kwargs["a"] = a_bank.reshape(
                num_query,
                num_pairs * budget_count,
                query_len,
            )
            transport_kwargs["b"] = b_bank.reshape(
                num_query,
                num_pairs * budget_count,
                support_len,
            )
            tam_payload = {
                "token_marginal_query_weight": query_prob_shared.detach(),
                "token_marginal_support_weight": support_prob.reshape(
                    num_query,
                    way_num,
                    shot_num,
                    support_len,
                ).detach(),
                "token_marginal/enabled": flat_cost.new_tensor(1.0),
                "token_marginal/share_qk": flat_cost.new_tensor(
                    float(self.tam_share_qk)
                ),
                "token_marginal/detach_weights": flat_cost.new_tensor(
                    float(self.tam_detach_weights)
                ),
            }
            for key, value in query_diag.items():
                suffix = key.removeprefix("token_marginal/")
                tam_payload[f"token_marginal/query_{suffix}"] = value
            for key, value in support_diag.items():
                suffix = key.removeprefix("token_marginal/")
                tam_payload[f"token_marginal/support_{suffix}"] = value
            for suffix in (
                "entropy",
                "normalized_entropy",
                "max_weight",
                "min_weight",
                "l1_from_uniform",
                "logit_std",
                "mass_error",
                "scorer_norm",
            ):
                tam_payload[f"token_marginal/{suffix}"] = 0.5 * (
                    tam_payload[f"token_marginal/query_{suffix}"]
                    + tam_payload[f"token_marginal/support_{suffix}"]
                )
            tam_payload["token_marginal/temperature"] = 0.5 * (
                tam_payload["token_marginal/query_temperature"]
                + tam_payload["token_marginal/support_temperature"]
            )
            tam_payload["token_marginal/uniform_floor"] = flat_cost.new_tensor(
                self.tam_uniform_floor
            )
        elif self.mea_marginal is not None:
            if query_weight is not None or support_weight is not None:
                raise ValueError("MEA marginal cannot be combined with explicit query/support weights")
            query_marginal, support_marginal, mea_aux = self.mea_marginal(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )
            query_prob = mea_aux["mea_query_pi"].reshape(num_query, num_pairs, query_len)
            support_prob = mea_aux["mea_support_pi"].reshape(num_query, num_pairs, support_len)
            a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)
            mea_payload = {
                key: value.to(device=flat_cost.device, dtype=flat_cost.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
                for key, value in mea_aux.items()
                if torch.is_tensor(value)
            }
            mea_payload["mea_query_marginal"] = query_marginal.reshape(num_query, way_num, shot_num, query_len)
            mea_payload["mea_support_marginal"] = support_marginal.reshape(num_query, way_num, shot_num, support_len)
        elif self.ccdm_marginal is not None:
            if query_weight is not None or support_weight is not None:
                raise ValueError("CCDM marginal cannot be combined with explicit query/support weights")
            ccdm_query_marginal, ccdm_support_marginal, ccdm_aux = self.ccdm_marginal(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )
            query_prob = ccdm_query_marginal / ccdm_query_marginal.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            support_prob = ccdm_support_marginal / ccdm_support_marginal.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)
            ccdm_payload = {
                key: value.to(device=flat_cost.device, dtype=flat_cost.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
                for key, value in ccdm_aux.items()
                if torch.is_tensor(value)
            }
        elif self.egsm_marginal is not None:
            if query_weight is not None or support_weight is not None:
                raise ValueError("EGSM marginal cannot be combined with explicit query/support weights")
            egsm_query_marginal, egsm_support_marginal, egsm_aux = self.egsm_marginal(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )
            query_prob = egsm_query_marginal / egsm_query_marginal.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            support_prob = egsm_support_marginal / egsm_support_marginal.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            egsm_rho_adaptive = egsm_aux.get("egsm_rho_adaptive")
            if egsm_rho_adaptive is not None:
                rho_scale = egsm_rho_adaptive.view(num_query, 1, 1, 1)
                a_bank = query_prob.unsqueeze(2).expand(
                    -1, -1, budget_count, -1,
                ) * rho_scale
                b_bank = support_prob.unsqueeze(2).expand(
                    -1, -1, budget_count, -1,
                ) * rho_scale
            else:
                a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
                b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)
            egsm_payload = {
                key: value.to(device=flat_cost.device, dtype=flat_cost.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
                for key, value in egsm_aux.items()
                if torch.is_tensor(value)
            }
        elif self.uses_ecot_ccem_marginal:
            if query_weight is not None or support_weight is not None:
                raise ValueError("CCEM marginal cannot be combined with explicit query/support weights")
            if support_tokens is None or query_tokens is None:
                raise ValueError("CCEM marginal requires support_tokens and query_tokens")
            query_prob, support_prob, ccem_payload = self._compute_ccem_marginal_probs(
                flat_cost,
                query_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype),
                support_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype),
                way_num=way_num,
                shot_num=shot_num,
            )
            a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)
        elif query_weight is not None:
            query_weight = query_weight.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp_min(0.0)
            if query_weight.dim() == 3 and tuple(query_weight.shape) == (num_query, num_pairs, query_len):
                query_prob = query_weight / query_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            elif tuple(query_weight.shape) == (num_query, query_len):
                query_prob = query_weight / query_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                a_bank = query_prob.reshape(num_query, 1, 1, query_len) * rho_bank.view(1, 1, budget_count, 1)
                a_bank = a_bank.expand(-1, num_pairs, -1, -1)
            else:
                raise ValueError(
                    "query_weight must have shape (NumQuery, QueryTokens) or "
                    "(NumQuery, Way*Shot, QueryTokens), "
                    f"got {tuple(query_weight.shape)}"
                )
            transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)

        if self.crs_marginal is not None:
            if support_weight is not None:
                raise ValueError("CRS and explicit support_weight cannot both shape the ECOT support marginal")
            if support_tokens is None or query_tokens is None:
                raise ValueError("CRS marginal requires support_tokens and query_tokens")
            if tuple(support_tokens.shape[:3]) != (way_num, shot_num, support_len):
                raise ValueError(
                    "support_tokens must have shape (Way, Shot, SupportTokens, Dim), "
                    f"got {tuple(support_tokens.shape)}"
                )
            if tuple(query_tokens.shape[:2]) != (num_query, query_len):
                raise ValueError(
                    "query_tokens must have shape (NumQuery, QueryTokens, Dim), "
                    f"got {tuple(query_tokens.shape)}"
                )

            # CRS-M2 preserves the ECOT budget bank.  The support probability
            # pi is computed once, then each fixed rho expert uses b = rho*pi.
            _crs_b, crs_aux = self.crs_marginal(
                support_tokens.unsqueeze(0).to(device=flat_cost.device, dtype=flat_cost.dtype),
                query_tokens.unsqueeze(0).to(device=flat_cost.device, dtype=flat_cost.dtype),
                rho=self._ecot_base_rho_tensor(flat_cost),
                spatial_hw=spatial_hw,
            )
            crs_pi = crs_aux["crs_pi"].squeeze(0)
            support_prob = crs_pi.reshape(num_query, num_pairs, support_len)
            b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)

            if query_weight is None and "crs_query_pi" in crs_aux:
                query_prob = crs_aux["crs_query_pi"].squeeze(0).reshape(num_query, num_pairs, query_len)
                a_bank = query_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
                transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)
            elif query_weight is None:
                a_bank = flat_cost.new_full(
                    (num_query, num_pairs, budget_count, query_len),
                    1.0 / float(query_len),
                )
                a_bank = a_bank * rho_bank.view(1, 1, budget_count, 1)
                transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)

            crs_payload = {
                key: value.squeeze(0) if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1 else value
                for key, value in crs_aux.items()
                if torch.is_tensor(value)
            }
        elif support_weight is not None:
            support_weight = support_weight.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp_min(0.0)
            if tuple(support_weight.shape) == (num_query, num_pairs, support_len):
                support_prob = support_weight / support_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            elif tuple(support_weight.shape) == (way_num, shot_num, support_len):
                support_prob = support_weight / support_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                support_prob = support_prob.reshape(1, num_pairs, support_len).expand(num_query, -1, -1)
                b_bank = support_prob.unsqueeze(2) * rho_bank.view(1, 1, budget_count, 1)
            else:
                raise ValueError(
                    "support_weight must have shape (Way, Shot, SupportTokens) or "
                    "(NumQuery, Way*Shot, SupportTokens), "
                    f"got {tuple(support_weight.shape)}"
                )
            transport_kwargs["b"] = b_bank.reshape(num_query, num_pairs * budget_count, support_len)

            if query_weight is None:
                a_bank = flat_cost.new_full(
                    (num_query, num_pairs, budget_count, query_len),
                    1.0 / float(query_len),
                )
                a_bank = a_bank * rho_bank.view(1, 1, budget_count, 1)
                transport_kwargs["a"] = a_bank.reshape(num_query, num_pairs * budget_count, query_len)

        if self.ecot_m2_use_aqm and "a" not in transport_kwargs:
            sim_max = (-cost_bank).max(dim=-1).values
            tau_aqm = max(float(self.ecot_m2_tau_aqm), float(self.eps))
            a_aqm = rho_bank_flat.unsqueeze(-1) * F.softmax(sim_max / tau_aqm, dim=-1)
            transport_kwargs["a"] = a_aqm
            a_norm = a_aqm / a_aqm.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            mean_aqm_a_entropy = -(a_norm * a_norm.clamp_min(self.eps).log()).sum(dim=-1).mean()

        if self.use_mncr:
            cost_bank = mncr_refine(
                cost_bank,
                temperature=self.mncr_temperature,
                lam=self.mncr_lam,
            )

        threshold_overridden = threshold_override is not None
        if threshold_override is None:
            threshold = self.transport_cost_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
        else:
            threshold = torch.as_tensor(
                threshold_override,
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            ).clamp_min(self.eps)
        noise_query_mass = None
        noise_support_mass = None
        noise_self_mass = None
        elastic_probe_cost_out = None
        elastic_probe_mass_out = None
        dustbin_augmented_plan = None
        dustbin_augmented_a = None
        dustbin_augmented_b = None
        dustbin_alpha = None
        if self.ecot_m2_score_mode == "dustbin_ot":
            if adaptive_tau_q_bank is not None:
                raise ValueError("dustbin_ot cannot be combined with token-wise UOT relaxation")
            if transport_kwargs:
                raise ValueError(
                    "dustbin_ot uses uniform token capacities and cannot be "
                    "combined with explicit or learned marginal weights"
                )
            if self.ecot_m2_dustbin_score is None:
                raise RuntimeError("dustbin_ot requires ecot_m2_dustbin_score")
            dustbin_alpha = self.ecot_m2_dustbin_score.to(
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            )
            dustbin_out = dustbin_transport_from_cost(
                cost_bank,
                threshold,
                dustbin_alpha,
                rho_bank_flat,
                temperature=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
                return_augmented=True,
            )
            plan_bank = dustbin_out["plan"]
            cost_out = dustbin_out["transport_cost"]
            mass_out = dustbin_out["real_mass"]
            dustbin_augmented_plan = dustbin_out["augmented_plan"]
            dustbin_augmented_a = dustbin_out["augmented_a"]
            dustbin_augmented_b = dustbin_out["augmented_b"]
        elif self.ecot_m2_score_mode == "elastic_ot":
            if adaptive_tau_q_bank is not None:
                raise ValueError("elastic_ot cannot be combined with token-wise UOT relaxation")
            if transport_kwargs:
                raise ValueError(
                    "elastic_ot uses uniform token capacities and cannot be combined "
                    "with explicit or learned marginal weights"
                )
            elastic_a = flat_cost.new_full(
                (num_query, num_pairs, budget_count, query_len),
                1.0 / float(query_len),
            )
            elastic_a = (
                elastic_a * rho_bank.view(1, 1, budget_count, 1)
            ).reshape(num_query, num_pairs * budget_count, query_len)
            elastic_b = flat_cost.new_full(
                (num_query, num_pairs, budget_count, support_len),
                1.0 / float(support_len),
            )
            elastic_b = (
                elastic_b * rho_bank.view(1, 1, budget_count, 1)
            ).reshape(num_query, num_pairs * budget_count, support_len)
            if self.ecot_m2_elastic_probe and not self.training:
                _, elastic_probe_cost_out, elastic_probe_mass_out = self._transport_match(
                    cost_bank,
                    rho_bank_flat,
                )
            plan_bank = sinkhorn_elastic_log(
                cost_bank - threshold,
                elastic_a,
                elastic_b,
                eps=self.sinkhorn_epsilon,
                sigma=self.ecot_m2_elastic_sigma,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
            )
            cost_out = compute_transport_cost(plan_bank, cost_bank)
            mass_out = compute_transported_mass(plan_bank)
            transport_kwargs["a"] = elastic_a
            transport_kwargs["b"] = elastic_b
        elif self.uses_ecot_noise_sink:
            if adaptive_tau_q_bank is not None:
                raise ValueError("noise-sink UOT cannot be combined with token-wise relaxation")
            if "a" in transport_kwargs:
                real_a = transport_kwargs["a"]
            else:
                real_a = flat_cost.new_full(
                    (num_query, num_pairs, budget_count, query_len),
                    1.0 / float(query_len),
                )
                real_a = real_a * rho_bank.view(1, 1, budget_count, 1)
                real_a = real_a.reshape(num_query, num_pairs * budget_count, query_len)

            if "b" in transport_kwargs:
                real_b = transport_kwargs["b"]
            else:
                real_b = flat_cost.new_full(
                    (num_query, num_pairs, budget_count, support_len),
                    1.0 / float(support_len),
                )
                real_b = real_b * rho_bank.view(1, 1, budget_count, 1)
                real_b = real_b.reshape(num_query, num_pairs * budget_count, support_len)

            cost_with_sink, query_mass_with_sink, support_mass_with_sink = self._append_noise_sink(
                cost_bank,
                real_a,
                real_b,
                rho_bank_flat,
            )
            plan_with_sink, _, _ = self._transport_match(
                cost_with_sink,
                rho_bank_flat,
                a=query_mass_with_sink,
                b=support_mass_with_sink,
            )
            plan_bank = plan_with_sink[..., :-1, :-1]
            cost_out = compute_transport_cost(plan_bank, cost_bank)
            mass_out = compute_transported_mass(plan_bank)
            noise_query_mass = plan_with_sink[..., :-1, -1].sum(dim=-1)
            noise_support_mass = plan_with_sink[..., -1, :-1].sum(dim=-1)
            noise_self_mass = plan_with_sink[..., -1, -1]
        else:
            plan_bank, cost_out, mass_out = self._transport_match(
                cost_bank,
                rho_bank_flat,
                tau_q=adaptive_tau_q_bank,
                tau_c=adaptive_tau_c_bank,
                **transport_kwargs,
            )
        plan_bank = plan_bank.reshape(num_query, way_num, shot_num, budget_count, query_len, support_len)
        shot_cost_bank = cost_out.reshape(num_query, way_num, shot_num, budget_count)
        shot_mass_bank = mass_out.reshape(num_query, way_num, shot_num, budget_count)
        D_bank = cost_bank.view(num_query, way_num, shot_num, budget_count, query_len, support_len)
        dustbin_augmented_plan_bank = (
            None
            if dustbin_augmented_plan is None
            else dustbin_augmented_plan.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                query_len + 1,
                support_len + 1,
            )
        )
        dustbin_augmented_a_bank = (
            None
            if dustbin_augmented_a is None
            else dustbin_augmented_a.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                query_len + 1,
            )
        )
        dustbin_augmented_b_bank = (
            None
            if dustbin_augmented_b is None
            else dustbin_augmented_b.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                support_len + 1,
            )
        )
        if "a" in transport_kwargs:
            target_query_bank = transport_kwargs["a"].reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                query_len,
            )
        else:
            target_query_bank = flat_cost.new_full(
                (num_query, way_num, shot_num, budget_count, query_len),
                1.0 / float(query_len),
            )
            target_query_bank = target_query_bank * rho_bank.view(1, 1, 1, budget_count, 1)
        if "b" in transport_kwargs:
            target_support_bank = transport_kwargs["b"].reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                support_len,
            )
        else:
            target_support_bank = flat_cost.new_full(
                (num_query, way_num, shot_num, budget_count, support_len),
                1.0 / float(support_len),
            )
            target_support_bank = target_support_bank * rho_bank.view(1, 1, 1, budget_count, 1)

        query_marginal_kl_bank = generalized_kl_divergence(
            plan_bank.sum(dim=-1),
            target_query_bank,
            eps=self.eps,
        )
        support_marginal_kl_bank = generalized_kl_divergence(
            plan_bank.sum(dim=-2),
            target_support_bank,
            eps=self.eps,
        )
        if adaptive_tau_q_bank is None or adaptive_tau_c_bank is None:
            query_marginal_penalty_bank = float(self.tau_q) * query_marginal_kl_bank
            support_marginal_penalty_bank = float(self.tau_c) * support_marginal_kl_bank
        else:
            adaptive_tau_q_view = adaptive_tau_q_bank.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                query_len,
            )
            adaptive_tau_c_view = adaptive_tau_c_bank.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
                support_len,
            )
            row_mass = plan_bank.sum(dim=-1)
            col_mass = plan_bank.sum(dim=-2)
            query_kl_terms = (
                row_mass
                * (
                    torch.log(row_mass.clamp_min(self.eps))
                    - torch.log(target_query_bank.clamp_min(self.eps))
                )
                - row_mass
                + target_query_bank
            )
            support_kl_terms = (
                col_mass
                * (
                    torch.log(col_mass.clamp_min(self.eps))
                    - torch.log(target_support_bank.clamp_min(self.eps))
                )
                - col_mass
                + target_support_bank
            )
            query_marginal_penalty_bank = (
                adaptive_tau_q_view * query_kl_terms
            ).sum(dim=-1)
            support_marginal_penalty_bank = (
                adaptive_tau_c_view * support_kl_terms
            ).sum(dim=-1)
        uot_classification_energy_bank = (
            shot_cost_bank
            + query_marginal_penalty_bank
            + support_marginal_penalty_bank
        )
        evidence_pair_bank: torch.Tensor | None = None
        evidence_mass_bank: torch.Tensor | None = None
        evidence_cost_bank: torch.Tensor | None = None
        evidence_background_mass_bank: torch.Tensor | None = None
        evidence_mass_for_score_bank: torch.Tensor | None = None
        if score_pair_evidence is not None:
            pair_evidence = torch.nan_to_num(
                score_pair_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if tuple(pair_evidence.shape) == (num_query, num_pairs, query_len, support_len):
                pair_evidence = pair_evidence.reshape(num_query, way_num, shot_num, query_len, support_len)
            elif tuple(pair_evidence.shape) != (num_query, way_num, shot_num, query_len, support_len):
                raise ValueError(
                    "score_pair_evidence must have shape "
                    "(NumQuery, Way*Shot, QueryTokens, SupportTokens) or "
                    "(NumQuery, Way, Shot, QueryTokens, SupportTokens), "
                    f"got {tuple(score_pair_evidence.shape)}"
                )
            evidence_pair_bank = pair_evidence.unsqueeze(3).expand(-1, -1, -1, budget_count, -1, -1)
        if (score_query_evidence is None) != (score_support_evidence is None):
            raise ValueError("score_query_evidence and score_support_evidence must be provided together")
        if (
            evidence_pair_bank is None
            and score_query_evidence is not None
            and score_support_evidence is not None
        ):
            if tuple(score_query_evidence.shape) != (num_query, query_len):
                raise ValueError(
                    "score_query_evidence must have shape (NumQuery, QueryTokens), "
                    f"got {tuple(score_query_evidence.shape)}"
                )
            query_evidence = torch.nan_to_num(
                score_query_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            support_evidence = score_support_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype)
            support_evidence = torch.nan_to_num(support_evidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            if tuple(support_evidence.shape) == (way_num, shot_num, support_len):
                support_evidence_flat = support_evidence.reshape(num_pairs, support_len)
            elif tuple(support_evidence.shape) == (num_pairs, support_len):
                support_evidence_flat = support_evidence
            else:
                raise ValueError(
                    "score_support_evidence must have shape (Way, Shot, SupportTokens) "
                    f"or (Way*Shot, SupportTokens), got {tuple(score_support_evidence.shape)}"
                )
            evidence_pair = torch.sqrt(
                (
                    query_evidence[:, None, :, None]
                    * support_evidence_flat[None, :, None, :]
                ).clamp_min(0.0)
            )
            evidence_pair_bank = evidence_pair.reshape(
                num_query,
                way_num,
                shot_num,
                query_len,
                support_len,
            ).unsqueeze(3).expand(-1, -1, -1, budget_count, -1, -1)
        if evidence_pair_bank is not None:
            evidence_mass_bank = (plan_bank * evidence_pair_bank).sum(dim=(-1, -2))
            evidence_cost_bank = (plan_bank * D_bank * evidence_pair_bank).sum(dim=(-1, -2))
            evidence_background_mass_bank = (plan_bank * (1.0 - evidence_pair_bank)).sum(dim=(-1, -2))
        noise_query_mass_bank = (
            None
            if noise_query_mass is None
            else noise_query_mass.reshape(num_query, way_num, shot_num, budget_count)
        )
        noise_support_mass_bank = (
            None
            if noise_support_mass is None
            else noise_support_mass.reshape(num_query, way_num, shot_num, budget_count)
        )
        noise_self_mass_bank = (
            None
            if noise_self_mass is None
            else noise_self_mass.reshape(num_query, way_num, shot_num, budget_count)
        )
        sink_penalty_bank = None
        if noise_query_mass_bank is not None and self.ecot_noise_sink_score_penalty > 0.0:
            sink_penalty_bank = 0.5 * (noise_query_mass_bank + noise_support_mass_bank)

        swts_pre_score: torch.Tensor | None = None
        swts_w_entropy_mean: torch.Tensor | None = None
        if self.ecot_m2_use_swts:
            swts_pre_score, swts_w_entropy_mean = self._ecot_swts_mass_removed_score_terms(
                plan_bank,
                D_bank,
                self.ecot_m2_swts_temp,
                self.eps,
            )

        mass_for_score_bank = shot_mass_bank
        mass_consensus_bank: torch.Tensor | None = None
        evidence_mass_consensus_bank: torch.Tensor | None = None
        if self.ecot_m2_mass_score_mode == "shot_consensus":
            mass_consensus_bank = shot_mass_bank.mean(dim=2, keepdim=True)
            alpha_mass = flat_cost.new_tensor(float(self.ecot_m2_consensus_mass_alpha))
            mass_for_score_bank = (1.0 - alpha_mass) * shot_mass_bank + alpha_mass * mass_consensus_bank
            if evidence_mass_bank is not None:
                evidence_mass_consensus_bank = evidence_mass_bank.mean(dim=2, keepdim=True)
                evidence_mass_for_score_bank = (
                    (1.0 - alpha_mass) * evidence_mass_bank
                    + alpha_mass * evidence_mass_consensus_bank
                )
        elif evidence_mass_bank is not None:
            evidence_mass_for_score_bank = evidence_mass_bank
        if self.ecot_m2_mass_reward_shot_scaling == "multi_shot_beta" and shot_num > 1:
            mass_reward_beta = flat_cost.new_tensor(float(self.ecot_m2_mass_reward_beta))
        else:
            mass_reward_beta = flat_cost.new_tensor(1.0)
        evidence_mass_weight = flat_cost.new_tensor(float(score_evidence_mass_weight))
        evidence_cost_weight = flat_cost.new_tensor(float(score_evidence_cost_weight))
        evidence_background_penalty = flat_cost.new_tensor(float(score_background_penalty))
        evidence_score_mode = str(score_evidence_mode).strip().lower().replace("-", "_")
        evidence_score_mode = {
            "score_only": "replace",
            "masked": "replace",
            "masked_score": "replace",
            "mass": "mass_mix",
            "mixed_mass": "mass_mix",
            "mass_reward": "mass_mix",
        }.get(evidence_score_mode, evidence_score_mode)
        if evidence_score_mode not in {"replace", "mass_mix"}:
            raise ValueError("score_evidence_mode must be 'replace' or 'mass_mix'")
        evidence_score_mix_value = float(score_evidence_mix)
        if not 0.0 <= evidence_score_mix_value <= 1.0:
            raise ValueError("score_evidence_mix must be in [0, 1]")
        evidence_score_mix = flat_cost.new_tensor(evidence_score_mix_value)
        evidence_score_mode_id = flat_cost.new_tensor(0.0 if evidence_score_mode == "replace" else 1.0)
        use_evidence_score = evidence_mass_bank is not None
        if self.ecot_m2_score_mode in {"elastic_ot", "dustbin_ot"} and use_evidence_score:
            raise ValueError(
                f"{self.ecot_m2_score_mode} cannot be combined with evidence-score "
                "replacement; its mixed-sign transport cost is the classification objective"
            )

        def ecot_budget_score(active_threshold: torch.Tensor) -> torch.Tensor:
            if self.ecot_m2_score_mode == "uot_energy":
                if use_evidence_score:
                    raise ValueError(
                        "ecot_m2_score_mode='uot_energy' cannot be combined with evidence-score replacement"
                    )
                score_terms = -uot_classification_energy_bank
            elif self.ecot_m2_cost_per_mass_score:
                replace_with_evidence = use_evidence_score and evidence_score_mode == "replace"
                active_mass_bank = evidence_mass_bank if replace_with_evidence else shot_mass_bank
                active_cost_bank = evidence_cost_bank if replace_with_evidence else shot_cost_bank
                if active_mass_bank is None or active_cost_bank is None:
                    raise RuntimeError("Evidence score state was not initialized")
                mass_denom = (
                    active_mass_bank.detach()
                    if self.ecot_m2_cost_per_mass_detach_mass
                    else active_mass_bank
                )
                score_terms = (
                    -float(self.ecot_m2_cost_per_mass_alpha)
                    * active_cost_bank
                    / (mass_denom + self.eps)
                )
                if use_evidence_score and evidence_background_mass_bank is not None:
                    background_scale = evidence_background_penalty
                    if evidence_score_mode == "mass_mix":
                        background_scale = evidence_score_mix * background_scale
                    score_terms = score_terms - background_scale * evidence_background_mass_bank
            else:
                thr = (
                    torch.zeros_like(active_threshold)
                    if self.ecot_m2_ablate_threshold_mass
                    else active_threshold
                )
                if self.ecot_m2_pst_scorer is not None and not self.ecot_m2_ablate_threshold_mass:
                    pst_uses_evidence = use_evidence_score and evidence_score_mode == "replace"
                    pst_cost_bank = evidence_cost_bank if pst_uses_evidence else shot_cost_bank
                    pst_mass_bank = evidence_mass_bank if pst_uses_evidence else shot_mass_bank
                    if pst_cost_bank is None or pst_mass_bank is None:
                        raise RuntimeError("Evidence score state was not initialized")
                    pst_features = torch.stack(
                        [
                            pst_cost_bank.detach(),
                            pst_mass_bank.detach(),
                            pst_cost_bank.detach() / pst_mass_bank.detach().clamp_min(self.eps),
                        ],
                        dim=-1,
                    )
                    network_dtype = self.ecot_m2_pst_scorer[0].weight.dtype
                    raw_delta = self.ecot_m2_pst_scorer(
                        pst_features.to(dtype=network_dtype)
                    ).squeeze(-1)
                    raw_delta = raw_delta.to(device=thr.device, dtype=thr.dtype)
                    thr = (thr * torch.exp(0.25 * torch.tanh(raw_delta))).clamp_min(self.eps)
                if self.ecot_m2_use_swts:
                    score_terms = swts_pre_score
                elif use_evidence_score:
                    if (
                        evidence_mass_for_score_bank is None
                        or evidence_cost_bank is None
                        or evidence_background_mass_bank is None
                    ):
                        raise RuntimeError("Evidence score state was not initialized")
                    if evidence_score_mode == "mass_mix":
                        pulse_mass_reward_bank = (
                            evidence_mass_weight * evidence_mass_for_score_bank
                            - evidence_background_penalty * evidence_background_mass_bank
                        ).clamp_min(0.0)
                        mixed_mass_bank = (
                            (1.0 - evidence_score_mix) * mass_for_score_bank
                            + evidence_score_mix * pulse_mass_reward_bank
                        )
                        score_terms = mass_reward_beta * thr * mixed_mass_bank - shot_cost_bank
                    else:
                        score_terms = (
                            mass_reward_beta
                            * thr
                            * (
                                evidence_mass_weight * evidence_mass_for_score_bank
                                - evidence_background_penalty * evidence_background_mass_bank
                            )
                            - evidence_cost_weight * evidence_cost_bank
                        )
                else:
                    score_terms = mass_reward_beta * thr * mass_for_score_bank - shot_cost_bank
            if sink_penalty_bank is not None:
                score_terms = score_terms - flat_cost.new_tensor(self.ecot_noise_sink_score_penalty) * sink_penalty_bank
            return self.score_scale * score_terms

        diagnostics = self._compute_ecot_diagnostics(
            ecot_budget_score(threshold),
            shot_cost_bank,
            shot_mass_bank,
        )
        controller_outputs = self.episode_controller(diagnostics)
        budget_logits = controller_outputs["budget_logits"]
        if self.ecot_uniform_budget_policy:
            budget_logits = torch.zeros_like(budget_logits)
            pi_budget = torch.full_like(budget_logits, 1.0 / float(budget_count))
        else:
            pi_budget = torch.softmax(budget_logits / float(self.ecot_budget_tau), dim=-1)
        raw_delta_threshold = controller_outputs.get("raw_delta_threshold")
        if raw_delta_threshold is not None:
            delta = raw_delta_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
            threshold = (threshold * torch.exp(0.25 * torch.tanh(delta))).clamp_min(self.eps)

        # ECOT variants are not learned-mass variants. They preserve the
        # low-variance fixed-budget inductive bias of J-FSL by evaluating a
        # deterministic bank of fixed UOT budgets and learning only an
        # episode-level policy over these stable experts.
        budget_scores = ecot_budget_score(threshold)
        base_idx = self._ecot_base_idx_value()
        base_rho = self._ecot_base_rho_tensor(flat_cost)
        base_score = budget_scores[..., base_idx]
        pi_view = pi_budget.view(1, 1, 1, budget_count)
        ecot_score = (pi_view * budget_scores).sum(dim=-1)
        lambda_ecot = self.ecot_lambda.to(device=flat_cost.device, dtype=flat_cost.dtype)
        shot_cost_diag = (pi_view * shot_cost_bank).sum(dim=-1)
        shot_mass_diag = (pi_view * shot_mass_bank).sum(dim=-1)
        shot_mass_for_score_diag = (pi_view * mass_for_score_bank).sum(dim=-1)
        evidence_mass_diag = None if evidence_mass_bank is None else (pi_view * evidence_mass_bank).sum(dim=-1)
        evidence_cost_diag = None if evidence_cost_bank is None else (pi_view * evidence_cost_bank).sum(dim=-1)
        evidence_background_mass_diag = (
            None
            if evidence_background_mass_bank is None
            else (pi_view * evidence_background_mass_bank).sum(dim=-1)
        )
        noise_query_mass_diag = None if noise_query_mass_bank is None else (pi_view * noise_query_mass_bank).sum(dim=-1)
        noise_support_mass_diag = (
            None if noise_support_mass_bank is None else (pi_view * noise_support_mass_bank).sum(dim=-1)
        )
        noise_self_mass_diag = None if noise_self_mass_bank is None else (pi_view * noise_self_mass_bank).sum(dim=-1)

        cp_payload: dict[str, torch.Tensor] = {}
        tau_shot = None
        if self.uses_cp_ecot:
            # CP-ECOT preserves consensus by pooling support shots within each
            # fixed budget before applying the episode-conditioned budget mix.
            (
                class_budget_scores,
                class_budget_cost,
                class_budget_mass,
                budget_shot_pool_weights,
                tau_k,
            ) = self._pool_cp_ecot_class_budget_scores(
                budget_scores,
                shot_cost_bank,
                shot_mass_bank,
            )
            class_base_score = class_budget_scores[..., base_idx]
            pi_class_view = pi_budget.view(1, 1, budget_count)
            class_mix_score = (pi_class_view * class_budget_scores).sum(dim=-1)
            lambda_k = lambda_ecot / math.sqrt(float(shot_num))
            logits = class_base_score + lambda_k * (class_mix_score - class_base_score)
            transport_cost = (pi_class_view * class_budget_cost).sum(dim=-1)
            transport_mass = (pi_class_view * class_budget_mass).sum(dim=-1)
            shot_logits = base_score
            shot_pool_weights = budget_shot_pool_weights[..., base_idx]
            cp_payload.update(
                {
                    "ecot_class_budget_scores": class_budget_scores,
                    "ecot_class_budget_transport_cost": class_budget_cost,
                    "ecot_class_budget_transported_mass": class_budget_mass,
                    "ecot_class_base_score": class_base_score,
                    "ecot_class_mix_score": class_mix_score,
                    "ecot_lambda_k": lambda_k,
                    "ecot_tau_k": tau_k,
                }
            )
            identity_loss = (logits - class_base_score).pow(2).mean()
        else:
            shot_logits = base_score + lambda_ecot * (ecot_score - base_score)
            if self.ccdm_entropy_shot_weight is not None and "ccdm_support_entropy" in ccdm_payload:
                lambda_h = F.softplus(self.ccdm_entropy_shot_weight).to(
                    device=shot_logits.device, dtype=shot_logits.dtype
                )
                shot_logits = shot_logits - lambda_h * ccdm_payload["ccdm_support_entropy"]
            logits, transport_cost, transport_mass, shot_pool_weights, tau_shot = self._pool_ecot_shot_scores(
                shot_logits,
                shot_cost_diag,
                shot_mass_diag,
                controller_outputs.get("raw_tau_shot"),
            )
            identity_loss = (shot_logits - base_score).pow(2).mean()

        elastic_probe_logits = None
        elastic_probe_shot_score = None
        if elastic_probe_cost_out is not None and elastic_probe_mass_out is not None:
            probe_cost_bank = elastic_probe_cost_out.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
            )
            probe_mass_bank = elastic_probe_mass_out.reshape(
                num_query,
                way_num,
                shot_num,
                budget_count,
            )
            probe_budget_scores = self.score_scale * (
                threshold * probe_mass_bank - probe_cost_bank
            )
            probe_base_score = probe_budget_scores[..., base_idx]
            probe_mix_score = (pi_view * probe_budget_scores).sum(dim=-1)
            elastic_probe_shot_score = (
                probe_base_score + lambda_ecot * (probe_mix_score - probe_base_score)
            )
            probe_cost_diag = (pi_view * probe_cost_bank).sum(dim=-1)
            probe_mass_diag = (pi_view * probe_mass_bank).sum(dim=-1)
            elastic_probe_logits, _, _, _, _ = self._pool_ecot_shot_scores(
                elastic_probe_shot_score,
                probe_cost_diag,
                probe_mass_diag,
                controller_outputs.get("raw_tau_shot"),
            )

        uot_energy_shot_score = (
            -self.score_scale * uot_classification_energy_bank[..., base_idx]
        )
        if uot_energy_shot_score.shape[-1] == 1:
            uot_energy_logits = uot_energy_shot_score.squeeze(-1)
        else:
            uot_energy_logits, _, _, _, _ = self._pool_ecot_shot_scores(
                uot_energy_shot_score,
                shot_cost_diag,
                shot_mass_diag,
                controller_outputs.get("raw_tau_shot"),
            )

        plan_diag = (pi_view.unsqueeze(-1).unsqueeze(-1) * plan_bank).sum(dim=3)
        shot_rho = base_rho.expand(num_query, way_num, shot_num).clone()
        policy_entropy = -(pi_budget.clamp_min(self.eps) * pi_budget.clamp_min(self.eps).log()).sum()
        normalized_policy_entropy = policy_entropy / math.log(float(max(budget_count, 2)))
        effective_rho = (pi_budget * rho_bank).sum()
        peak_rho = rho_bank[torch.argmax(pi_budget)]
        crs_aux_loss = crs_payload.get("crs_aux_loss", identity_loss.new_zeros(()))
        mea_aux_loss = mea_payload.get("mea_aux_loss", identity_loss.new_zeros(()))
        ccdm_aux_loss = ccdm_payload.get("ccdm_aux_loss", identity_loss.new_zeros(()))
        egsm_aux_loss = egsm_payload.get("egsm_aux_loss", identity_loss.new_zeros(()))
        ecot_aux_loss = (
            self.ecot_identity_reg * identity_loss + crs_aux_loss + mea_aux_loss + ccdm_aux_loss + egsm_aux_loss
        )

        payload: dict[str, torch.Tensor] = {
            "logits": logits,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": shot_rho,
            "shot_rho": shot_rho,
            "shot_transport_cost": shot_cost_diag,
            "shot_transported_mass": shot_mass_diag,
            "shot_logits": shot_logits,
            "shot_pool_weights": shot_pool_weights,
            "transport_plan": plan_diag,
            "ecot_pi_budget": pi_budget,
            "ecot_budget_logits": budget_logits,
            "ecot_lambda": lambda_ecot,
            "ecot_base_idx": flat_cost.new_tensor(base_idx, dtype=torch.long),
            "ecot_rho_bank": rho_bank,
            "ecot_effective_rho": effective_rho,
            "ecot_peak_rho": peak_rho,
            "ecot_base_rho": base_rho.detach(),
            "ecot_budget_shift_from_base": effective_rho - base_rho,
            "ecot_budget_base_weight": pi_budget[base_idx],
            "ecot_base_score": base_score,
            "ecot_score": ecot_score,
            "ecot_budget_scores": budget_scores,
            "ecot_shot_transport_cost_bank": shot_cost_bank,
            "ecot_shot_transported_mass_bank": shot_mass_bank,
            "ecot_uot_classification_energy_bank": uot_classification_energy_bank,
            "ecot_uot_query_kl_bank": query_marginal_kl_bank,
            "ecot_uot_support_kl_bank": support_marginal_kl_bank,
            "ecot_uot_query_penalty_bank": query_marginal_penalty_bank,
            "ecot_uot_support_penalty_bank": support_marginal_penalty_bank,
            "ecot_uot_energy_shot_score": uot_energy_shot_score,
            "ecot_uot_energy_logits": uot_energy_logits,
            "ecot_diagnostics": diagnostics,
            "ecot_identity_loss": identity_loss,
            "ecot_policy_entropy": policy_entropy,
            "ecot_policy_entropy_normalized": normalized_policy_entropy,
            "ecot_policy_entropy_loss": -self.ecot_policy_entropy_reg * policy_entropy,
            "ecot_aux_loss": ecot_aux_loss,
            "uot_energy/enabled": flat_cost.new_tensor(
                float(self.ecot_m2_score_mode == "uot_energy")
            ),
            "uot_energy/query_kl": query_marginal_kl_bank.mean().detach(),
            "uot_energy/support_kl": support_marginal_kl_bank.mean().detach(),
            "uot_energy/transport_cost": shot_cost_bank.mean().detach(),
            "uot_energy/classification_energy": uot_classification_energy_bank.mean().detach(),
            "uot_energy/marginal_penalty_share": (
                (
                    query_marginal_penalty_bank
                    + support_marginal_penalty_bank
                ).mean()
                / uot_classification_energy_bank.abs().mean().clamp_min(self.eps)
            ).detach(),
        }
        if self.ecot_m2_score_mode == "dustbin_ot":
            if (
                dustbin_augmented_plan_bank is None
                or dustbin_augmented_a_bank is None
                or dustbin_augmented_b_bank is None
                or dustbin_alpha is None
            ):
                raise RuntimeError("dustbin_ot diagnostics were not initialized")
            active_score_bank = threshold - D_bank
            dustbin_diag = dustbin_transport_diagnostics(
                plan_bank,
                dustbin_augmented_plan_bank,
                dustbin_augmented_a_bank,
                dustbin_augmented_b_bank,
                active_score_bank,
                rho_bank.view(1, 1, 1, budget_count),
                eps=self.eps,
            )
            dustbin_real_mass_fraction_bank = dustbin_diag["real_mass_fraction"]
            dustbin_query_fraction_bank = dustbin_diag["query_dustbin_fraction"]
            dustbin_support_fraction_bank = dustbin_diag["support_dustbin_fraction"]
            dustbin_score_terms = threshold * shot_mass_bank - shot_cost_bank
            dustbin_real_mass_fraction_diag = (
                pi_view * dustbin_real_mass_fraction_bank
            ).sum(dim=-1)
            dustbin_query_fraction_diag = (pi_view * dustbin_query_fraction_bank).sum(dim=-1)
            dustbin_support_fraction_diag = (pi_view * dustbin_support_fraction_bank).sum(dim=-1)
            dustbin_class_real_fraction = (
                shot_pool_weights * dustbin_real_mass_fraction_diag
            ).sum(dim=-1)
            dustbin_class_query_fraction = (
                shot_pool_weights * dustbin_query_fraction_diag
            ).sum(dim=-1)
            dustbin_class_support_fraction = (
                shot_pool_weights * dustbin_support_fraction_diag
            ).sum(dim=-1)
            payload.update(
                {
                    "dustbin_real_mass": transport_mass,
                    "dustbin_score": logits,
                    "dustbin_real_mass_fraction": dustbin_class_real_fraction,
                    "dustbin_query_dustbin_fraction": dustbin_class_query_fraction,
                    "dustbin_support_dustbin_fraction": dustbin_class_support_fraction,
                    "dustbin_real_mass_fraction_bank": dustbin_real_mass_fraction_bank,
                    "dustbin_query_dustbin_fraction_bank": dustbin_query_fraction_bank,
                    "dustbin_support_dustbin_fraction_bank": dustbin_support_fraction_bank,
                    "dustbin_score_bank": dustbin_score_terms,
                    "dustbin_ot/enabled": flat_cost.new_tensor(1.0),
                    "dustbin_ot/real_mass_fraction": (
                        dustbin_real_mass_fraction_bank.mean().detach()
                    ),
                    "dustbin_ot/query_dustbin_fraction": (
                        dustbin_query_fraction_bank.mean().detach()
                    ),
                    "dustbin_ot/support_dustbin_fraction": (
                        dustbin_support_fraction_bank.mean().detach()
                    ),
                    "dustbin_ot/dustbin_score": dustbin_alpha.detach(),
                    "dustbin_ot/accepted_edge_score_mean": (
                        dustbin_diag["accepted_edge_score_mean"].mean().detach()
                    ),
                    "dustbin_ot/rejected_query_score_mean": (
                        dustbin_diag["rejected_query_score_mean"].mean().detach()
                    ),
                    "dustbin_ot/row_residual": dustbin_diag["row_residual"].max().detach(),
                    "dustbin_ot/column_residual": (
                        dustbin_diag["column_residual"].max().detach()
                    ),
                    "dustbin_ot/mass_residual": dustbin_diag["mass_residual"].max().detach(),
                    "dustbin_ot/score_identity_error": (
                        budget_scores / float(self.score_scale) - dustbin_score_terms
                    ).abs().max().detach(),
                }
            )
        if self.ecot_m2_score_mode == "elastic_ot":
            elastic_objective_bank = shot_cost_bank - threshold * shot_mass_bank
            elastic_capacity = elastic_capacity_residuals(
                plan_bank,
                target_query_bank,
                target_support_bank,
            )
            beneficial_mass = (
                plan_bank * (D_bank < threshold).to(dtype=plan_bank.dtype)
            ).sum(dim=(-1, -2))
            transported_fraction = shot_mass_bank / rho_bank.view(
                1,
                1,
                1,
                budget_count,
            ).clamp_min(self.eps)
            payload.update(
                {
                    "elastic_ot_objective_bank": elastic_objective_bank,
                    "elastic_ot_transported_fraction_bank": transported_fraction,
                    "elastic_ot/enabled": flat_cost.new_tensor(1.0),
                    "elastic_ot/transported_fraction": transported_fraction.mean().detach(),
                    "elastic_ot/unmatched_fraction": (
                        1.0 - transported_fraction
                    ).mean().detach(),
                    "elastic_ot/beneficial_edge_mass_share": (
                        beneficial_mass / shot_mass_bank.clamp_min(self.eps)
                    ).mean().detach(),
                    "elastic_ot/row_capacity_violation": (
                        elastic_capacity["row_violation"].max().detach()
                    ),
                    "elastic_ot/column_capacity_violation": (
                        elastic_capacity["column_violation"].max().detach()
                    ),
                    "elastic_ot/objective": elastic_objective_bank.mean().detach(),
                    "elastic_ot/score_identity_error": (
                        budget_scores / float(self.score_scale) + elastic_objective_bank
                    ).abs().max().detach(),
                }
            )
            if elastic_probe_logits is not None and elastic_probe_shot_score is not None:
                payload["elastic_probe_uniform_uot_logits"] = elastic_probe_logits
                payload["elastic_probe_uniform_uot_shot_score"] = elastic_probe_shot_score
        if self.ecot_m2_mass_score_mode == "shot_consensus":
            payload.update(
                {
                    "shot_mass_for_score": shot_mass_for_score_diag,
                    "ecot_m2_mass_for_score_bank": mass_for_score_bank,
                    "ecot_m2_mass_consensus_bank": mass_consensus_bank.expand_as(shot_mass_bank),
                    "ecot_m2_consensus_mass_alpha": flat_cost.new_tensor(
                        float(self.ecot_m2_consensus_mass_alpha)
                    ),
                }
            )
        if use_evidence_score:
            if (
                evidence_mass_bank is None
                or evidence_cost_bank is None
                or evidence_background_mass_bank is None
                or evidence_mass_diag is None
                or evidence_cost_diag is None
                or evidence_background_mass_diag is None
            ):
                raise RuntimeError("Evidence score diagnostics were not initialized")
            payload.update(
                {
                    "shot_evidence_score_mass": evidence_mass_diag,
                    "shot_evidence_score_cost": evidence_cost_diag,
                    "shot_evidence_score_background_mass": evidence_background_mass_diag,
                    "evidence_score_mass_bank": evidence_mass_bank,
                    "evidence_score_cost_bank": evidence_cost_bank,
                    "evidence_score_background_mass_bank": evidence_background_mass_bank,
                    "evidence_score_mass_for_score_bank": evidence_mass_for_score_bank,
                    "evidence_score_mass_weight": evidence_mass_weight.detach(),
                    "evidence_score_cost_weight": evidence_cost_weight.detach(),
                    "evidence_score_background_penalty": evidence_background_penalty.detach(),
                    "evidence_score_mix": evidence_score_mix.detach(),
                    "evidence_score_mode_id": evidence_score_mode_id.detach(),
                    "evidence_score_pair_mean": evidence_pair_bank.mean().detach(),
                    "evidence_score_pair_peak": evidence_pair_bank.amax(dim=(-1, -2)).mean().detach(),
                }
            )
            if evidence_mass_consensus_bank is not None:
                payload["evidence_score_mass_consensus_bank"] = evidence_mass_consensus_bank.expand_as(
                    evidence_mass_bank
                )
        if self.ecot_m2_mass_reward_shot_scaling != "none":
            payload["ecot_m2_mass_reward_beta_effective"] = mass_reward_beta
        if swts_w_entropy_mean is not None:
            payload["mean_ecot_m2_swts_w_entropy"] = swts_w_entropy_mean.detach()
        if mean_aqm_a_entropy is not None:
            payload["mean_aqm_a_entropy"] = mean_aqm_a_entropy.detach()
        payload.update(crs_payload)
        payload.update(mea_payload)
        payload.update(ccdm_payload)
        payload.update(egsm_payload)
        payload.update(ccem_payload)
        if tam_payload:
            base_target_query = target_query_bank[..., base_idx, :]
            base_target_support = target_support_bank[..., base_idx, :]
            base_plan = plan_bank[..., base_idx, :, :]
            query_l1 = (
                base_plan.sum(dim=-1) - base_target_query
            ).abs().sum(dim=-1)
            support_l1 = (
                base_plan.sum(dim=-2) - base_target_support
            ).abs().sum(dim=-1)
            retained_fraction = shot_mass_bank[..., base_idx] / base_rho.clamp_min(
                self.eps
            )
            tam_payload.update(
                {
                    "token_marginal/query_l1_drift": query_l1.mean().detach(),
                    "token_marginal/support_l1_drift": support_l1.mean().detach(),
                    "token_marginal/l1_drift": (
                        0.5 * (query_l1.mean() + support_l1.mean())
                    ).detach(),
                    "token_marginal/transported_mass_fraction": (
                        retained_fraction.mean().detach()
                    ),
                    "token_marginal/transported_mass_fraction_min": (
                        retained_fraction.amin().detach()
                    ),
                }
            )
            payload.update(tam_payload)
        payload.update(cp_payload)
        if noise_query_mass_bank is not None:
            payload.update(
                {
                    "ecot_noise_sink_query_mass_bank": noise_query_mass_bank,
                    "ecot_noise_sink_support_mass_bank": noise_support_mass_bank,
                    "ecot_noise_sink_self_mass_bank": noise_self_mass_bank,
                    "ecot_noise_sink_query_mass": noise_query_mass_diag,
                    "ecot_noise_sink_support_mass": noise_support_mass_diag,
                    "ecot_noise_sink_self_mass": noise_self_mass_diag,
                    "ecot_noise_sink_cost": self.noise_sink_cost.to(device=flat_cost.device, dtype=flat_cost.dtype),
                    "ecot_noise_sink_score_penalty": flat_cost.new_tensor(self.ecot_noise_sink_score_penalty),
                }
            )
        if tau_shot is not None:
            payload["ecot_tau_shot"] = tau_shot
        if threshold_overridden:
            payload["ecot_threshold"] = threshold.detach()
            payload["transport_cost_threshold"] = threshold.detach()
        elif self.ecot_m2_score_mode == "elastic_ot":
            payload["ecot_threshold"] = threshold
        elif raw_delta_threshold is not None:
            payload["ecot_threshold"] = threshold
            payload["ecot_raw_delta_threshold"] = raw_delta_threshold.to(
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            )
        return payload

    def _compute_q_adaptive_threshold(
        self,
        q_features: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        base_threshold = self.transport_cost_threshold.to(device=reference.device, dtype=reference.dtype)
        if self.q_threshold_scorer is None:
            return base_threshold.expand_as(reference)
        standardized = self._standardize_over_shots(q_features)
        network_dtype = self.q_threshold_scorer.weight.dtype
        raw_delta = self.q_threshold_scorer(standardized.to(dtype=network_dtype)).squeeze(-1)
        raw_delta = raw_delta.to(device=reference.device, dtype=reference.dtype)
        multiplier = torch.exp(0.25 * torch.tanh(raw_delta))
        return (base_threshold * multiplier).clamp_min(self.eps)

    def _compute_structural_cover_logits(
        self,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
        shot_rho: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        explained_mass = shot_transport_mass.clamp_min(self.eps)
        shot_explanation_distance = shot_transport_cost / explained_mass
        shot_coverage_ratio = None
        if self.uses_coverage_regularized_cover:
            if shot_rho is None:
                raise ValueError("Coverage-regularized structural cover requires shot-wise rho budgets.")
            shot_rho = shot_rho.to(device=shot_transport_mass.device, dtype=shot_transport_mass.dtype)
            shot_coverage_ratio = (shot_transport_mass / shot_rho.clamp_min(self.eps)).clamp(0.0, 1.0)
            shot_explanation_distance = shot_explanation_distance + self.coverage_penalty_weight * (
                1.0 - shot_coverage_ratio
            )
        shot_logits = -self.score_scale * shot_explanation_distance
        if shot_logits.shape[-1] == 1:
            shot_cover_weights = torch.ones_like(shot_logits)
        else:
            shot_cover_weights = torch.softmax(shot_logits, dim=-1)

        cover_logits = torch.logsumexp(shot_logits, dim=-1) - math.log(float(max(shot_logits.shape[-1], 1)))
        cover_distance = -cover_logits / float(self.score_scale)
        cover_mass = (shot_cover_weights * shot_transport_mass).sum(dim=-1)
        return (
            cover_logits,
            cover_distance,
            cover_mass,
            shot_cover_weights,
            shot_explanation_distance,
            shot_coverage_ratio,
        )

    def _compute_hlm_aux_losses(
        self,
        shot_rho: torch.Tensor,
        shot_transport_mass: torch.Tensor,
        shot_pool_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = shot_rho.new_zeros(())
        if not self.uses_hlm_mass:
            return zero, zero, zero

        target = shot_rho.new_tensor(self.hlm_init_mass)
        rho_loss = (shot_rho - target).pow(2).mean()

        if self.hlm_lambda_eff > 0.0:
            efficiency = shot_transport_mass / shot_rho.clamp_min(self.eps)
            eff_loss = (self.hlm_eff_margin - efficiency).clamp_min(0.0).mean()
        else:
            eff_loss = zero

        if self.hlm_lambda_shot_cov > 0.0 and shot_pool_weights is not None and shot_pool_weights.shape[-1] > 1:
            weights = shot_pool_weights.clamp_min(self.eps)
            entropy = -(weights * weights.log()).sum(dim=-1) / math.log(float(weights.shape[-1]))
            shot_cov_loss = (self.hlm_shot_entropy_floor - entropy).clamp_min(0.0).mean()
        else:
            shot_cov_loss = zero
        return rho_loss, eff_loss, shot_cov_loss

    def _compute_egtw_token_marginals(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if query_tokens.dim() != 3 or support_tokens.dim() != 4:
            raise ValueError(
                "EGTW token marginals expect query/support tokens shaped "
                "(NumQuery, QueryTokens, Dim) and (Way, Shot, SupportTokens, Dim)"
            )
        num_query, query_len, embed_dim = query_tokens.shape
        way_num, shot_num, support_len, support_dim = support_tokens.shape
        if support_dim != embed_dim:
            raise ValueError(f"Query/support token dims must match, got {embed_dim} vs {support_dim}")

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens.dtype
        query_cast = query_tokens.to(dtype=calc_dtype)
        support_cast = support_tokens.to(dtype=calc_dtype)
        tau = self.egtw_tau.to(device=query_cast.device, dtype=query_cast.dtype).clamp_min(float(self.egtw_eps))
        discriminative_weight = self.egtw_lambda.to(device=query_cast.device, dtype=query_cast.dtype)
        attention_temperature = self.egtw_attention_temperature.to(
            device=query_cast.device,
            dtype=query_cast.dtype,
        ).clamp_min(float(self.egtw_eps))
        support_similarity_weight = query_cast.new_tensor(self.egtw_support_similarity_weight)
        uniform_mix = query_cast.new_tensor(self.egtw_uniform_mix)
        rho0 = query_cast.new_tensor(self.fixed_mass)
        log_eps = float(self.egtw_eps)

        all_support = support_cast.reshape(way_num * shot_num * support_len, embed_dim)
        sim_q = torch.matmul(query_cast, all_support.transpose(0, 1))
        attn_q = torch.softmax(sim_q / attention_temperature, dim=-1)
        attn_q_blocks = attn_q.reshape(num_query, query_len, way_num, shot_num, support_len)
        class_attention = attn_q_blocks.sum(dim=(-1, -2))
        log_class_attention = torch.log(class_attention.clamp_min(log_eps))
        class_entropy = -(class_attention * log_class_attention).sum(dim=-1)
        class_confidence = math.log(float(way_num)) - class_entropy
        query_importance = log_class_attention + discriminative_weight * class_confidence.unsqueeze(-1)
        query_mass = torch.softmax(query_importance.transpose(1, 2) / tau, dim=-1) * rho0

        sim_s = torch.einsum("wkld,ntd->nwklt", support_cast, query_cast)
        attn_s = torch.softmax(sim_s / attention_temperature, dim=-1)
        support_entropy = -(attn_s * torch.log(attn_s.clamp_min(log_eps))).sum(dim=-1)
        support_confidence = math.log(float(query_len)) - support_entropy
        support_relevance = sim_s.max(dim=-1).values
        support_importance = support_similarity_weight * support_relevance + discriminative_weight * support_confidence
        support_mass = torch.softmax(support_importance / tau, dim=-1) * rho0

        if self.egtw_uniform_mix > 0.0:
            query_uniform = query_mass.new_full(query_mass.shape, self.fixed_mass / float(query_len))
            support_uniform = support_mass.new_full(support_mass.shape, self.fixed_mass / float(support_len))
            query_mass = (1.0 - uniform_mix) * query_mass + uniform_mix * query_uniform
            support_mass = (1.0 - uniform_mix) * support_mass + uniform_mix * support_uniform

        if self.egtw_detach_masses:
            query_mass = query_mass.detach()
            support_mass = support_mass.detach()

        if __debug__:
            expected_query_mass = query_mass.new_full(query_mass.shape[:-1], self.fixed_mass)
            expected_support_mass = support_mass.new_full(support_mass.shape[:-1], self.fixed_mass)
            torch._assert(torch.isfinite(query_mass).all(), "EGTW query masses contain non-finite values")
            torch._assert(torch.isfinite(support_mass).all(), "EGTW support masses contain non-finite values")
            torch._assert(
                torch.isclose(query_mass.sum(dim=-1), expected_query_mass, atol=1e-5, rtol=1e-4).all(),
                "EGTW query masses must sum to the fixed mass budget",
            )
            torch._assert(
                torch.isclose(support_mass.sum(dim=-1), expected_support_mass, atol=1e-5, rtol=1e-4).all(),
                "EGTW support masses must sum to the fixed mass budget",
            )

        diagnostics = {
            "egtw_class_attention": class_attention.to(dtype=query_tokens.dtype),
            "egtw_class_entropy": class_entropy.to(dtype=query_tokens.dtype),
            "egtw_support_entropy": support_entropy.to(dtype=support_tokens.dtype),
            "egtw_support_relevance": support_relevance.to(dtype=support_tokens.dtype),
            "egtw_support_importance": support_importance.to(dtype=support_tokens.dtype),
        }
        return (
            query_mass.to(dtype=query_tokens.dtype),
            support_mass.to(dtype=support_tokens.dtype),
            diagnostics,
        )

    def _compute_hyperbolic_token_marginals(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        rho: torch.Tensor,
        ball: PoincareBall | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens_hyp.dim() != 3 or support_tokens_hyp.dim() != 3:
            raise ValueError(
                "Hyperbolic token marginals expect query/support tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumPairs, Tokens, Dim)"
            )
        num_query = query_tokens_hyp.shape[0]
        num_pairs = support_tokens_hyp.shape[0]
        if tuple(rho.shape) != (num_query, num_pairs):
            raise ValueError(
                "rho must have shape (NumQuery, NumPairs), "
                f"got {tuple(rho.shape)} vs {(num_query, num_pairs)}"
            )

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        support_cast = support_tokens_hyp.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast) if ball is None else ball
        query_cast = ball.project(query_cast)
        support_cast = ball.project(support_cast)

        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        support_stats = summarize_hyperbolic_tokens(support_cast, ball)

        query_to_support = ball.dist(
            query_cast[:, None, :, :],
            support_stats.mean_hyp[None, :, None, :],
        )
        support_to_query = ball.dist(
            support_cast[None, :, :, :],
            query_stats.mean_hyp[:, None, None, :],
        )
        temperature = self.token_temperature.to(device=query_to_support.device, dtype=query_to_support.dtype)
        query_weights = torch.softmax(-query_to_support / temperature, dim=-1)
        support_weights = torch.softmax(-support_to_query / temperature, dim=-1)

        rho_cast = rho.to(device=query_weights.device, dtype=query_weights.dtype)
        query_mass = query_weights * rho_cast[..., None]
        support_mass = support_weights * rho_cast[..., None]
        return query_mass.to(dtype=query_tokens_hyp.dtype), support_mass.to(dtype=support_tokens_hyp.dtype)

    def _transport_match(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        tau_q: torch.Tensor | None = None,
        tau_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_query, num_way, query_tokens, class_tokens = cost.shape
        partial_transport_mass = None
        if a is None:
            base_a = cost.new_full((num_query, query_tokens), 1.0 / float(query_tokens))
            a = base_a.unsqueeze(1).expand(-1, num_way, -1)
            if self.uses_partial_transport:
                partial_transport_mass = rho.to(device=cost.device, dtype=cost.dtype)
            elif self.uses_unbalanced_transport:
                a = a * rho.unsqueeze(-1)
        else:
            if tuple(a.shape) != (num_query, num_way, query_tokens):
                raise ValueError(
                    "a must have shape (NumQuery, NumPairs, QueryTokens), "
                    f"got {tuple(a.shape)}"
                )
            a = a.to(device=cost.device, dtype=cost.dtype)

        if b is None:
            base_b = cost.new_full((num_way, class_tokens), 1.0 / float(class_tokens))
            b = base_b.unsqueeze(0).expand(num_query, -1, -1)
            if self.uses_partial_transport:
                partial_transport_mass = rho.to(device=cost.device, dtype=cost.dtype)
            elif self.uses_unbalanced_transport:
                b = b * rho.unsqueeze(-1)
        else:
            if tuple(b.shape) != (num_query, num_way, class_tokens):
                raise ValueError(
                    "b must have shape (NumQuery, NumPairs, SupportTokens), "
                    f"got {tuple(b.shape)}"
                )
            b = b.to(device=cost.device, dtype=cost.dtype)

        pair_cost = cost.reshape(num_query * num_way, query_tokens, class_tokens)
        pair_a = a.reshape(num_query * num_way, query_tokens)
        pair_b = b.reshape(num_query * num_way, class_tokens)
        if (tau_q is None) != (tau_c is None):
            raise ValueError("tau_q and tau_c must be provided together")
        pair_tau_q = None
        pair_tau_c = None
        if tau_q is not None and tau_c is not None:
            if tuple(tau_q.shape) != (num_query, num_way, query_tokens):
                raise ValueError(
                    f"tau_q must have shape {(num_query, num_way, query_tokens)}, got {tuple(tau_q.shape)}"
                )
            if tuple(tau_c.shape) != (num_query, num_way, class_tokens):
                raise ValueError(
                    f"tau_c must have shape {(num_query, num_way, class_tokens)}, got {tuple(tau_c.shape)}"
                )
            pair_tau_q = tau_q.to(device=cost.device, dtype=cost.dtype).reshape(
                num_query * num_way,
                query_tokens,
            )
            pair_tau_c = tau_c.to(device=cost.device, dtype=cost.dtype).reshape(
                num_query * num_way,
                class_tokens,
            )

        if self.uses_partial_transport:
            if pair_tau_q is not None:
                raise ValueError("partial OT does not support token-wise UOT relaxation")
            max_mass = torch.minimum(pair_a.sum(dim=-1), pair_b.sum(dim=-1))
            if partial_transport_mass is None:
                pair_transport_mass = max_mass
            else:
                pair_transport_mass = partial_transport_mass.reshape(num_query * num_way).to(
                    device=cost.device,
                    dtype=cost.dtype,
                )
                pair_transport_mass = torch.minimum(pair_transport_mass, max_mass)
            pair_plan = solve_partial_transport(
                pair_cost,
                pair_a,
                pair_b,
                transport_mass=pair_transport_mass,
                backend=self.partial_backend,
                reg=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
                exact=self.partial_exact,
                eps=self.eps,
            )
            pair_transport_cost = compute_partial_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_partial_transported_mass(pair_plan)
        elif self.ot_backend == "pot":
            if pair_tau_q is not None:
                raise ValueError("POT backend does not support token-wise UOT relaxation")
            if self.uses_unbalanced_transport:
                pair_plan = sinkhorn_unbalanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    tau_q=self.tau_q,
                    tau_c=self.tau_c,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            else:
                pair_plan = sinkhorn_balanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            pair_transport_cost = compute_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_transported_mass(pair_plan)
        else:
            if self.uses_unbalanced_transport:
                if pair_tau_q is not None and pair_tau_c is not None:
                    pair_plan = sinkhorn_unbalanced_log_adaptive(
                        pair_cost,
                        pair_a,
                        pair_b,
                        tau_q=pair_tau_q,
                        tau_c=pair_tau_c,
                        eps=self.sinkhorn_epsilon,
                        max_iter=self.sinkhorn_iterations,
                        tol=self.sinkhorn_tolerance,
                    )
                else:
                    pair_plan = sinkhorn_unbalanced_log(
                        pair_cost,
                        pair_a,
                        pair_b,
                        tau_q=self.tau_q,
                        tau_c=self.tau_c,
                        eps=self.sinkhorn_epsilon,
                        max_iter=self.sinkhorn_iterations,
                        tol=self.sinkhorn_tolerance,
                    )
            else:
                pair_plan = sinkhorn_balanced_log(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            pair_transport_cost = compute_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_transported_mass(pair_plan)
        plan = pair_plan.reshape(num_query, num_way, query_tokens, class_tokens)
        transport_cost = pair_transport_cost.reshape(num_query, num_way)
        transport_mass = pair_transport_mass.reshape(num_query, num_way)
        return plan, transport_cost, transport_mass

    def _rho_rank_loss(
        self,
        rho: torch.Tensor,
        geodesic_distance: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = rho.new_zeros(())
        if (
            not self.training
            or not self.uses_rho_rank_loss
            or self.lambda_rho_rank <= 0.0
            or geodesic_distance is None
            or rho.dim() != 3
        ):
            return zero

        distance = geodesic_distance.to(device=rho.device, dtype=rho.dtype)
        if distance.shape != rho.shape:
            raise ValueError(f"geodesic_distance must match rho shape, got {tuple(distance.shape)} vs {tuple(rho.shape)}")

        rho_flat = rho.reshape(rho.shape[0], -1)
        distance_flat = distance.detach().reshape(distance.shape[0], -1)
        distance_scale = distance_flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        distance_norm = (distance_flat - distance_flat.mean(dim=1, keepdim=True)) / distance_scale

        distance_gap = distance_norm.unsqueeze(1) - distance_norm.unsqueeze(2)
        order_weight = distance_gap.clamp_min(0.0)
        rho_gap = rho_flat.unsqueeze(2) - rho_flat.unsqueeze(1)
        penalty = F.softplus((self.rho_rank_margin - rho_gap) / self.rho_rank_temperature)
        penalty = penalty * self.rho_rank_temperature
        per_query_loss = (penalty * order_weight).sum(dim=(1, 2)) / order_weight.sum(dim=(1, 2)).clamp_min(self.eps)
        return per_query_loss.mean()

    def _forward_hrot_v_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_num = query.shape[0]
        num_pairs = way_num * shot_num

        query_euc, query_hyp, query_hw = self._encode_images(query)
        support_flat = support.reshape(num_pairs, *support.shape[-3:])
        support_euc, support_hyp, support_hw = self._encode_images(support_flat)
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        flat_cost = self._ground_cost(query_euc, support_euc)
        flat_cost, tsw_payload = self._apply_tsw_to_cost(flat_cost, query_euc, support_euc)

        support_hyp_by_shot = support_hyp.reshape(
            way_num,
            shot_num,
            support_hyp.shape[-2],
            support_hyp.shape[-1],
        )
        geodesic_features = self._build_geodesic_eam_features(query_hyp, support_hyp_by_shot)
        flat_geodesic_features = geodesic_features.reshape(query_num, num_pairs, geodesic_features.shape[-1])
        flat_rho = self.eam.forward_features(flat_geodesic_features).to(dtype=query_euc.dtype)

        query_mass, support_mass = self._compute_token_marginals(query_euc, support_euc, flat_rho)
        structure_cost = self._compute_gw_structure_cost(query_euc, support_euc)
        final_cost = flat_cost + structure_cost

        cost_with_sink, query_mass_with_sink, support_mass_with_sink = self._append_noise_sink(
            final_cost,
            query_mass,
            support_mass,
            flat_rho,
        )
        plan_with_sink, _, _ = self._transport_match(
            cost_with_sink,
            flat_rho,
            a=query_mass_with_sink,
            b=support_mass_with_sink,
        )
        plan = plan_with_sink[..., :-1, :-1]
        flat_transport_cost = (plan * final_cost).sum(dim=(-1, -2))
        flat_transport_mass = plan.sum(dim=(-1, -2))

        adaptive_threshold = self._compute_threshold(flat_rho)
        shot_logits_flat = self.score_scale * (adaptive_threshold * flat_transport_mass - flat_transport_cost)

        shot_logits = shot_logits_flat.reshape(query_num, way_num, shot_num)
        shot_transport_cost = flat_transport_cost.reshape(query_num, way_num, shot_num)
        shot_transport_mass = flat_transport_mass.reshape(query_num, way_num, shot_num)
        shot_rho = flat_rho.reshape(query_num, way_num, shot_num)

        logits = shot_logits.mean(dim=-1)
        transport_cost = shot_transport_cost.mean(dim=-1)
        transport_mass = shot_transport_mass.mean(dim=-1)

        rho_regularization = (flat_rho - self.rho_target).pow(2).mean()
        curvature_regularization = logits.new_zeros(())
        if self.lambda_curvature > 0.0:
            curvature_regularization = (self.min_curvature - self.curvature).clamp_min(0.0).pow(2)
        aux_loss = self.lambda_rho * rho_regularization + self.lambda_curvature * curvature_regularization

        if not needs_payload:
            return logits

        zero = logits.new_zeros(())
        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": transport_cost,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": flat_rho,
            "rho_regularization": rho_regularization,
            "rho_rank_loss": zero,
            "curvature_regularization": curvature_regularization,
            "curvature": self.curvature.detach().to(device=logits.device, dtype=logits.dtype),
            "hyperbolic_backend": logits.new_tensor(0.0 if self.hyperbolic_backend == "native" else 1.0),
            "ot_backend": self._ot_backend_code(logits),
            "shot_transport_cost": shot_transport_cost,
            "shot_transported_mass": shot_transport_mass,
            "shot_rho": shot_rho,
        }

        if return_aux:
            outputs.update(
                {
                    "transport_plan": plan,
                    "base_cost_matrix": flat_cost,
                    "structure_cost": structure_cost,
                    "cost_matrix": final_cost,
                    "adaptive_threshold": adaptive_threshold,
                    "shot_logits": shot_logits,
                    "query_token_mass": query_mass,
                    "support_token_mass": support_mass,
                    "structure_cost_weight": self.structure_cost_weight.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                    "noise_sink_cost": self.noise_sink_cost.to(device=logits.device, dtype=logits.dtype),
                    "query_euclidean_tokens": query_euc,
                    "support_euclidean_tokens": support_euc,
                    "query_hyperbolic_tokens": query_hyp,
                    "support_hyperbolic_tokens": support_hyp,
                }
            )
            outputs.update(self._format_tsw_payload(tsw_payload))
        return outputs

    def _forward_care_uot(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        num_pairs = way_num * shot_num
        query_euclidean, query_hyperbolic, query_prenorm, query_hw = self._encode_images_with_prenorm(query)
        support_euclidean, support_hyperbolic, support_prenorm, support_hw = self._encode_images_with_prenorm(
            support.reshape(num_pairs, *support.shape[-3:])
        )
        del support_prenorm
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        support_euclidean = support_euclidean.reshape(
            way_num,
            shot_num,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )
        support_hyperbolic = support_hyperbolic.reshape(
            way_num,
            shot_num,
            support_hyperbolic.shape[-2],
            support_hyperbolic.shape[-1],
        )
        flat_support_euclidean = support_euclidean.reshape(
            num_pairs,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )

        fisher_weight = None
        if self.care_enable_fwec:
            fisher_weight = compute_fisher_weights(
                support_euclidean,
                eps=self.care_fwec_eps,
                clamp=(self.care_fwec_w_clamp_min, self.care_fwec_w_clamp_max),
            ).to(device=query_euclidean.device, dtype=query_euclidean.dtype)
            flat_cost = self._fisher_weighted_cost(query_euclidean, flat_support_euclidean, fisher_weight)
        else:
            flat_cost = self._ground_cost(query_euclidean, flat_support_euclidean)
        flat_cost, tsw_payload = self._apply_tsw_to_cost(flat_cost, query_euclidean, flat_support_euclidean)

        query_weight = None
        query_marginal = None
        query_energy = None
        if self.care_enable_qesm:
            query_energy = query_prenorm.norm(dim=-1)
            query_prob = torch.softmax(query_energy / float(self.care_qesm_tau_e), dim=-1).detach()
            query_weight = query_prob * float(query_prob.shape[-1])
            query_marginal = query_prob * float(self.ecot_base_rho)

        ecot_payload = self._forward_ecot_budget_bank(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            query_weight=query_weight,
        )
        logits = ecot_payload["logits"]
        transport_cost = ecot_payload["transport_cost"]
        transport_mass = ecot_payload["transported_mass"]
        rho = ecot_payload["rho"]
        shot_rho = ecot_payload["shot_rho"]
        shot_transport_cost = ecot_payload["shot_transport_cost"]
        shot_transport_mass = ecot_payload["shot_transported_mass"]
        plan = ecot_payload["transport_plan"]
        cost = flat_cost.reshape(
            query.shape[0],
            way_num,
            shot_num,
            flat_cost.shape[-2],
            flat_cost.shape[-1],
        )
        transport_probe_payload = {
            key: value
            for key, value in ecot_payload.items()
            if key
            not in {
                "logits",
                "transport_cost",
                "transported_mass",
                "rho",
                "shot_rho",
                "shot_transport_cost",
                "shot_transported_mass",
                "transport_plan",
            }
        }

        care_mdr_loss = logits.new_zeros(())
        if self.training and self.care_enable_mdr and query_targets is not None:
            care_mdr_loss = compute_mdr_loss(
                shot_transport_mass,
                query_targets,
                margin=self.care_mdr_margin,
            )

        zero = logits.new_zeros(())
        rho_regularization = zero
        rho_rank_loss = zero
        hlm_rho_regularization = zero
        hlm_eff_loss = zero
        hlm_shot_cov_loss = zero
        curvature_regularization = zero
        aux_loss = transport_probe_payload.get("ecot_aux_loss", zero)
        aux_loss = aux_loss + float(self.care_mdr_lambda) * care_mdr_loss

        if not needs_payload:
            return logits

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": transport_cost,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": rho,
            "rho_regularization": rho_regularization,
            "rho_rank_loss": rho_rank_loss,
            "hlm_rho_regularization": hlm_rho_regularization,
            "hlm_eff_loss": hlm_eff_loss,
            "hlm_shot_cov_loss": hlm_shot_cov_loss,
            "curvature_regularization": curvature_regularization,
            "curvature": self.curvature.detach().to(dtype=logits.dtype),
            "mass_bonus": self._mass_reward_weight(logits).detach().to(dtype=logits.dtype),
            "transport_cost_threshold": self.transport_cost_threshold.detach().to(
                device=logits.device,
                dtype=logits.dtype,
            ),
            "hyperbolic_backend": logits.new_tensor(0.0 if self.hyperbolic_backend == "native" else 1.0),
            "ot_backend": self._ot_backend_code(logits),
            "shot_transport_cost": shot_transport_cost,
            "shot_transported_mass": shot_transport_mass,
            "shot_rho": shot_rho,
            "care_mdr_loss": care_mdr_loss,
            "care_mdr_weighted_loss": float(self.care_mdr_lambda) * care_mdr_loss,
        }
        outputs.update(transport_probe_payload)
        if fisher_weight is not None:
            outputs["care_fwec_weight"] = fisher_weight.detach()
        if query_marginal is not None:
            outputs["care_query_marginal"] = query_marginal.to(dtype=logits.dtype)
            outputs["care_query_energy"] = query_energy.detach().to(dtype=logits.dtype)
            outputs["care_qesm_tau_e"] = logits.new_tensor(self.care_qesm_tau_e)

        if return_aux:
            outputs.update(
                {
                    "transport_plan": plan,
                    "cost_matrix": cost,
                    "query_euclidean_tokens": query_euclidean,
                    "support_euclidean_tokens": support_euclidean,
                    "query_hyperbolic_tokens": query_hyperbolic,
                    "support_hyperbolic_tokens": support_hyperbolic,
                }
            )
            outputs.update(
                self._format_tsw_payload(
                    tsw_payload,
                    query_count=query.shape[0],
                    way_num=way_num,
                    shot_num=shot_num,
                )
            )
        return outputs

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.uses_hrot_v:
            return self._forward_hrot_v_episode(query, support, needs_payload=needs_payload, return_aux=return_aux)
        if self.uses_care_uot:
            return self._forward_care_uot(
                query,
                support,
                query_targets=query_targets,
                needs_payload=needs_payload,
                return_aux=return_aux,
            )

        way_num, shot_num = support.shape[:2]
        query_amp_norms = None
        support_amp_norms = None
        use_m2_episode_norm = self.variant == "J_ECOT_M2" and self.ecot_episode_feature_normalize
        if self.hrot_amp_marginals:
            if use_m2_episode_norm:
                query_proj, query_amp_norms, query_hw = self._project_and_amp_norms_from_images(query)
                support_proj, support_amp_norms, support_hw = self._project_and_amp_norms_from_images(
                    support.reshape(way_num * shot_num, *support.shape[-3:])
                )
                query_proj, support_proj = self._episode_joint_standardize_projected(
                    query_proj,
                    support_proj,
                    self.ecot_episode_feature_norm_eps,
                )
                query_euclidean, query_hyperbolic = self._euclidean_and_hyperbolic_from_projected(query_proj)
                support_euclidean, support_hyperbolic = self._euclidean_and_hyperbolic_from_projected(support_proj)
            else:
                query_euclidean, query_hyperbolic, query_amp_norms, query_hw = (
                    self._encode_images_with_amp_norms(query)
                )
                support_euclidean, support_hyperbolic, support_amp_norms, support_hw = (
                    self._encode_images_with_amp_norms(
                        support.reshape(way_num * shot_num, *support.shape[-3:])
                    )
                )
        elif use_m2_episode_norm:
            query_proj, query_hw = self._project_tokens_from_images(query)
            support_proj, support_hw = self._project_tokens_from_images(
                support.reshape(way_num * shot_num, *support.shape[-3:])
            )
            query_proj, support_proj = self._episode_joint_standardize_projected(
                query_proj,
                support_proj,
                self.ecot_episode_feature_norm_eps,
            )
            query_euclidean, query_hyperbolic = self._euclidean_and_hyperbolic_from_projected(query_proj)
            support_euclidean, support_hyperbolic = self._euclidean_and_hyperbolic_from_projected(support_proj)
        else:
            query_euclidean, query_hyperbolic, query_hw = self._encode_images(query)
            support_euclidean, support_hyperbolic, support_hw = self._encode_images(
                support.reshape(way_num * shot_num, *support.shape[-3:])
            )
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        support_euclidean = support_euclidean.reshape(
            way_num,
            shot_num,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )
        support_hyperbolic = support_hyperbolic.reshape(
            way_num,
            shot_num,
            support_hyperbolic.shape[-2],
            support_hyperbolic.shape[-1],
        )
        class_euclidean = merge_support_tokens(support_euclidean, merge_mode="concat")
        class_hyperbolic = merge_support_tokens(support_hyperbolic, merge_mode="concat")

        shot_transport_cost = None
        shot_transport_mass = None
        shot_rho = None
        shot_geodesic_distance = None
        transport_probe_payload = None
        tsw_payload = None
        if self.uses_shot_decomposed_transport:
            matching_shot_num = shot_num
            if self.uses_ecot and self.pre_transport_shot_pool:
                matching_shot_num = 1
                flat_support_euclidean = merge_support_tokens(
                    support_euclidean,
                    merge_mode=self.pre_transport_shot_pool_mode,
                )
                flat_support_hyperbolic = merge_support_tokens(
                    support_hyperbolic,
                    merge_mode=self.pre_transport_shot_pool_mode,
                )
            else:
                flat_support_euclidean = support_euclidean.reshape(
                    way_num * shot_num,
                    support_euclidean.shape[-2],
                    support_euclidean.shape[-1],
                )
                flat_support_hyperbolic = support_hyperbolic.reshape(
                    way_num * shot_num,
                    support_hyperbolic.shape[-2],
                    support_hyperbolic.shape[-1],
                )
            if self.uses_noise_calibrated_transport:
                if self.q_eam is None:
                    raise RuntimeError("HROT-Q requires a dedicated q_eam head")
                flat_cost = self._ground_cost(query_euclidean, flat_support_euclidean)
                flat_cost, tsw_payload = self._apply_tsw_to_cost(
                    flat_cost,
                    query_euclidean,
                    flat_support_euclidean,
                )
                q_geodesic_features, geodesic_features = self._build_q_geodesic_eam_features(
                    query_hyperbolic,
                    support_hyperbolic,
                )
                shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                shot_rho = self.q_eam.forward_features(q_geodesic_features)
                shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)

                query_token_mass, support_token_mass, transport_probe_payload = (
                    self._compute_noise_calibrated_token_marginals(
                        flat_cost,
                        flat_rho,
                        support_euclidean,
                        query_hyperbolic,
                        flat_support_hyperbolic,
                    )
                )
                final_cost = flat_cost
                structure_payload = None
                if self.uses_structure_consistent_transport:
                    structure_cost, structure_payload = self._compute_structure_consistency_cost(
                        flat_cost,
                        query_euclidean,
                        flat_support_euclidean,
                        flat_rho,
                        query_token_mass,
                        support_token_mass,
                    )
                    final_cost = flat_cost + structure_cost
                cost_with_sink, query_mass_with_sink, support_mass_with_sink = self._append_noise_sink(
                    final_cost,
                    query_token_mass,
                    support_token_mass,
                    flat_rho,
                )
                flat_plan_with_sink, _, _ = self._transport_match(
                    cost_with_sink,
                    flat_rho,
                    a=query_mass_with_sink,
                    b=support_mass_with_sink,
                )
                flat_plan = flat_plan_with_sink[..., :-1, :-1]
                flat_transport_cost = compute_transport_cost(flat_plan, final_cost)
                flat_transport_mass = compute_transported_mass(flat_plan)

                shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                if self.uses_structural_cover_objective:
                    (
                        logits,
                        transport_cost,
                        transport_mass,
                        shot_cover_weights,
                        shot_explanation_distance,
                        shot_coverage_ratio,
                    ) = self._compute_structural_cover_logits(
                        shot_transport_cost,
                        shot_transport_mass,
                        shot_rho=shot_rho,
                    )
                    shot_logits = -self.score_scale * shot_explanation_distance
                else:
                    h_anchor_rho = self.eam.forward_features(geodesic_features)
                    h_anchor_rho = self._normalize_rho_budget(h_anchor_rho).to(dtype=query_hyperbolic.dtype)
                    flat_h_anchor_rho = h_anchor_rho.reshape(query.shape[0], way_num * shot_num)

                    _h_anchor_plan, h_anchor_flat_cost, h_anchor_flat_mass = self._transport_match(
                        flat_cost,
                        flat_h_anchor_rho,
                    )
                    h_anchor_shot_transport_cost = h_anchor_flat_cost.reshape(query.shape[0], way_num, shot_num)
                    h_anchor_shot_transport_mass = h_anchor_flat_mass.reshape(query.shape[0], way_num, shot_num)
                    global_threshold = self.transport_cost_threshold.to(
                        device=h_anchor_shot_transport_cost.device,
                        dtype=h_anchor_shot_transport_cost.dtype,
                    )
                    h_anchor_shot_logits = self.score_scale * (
                        global_threshold * h_anchor_shot_transport_mass - h_anchor_shot_transport_cost
                    )
                    h_anchor_logits = h_anchor_shot_logits.mean(dim=-1)
                    h_anchor_transport_cost = h_anchor_shot_transport_cost.mean(dim=-1)
                    h_anchor_transport_mass = h_anchor_shot_transport_mass.mean(dim=-1)

                    adaptive_threshold = self._compute_q_adaptive_threshold(q_geodesic_features, shot_transport_cost)
                    shot_logits = self.score_scale * (adaptive_threshold * shot_transport_mass - shot_transport_cost)

                    q_logits, q_transport_cost, q_transport_mass, shot_pool_weights = self._pool_shot_scores(
                        shot_logits,
                        shot_transport_cost,
                        shot_transport_mass,
                        geodesic_features=q_geodesic_features,
                    )
                    q_mix = self.q_enhancement_mix.to(device=q_logits.device, dtype=q_logits.dtype)
                    logits = (1.0 - q_mix) * h_anchor_logits + q_mix * q_logits
                    transport_cost = (1.0 - q_mix) * h_anchor_transport_cost + q_mix * q_transport_cost
                    transport_mass = (1.0 - q_mix) * h_anchor_transport_mass + q_mix * q_transport_mass
                rho = shot_rho
                cost = final_cost.reshape(query.shape[0], way_num, shot_num, final_cost.shape[-2], final_cost.shape[-1])
                plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
                if structure_payload is not None:
                    transport_probe_payload.update(
                        {
                            "base_cost_matrix": flat_cost.reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                                flat_cost.shape[-2],
                                flat_cost.shape[-1],
                            ),
                            "structure_cost": structure_payload["structure_cost"].reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                                flat_cost.shape[-2],
                                flat_cost.shape[-1],
                            ),
                            "structure_probe_mass": structure_payload["structure_probe_mass"].reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                            ),
                            "structure_cost_weight": self.structure_cost_weight.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                        }
                    )
                transport_probe_payload.update(
                    {
                        "transport_probe_cost": transport_probe_payload["transport_probe_cost"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_mass": transport_probe_payload["transport_probe_mass"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_entropy": transport_probe_payload["transport_probe_entropy"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_min_cost": transport_probe_payload["transport_probe_min_cost"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "query_token_mass": query_token_mass.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            query_token_mass.shape[-1],
                        ),
                        "support_token_mass": support_token_mass.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "probe_query_reliability": transport_probe_payload["probe_query_reliability"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            query_token_mass.shape[-1],
                        ),
                        "probe_support_reliability": transport_probe_payload["probe_support_reliability"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "support_consensus": transport_probe_payload["support_consensus"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "query_hyperbolic_token_prior": transport_probe_payload[
                            "query_hyperbolic_token_prior"
                        ].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            query_token_mass.shape[-1],
                        ),
                        "support_hyperbolic_token_prior": transport_probe_payload[
                            "support_hyperbolic_token_prior"
                        ].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "shot_logits": shot_logits,
                        "q_eam_cross_attention": q_geodesic_features[..., -1].to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "noise_sink_query_mass": flat_plan_with_sink[..., :-1, -1].sum(dim=-1).reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "noise_sink_support_mass": flat_plan_with_sink[..., -1, :-1].sum(dim=-1).reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "noise_sink_self_mass": flat_plan_with_sink[..., -1, -1].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "token_temperature": self.token_temperature.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "token_reliability_mix": self.token_reliability_mix.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "support_consensus_mix": self.support_consensus_mix.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "hyperbolic_token_prior_mix": self.hyperbolic_token_prior_mix.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "eam_cross_attention_temperature": self.eam_cross_attention_temperature.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "noise_sink_cost": self.noise_sink_cost.detach().to(device=logits.device, dtype=logits.dtype),
                    }
                )
                if self.uses_structural_cover_objective:
                    transport_probe_payload.update(
                        {
                            "shot_cover_weights": shot_cover_weights,
                            "shot_explanation_distance": shot_explanation_distance,
                        }
                    )
                    if shot_coverage_ratio is not None:
                        transport_probe_payload["shot_coverage_ratio"] = shot_coverage_ratio
                else:
                    transport_probe_payload.update(
                        {
                            "adaptive_transport_cost_threshold": adaptive_threshold,
                            "shot_pool_weights": shot_pool_weights,
                            "q_enhanced_logits": q_logits,
                            "h_anchor_logits": h_anchor_logits,
                            "h_anchor_shot_logits": h_anchor_shot_logits,
                            "h_anchor_shot_transport_cost": h_anchor_shot_transport_cost,
                            "h_anchor_shot_transported_mass": h_anchor_shot_transport_mass,
                            "h_anchor_rho": h_anchor_rho,
                            "shot_pool_temperature": self.shot_pool_temperature.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                            "shot_pool_mix": self.shot_pool_mix.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                            "q_enhancement_mix": self.q_enhancement_mix.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                        }
                    )
            elif self.uses_hyperbolic_token_attention:
                flat_cost = self._ground_cost(query_euclidean, flat_support_euclidean)
                flat_cost, tsw_payload = self._apply_tsw_to_cost(
                    flat_cost,
                    query_euclidean,
                    flat_support_euclidean,
                )
                geodesic_features = self._build_geodesic_eam_features(query_hyperbolic, support_hyperbolic)
                shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                shot_rho = self.eam.forward_features(geodesic_features)
                shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                query_token_mass, support_token_mass = self._compute_hyperbolic_token_marginals(
                    query_hyperbolic,
                    flat_support_hyperbolic,
                    flat_rho,
                )
                flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(
                    flat_cost,
                    flat_rho,
                    a=query_token_mass,
                    b=support_token_mass,
                )

                shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                shot_logits = -self.score_scale * shot_transport_cost
                if self.uses_unbalanced_transport:
                    shot_logits = shot_logits + self._mass_reward_weight(shot_logits) * shot_transport_mass

                if self.uses_geodesic_shot_pooling:
                    logits, transport_cost, transport_mass, shot_pool_weights = self._pool_shot_scores(
                        shot_logits,
                        shot_transport_cost,
                        shot_transport_mass,
                        geodesic_features=geodesic_features,
                    )
                else:
                    shot_pool_weights = None
                    logits = shot_logits.mean(dim=-1)
                    transport_cost = shot_transport_cost.mean(dim=-1)
                    transport_mass = shot_transport_mass.mean(dim=-1)
                rho = shot_rho
                cost = flat_cost.reshape(query.shape[0], way_num, shot_num, flat_cost.shape[-2], flat_cost.shape[-1])
                plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
                transport_probe_payload = {
                    "query_token_mass": query_token_mass.reshape(
                        query.shape[0],
                        way_num,
                        shot_num,
                        query_token_mass.shape[-1],
                    ),
                    "support_token_mass": support_token_mass.reshape(
                        query.shape[0],
                        way_num,
                        shot_num,
                        support_token_mass.shape[-1],
                    ),
                    "shot_logits": shot_logits,
                    "token_temperature": self.token_temperature.detach().to(device=logits.device, dtype=logits.dtype),
                }
                if shot_pool_weights is not None:
                    transport_probe_payload.update(
                        {
                            "shot_pool_weights": shot_pool_weights,
                            "shot_pool_temperature": self.shot_pool_temperature.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                            "shot_pool_mix": self.shot_pool_mix.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                        }
                    )
            else:
                cost_query_tokens = query_euclidean
                cost_support_tokens = flat_support_euclidean
                if self.hrot_token_center:
                    support_for_center = cost_support_tokens.reshape(
                        way_num,
                        matching_shot_num,
                        cost_support_tokens.shape[-2],
                        cost_support_tokens.shape[-1],
                    )
                    support_mean = support_for_center.mean(dim=2, keepdim=True)
                    centered_support = support_for_center - support_mean
                    if self.training:
                        centered_norms = centered_support.norm(dim=-1)
                        if (centered_norms < 1e-4).any():
                            import logging
                            logging.getLogger(__name__).warning(
                                "Token centering produced near-zero vectors "
                                "(min norm %.2e)", centered_norms.min().item(),
                            )
                    cost_support_tokens = centered_support.reshape(
                        way_num * matching_shot_num,
                        cost_support_tokens.shape[-2],
                        cost_support_tokens.shape[-1],
                    )
                    if self.hrot_token_center_query:
                        query_mean = cost_query_tokens.mean(dim=1, keepdim=True)
                        cost_query_tokens = cost_query_tokens - query_mean

                flat_cost = self._ground_cost(cost_query_tokens, cost_support_tokens)
                flat_cost, tsw_payload = self._apply_tsw_to_cost(
                    flat_cost,
                    cost_query_tokens,
                    cost_support_tokens,
                )
                if self.uses_ecot:
                    nncs_support_weight = None
                    nncs_payload = None
                    if self.uses_ecot_nncs_marginal:
                        nncs_support_weight, nncs_payload = self._compute_nncs_support_payload(support_euclidean)

                    amp_query_weight = None
                    amp_support_weight = None
                    if self.hrot_amp_marginals and query_amp_norms is not None:
                        amp_query_weight = self._build_amp_marginals(
                            query_amp_norms,
                            rho=self.ecot_base_rho,
                            tau_w=self.hrot_amp_marginals_tau,
                        )
                        amp_query_weight = (
                            amp_query_weight
                            / amp_query_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                            * float(amp_query_weight.shape[-1])
                        )
                    if self.hrot_amp_marginals and support_amp_norms is not None:
                        amp_support_weight = self._build_amp_marginals(
                            support_amp_norms,
                            rho=self.ecot_base_rho,
                            tau_w=self.hrot_amp_marginals_tau,
                        )
                        amp_support_weight = (
                            amp_support_weight
                            / amp_support_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                        )
                        amp_support_weight = amp_support_weight.reshape(
                            way_num,
                            matching_shot_num,
                            amp_support_weight.shape[-1],
                        )

                    crs_support_tokens = flat_support_euclidean.reshape(
                        way_num,
                        matching_shot_num,
                        flat_support_euclidean.shape[-2],
                        flat_support_euclidean.shape[-1],
                    )
                    ecot_payload = self._forward_ecot_budget_bank(
                        flat_cost,
                        way_num=way_num,
                        shot_num=matching_shot_num,
                        support_weight=amp_support_weight if amp_support_weight is not None else nncs_support_weight,
                        query_weight=amp_query_weight,
                        support_tokens=crs_support_tokens,
                        query_tokens=query_euclidean,
                        spatial_hw=support_hw,
                    )
                    if nncs_payload is not None:
                        ecot_payload.update(nncs_payload)
                    logits = ecot_payload["logits"]
                    transport_cost = ecot_payload["transport_cost"]
                    transport_mass = ecot_payload["transported_mass"]
                    rho = ecot_payload["rho"]
                    shot_rho = ecot_payload["shot_rho"]
                    shot_transport_cost = ecot_payload["shot_transport_cost"]
                    shot_transport_mass = ecot_payload["shot_transported_mass"]
                    cost = flat_cost.reshape(
                        query.shape[0],
                        way_num,
                        matching_shot_num,
                        flat_cost.shape[-2],
                        flat_cost.shape[-1],
                    )
                    plan = ecot_payload["transport_plan"]
                    transport_probe_payload = {
                        key: value
                        for key, value in ecot_payload.items()
                        if key
                        not in {
                            "logits",
                            "transport_cost",
                            "transported_mass",
                            "rho",
                            "shot_rho",
                            "shot_transport_cost",
                            "shot_transported_mass",
                            "transport_plan",
                        }
                    }
                else:
                    transport_match_kwargs = {}
                    if self.uses_nuisance_decoupled_transport:
                        shot_rho = query_euclidean.new_full((query.shape[0], way_num, shot_num), self.fixed_mass)
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                        base_plan, base_flat_cost, base_flat_mass = self._transport_match(flat_cost, flat_rho)
                        query_nuisance, support_nuisance, ncet_confidence = self._compute_ncet_nuisance_scores(
                            flat_cost,
                            base_plan,
                            support_euclidean,
                            way_num=way_num,
                            shot_num=shot_num,
                        )
                        base_query_mass = flat_cost.new_full(
                            (query.shape[0], way_num * shot_num, flat_cost.shape[-2]),
                            1.0 / float(flat_cost.shape[-2]),
                        )
                        base_support_mass = flat_cost.new_full(
                            (query.shape[0], way_num * shot_num, flat_cost.shape[-1]),
                            1.0 / float(flat_cost.shape[-1]),
                        )
                        query_mass = base_query_mass * flat_rho.unsqueeze(-1)
                        support_mass = base_support_mass * flat_rho.unsqueeze(-1)
                        cost_with_sink, query_mass_with_sink, support_mass_with_sink, adjusted_real_cost = (
                            self._append_ncet_nuisance_sink(
                                flat_cost,
                                query_mass,
                                support_mass,
                                flat_rho,
                                query_nuisance,
                                support_nuisance,
                            )
                        )
                        flat_plan_with_sink, _, _ = self._transport_match(
                            cost_with_sink,
                            flat_rho,
                            a=query_mass_with_sink,
                            b=support_mass_with_sink,
                        )
                        flat_plan = flat_plan_with_sink[..., :-1, :-1]
                        ncet_flat_cost = compute_transport_cost(flat_plan, adjusted_real_cost)
                        ncet_flat_mass = compute_transported_mass(flat_plan)
                        query_sink_mass = flat_plan_with_sink[..., :-1, -1].sum(dim=-1)
                        support_sink_mass = flat_plan_with_sink[..., -1, :-1].sum(dim=-1)
                        null_mass = query_sink_mass + support_sink_mass

                        threshold = self.transport_cost_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
                        base_flat_logits = self.score_scale * (threshold * base_flat_mass - base_flat_cost)
                        ncet_flat_logits = self.score_scale * (threshold * ncet_flat_mass - ncet_flat_cost)
                        ncet_flat_logits = ncet_flat_logits - self.ncet_null_penalty.to(
                            device=flat_cost.device,
                            dtype=flat_cost.dtype,
                        ) * null_mass
                        gate = self.ncet_mix.to(device=flat_cost.device, dtype=flat_cost.dtype) * ncet_confidence
                        mixed_flat_logits = base_flat_logits + gate * (ncet_flat_logits - base_flat_logits)
                        mixed_flat_cost = base_flat_cost + gate * (ncet_flat_cost - base_flat_cost)
                        mixed_flat_mass = base_flat_mass + gate * (ncet_flat_mass - base_flat_mass)

                        shot_logits = mixed_flat_logits.reshape(query.shape[0], way_num, shot_num)
                        shot_transport_cost = mixed_flat_cost.reshape(query.shape[0], way_num, shot_num)
                        shot_transport_mass = mixed_flat_mass.reshape(query.shape[0], way_num, shot_num)
                        logits, transport_cost, transport_mass, shot_pool_weights = self._pool_j_shot_scores(
                            shot_logits,
                            shot_transport_cost,
                            shot_transport_mass,
                        )
                        rho = shot_rho
                        cost = cost_with_sink.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            cost_with_sink.shape[-2],
                            cost_with_sink.shape[-1],
                        )
                        plan = flat_plan_with_sink.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            flat_plan_with_sink.shape[-2],
                            flat_plan_with_sink.shape[-1],
                        )
                        transport_probe_payload = {
                            "shot_logits": shot_logits,
                            "shot_pool_weights": shot_pool_weights,
                            "ncet_base_shot_logits": base_flat_logits.reshape(query.shape[0], way_num, shot_num),
                            "ncet_refined_shot_logits": ncet_flat_logits.reshape(query.shape[0], way_num, shot_num),
                            "ncet_gate": gate.reshape(query.shape[0], way_num, shot_num),
                            "ncet_confidence": ncet_confidence.reshape(query.shape[0], way_num, shot_num),
                            "ncet_query_nuisance": query_nuisance.reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                                query_nuisance.shape[-1],
                            ),
                            "ncet_support_nuisance": support_nuisance.reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                                support_nuisance.shape[-1],
                            ),
                            "ncet_base_transport_cost": base_flat_cost.reshape(query.shape[0], way_num, shot_num),
                            "ncet_base_transported_mass": base_flat_mass.reshape(query.shape[0], way_num, shot_num),
                            "ncet_real_transport_cost": ncet_flat_cost.reshape(query.shape[0], way_num, shot_num),
                            "ncet_real_transported_mass": ncet_flat_mass.reshape(query.shape[0], way_num, shot_num),
                            "ncet_null_mass": null_mass.reshape(query.shape[0], way_num, shot_num),
                            "ncet_query_sink_mass": query_sink_mass.reshape(query.shape[0], way_num, shot_num),
                            "ncet_support_sink_mass": support_sink_mass.reshape(query.shape[0], way_num, shot_num),
                            "ncet_real_penalty": self.ncet_real_penalty.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                            "ncet_null_penalty": self.ncet_null_penalty.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                            "ncet_mix": self.ncet_mix.detach().to(device=logits.device, dtype=logits.dtype),
                            "noise_sink_cost": self.noise_sink_cost.detach().to(
                                device=logits.device,
                                dtype=logits.dtype,
                            ),
                        }
                    elif self.uses_egtw_mass:
                        egtw_query_mass, egtw_support_mass, egtw_diagnostics = self._compute_egtw_token_marginals(
                            query_euclidean,
                            support_euclidean,
                        )
                        shot_rho = query_euclidean.new_full((query.shape[0], way_num, shot_num), self.fixed_mass)
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                        flat_query_mass = (
                            egtw_query_mass[:, :, None, :]
                            .expand(-1, -1, shot_num, -1)
                            .reshape(query.shape[0], way_num * shot_num, query_euclidean.shape[-2])
                        )
                        flat_support_mass = egtw_support_mass.reshape(
                            query.shape[0],
                            way_num * shot_num,
                            support_euclidean.shape[-2],
                        )
                        transport_match_kwargs = {
                            "a": flat_query_mass,
                            "b": flat_support_mass,
                        }
                        transport_probe_payload = {
                            "egtw_query_mass": egtw_query_mass,
                            "query_token_mass": flat_query_mass.reshape(
                                query.shape[0],
                                way_num,
                                shot_num,
                                flat_query_mass.shape[-1],
                            ),
                            "support_token_mass": egtw_support_mass,
                            "egtw_tau": self.egtw_tau.detach().to(
                                device=query_euclidean.device,
                                dtype=query_euclidean.dtype,
                            ),
                            "egtw_lambda": self.egtw_lambda.detach().to(
                                device=query_euclidean.device,
                                dtype=query_euclidean.dtype,
                            ),
                            "egtw_attention_temperature": self.egtw_attention_temperature.detach().to(
                                device=query_euclidean.device,
                                dtype=query_euclidean.dtype,
                            ),
                            "egtw_support_similarity_weight": query_euclidean.new_tensor(
                                self.egtw_support_similarity_weight,
                            ),
                            "egtw_uniform_mix": query_euclidean.new_tensor(self.egtw_uniform_mix),
                        }
                        transport_probe_payload.update(egtw_diagnostics)
                    elif self.uses_hybrid_ablation_eam:
                        shot_rho = self._build_hybrid_rho_per_shot(
                            query_hyperbolic,
                            support_hyperbolic,
                            query_euclidean,
                            support_euclidean,
                        )
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                    elif self.uses_euclidean_geometric_eam:
                        shot_rho = self._build_euclidean_rho_per_shot(query_euclidean, support_euclidean)
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                    elif self.uses_transport_aware_eam:
                        shot_rho, transport_aware_features, transport_probe_payload = (
                            self._build_transport_aware_geodesic_rho_per_shot(
                                query_hyperbolic,
                                support_hyperbolic,
                                flat_cost,
                            )
                        )
                        shot_geodesic_distance = transport_aware_features[..., 0].to(dtype=query_hyperbolic.dtype)
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                    elif self.uses_geodesic_eam:
                        if self.uses_rho_rank_loss:
                            geodesic_features = self._build_geodesic_eam_features(query_hyperbolic, support_hyperbolic)
                            shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                            shot_rho = self.eam.forward_features(geodesic_features)
                            shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                        else:
                            shot_rho = self._build_geodesic_rho_per_shot(query_hyperbolic, support_hyperbolic)
                        flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                    else:
                        flat_rho = self._build_pairwise_rho(query_hyperbolic, flat_support_hyperbolic)
                    if not self.uses_nuisance_decoupled_transport:
                        flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(
                            flat_cost,
                            flat_rho,
                            **transport_match_kwargs,
                        )

                        shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                        shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                        if shot_rho is None:
                            shot_rho = flat_rho.reshape(query.shape[0], way_num, shot_num)
                        shot_logits = -self.score_scale * shot_transport_cost
                        if self.uses_unbalanced_transport:
                            shot_logits = shot_logits + self._mass_reward_weight(shot_logits) * shot_transport_mass

                        if self.variant in {"J", "JE"}:
                            logits, transport_cost, transport_mass, shot_pool_weights = self._pool_j_shot_scores(
                                shot_logits,
                                shot_transport_cost,
                                shot_transport_mass,
                            )
                            j_transport_payload = {
                                "shot_logits": shot_logits,
                                "shot_pool_weights": shot_pool_weights,
                            }
                            if transport_probe_payload is None:
                                transport_probe_payload = j_transport_payload
                            else:
                                transport_probe_payload.update(j_transport_payload)
                        else:
                            logits = shot_logits.mean(dim=-1)
                            transport_cost = shot_transport_cost.mean(dim=-1)
                            transport_mass = shot_transport_mass.mean(dim=-1)
                        rho = shot_rho
                        cost = flat_cost.reshape(query.shape[0], way_num, shot_num, flat_cost.shape[-2], flat_cost.shape[-1])
                        plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
        else:
            cost = self._class_ground_cost(
                query_euclidean,
                class_euclidean,
                query_hyperbolic,
                class_hyperbolic,
            )
            cost, tsw_payload = self._apply_tsw_to_cost(cost, query_euclidean, class_euclidean)

            rho = self._build_pairwise_rho(query_hyperbolic, class_hyperbolic)
            plan, transport_cost, transport_mass = self._transport_match(cost, rho)

            logits = -self.score_scale * transport_cost
            if self.uses_unbalanced_transport:
                logits = logits + self._mass_reward_weight(logits) * transport_mass

        rho_regularization = logits.new_zeros(())
        if (
            self.uses_learned_mass
            and not self.uses_hlm_mass
            and self.lambda_rho > 0.0
            and not self.uses_structural_cover_objective
        ):
            rho_regularization = (rho - self.rho_target).pow(2).mean()

        rho_rank_loss = self._rho_rank_loss(rho, shot_geodesic_distance)

        hlm_rho_regularization = logits.new_zeros(())
        hlm_eff_loss = logits.new_zeros(())
        hlm_shot_cov_loss = logits.new_zeros(())
        if self.uses_hlm_mass:
            hlm_shot_pool_weights = None
            if transport_probe_payload is not None:
                hlm_shot_pool_weights = transport_probe_payload.get("shot_pool_weights")
            hlm_rho_regularization, hlm_eff_loss, hlm_shot_cov_loss = self._compute_hlm_aux_losses(
                rho,
                shot_transport_mass,
                hlm_shot_pool_weights,
            )
            rho_regularization = hlm_rho_regularization

        curvature_regularization = logits.new_zeros(())
        if self.lambda_curvature > 0.0:
            curvature_regularization = (self.min_curvature - self.curvature).clamp_min(0.0).pow(2)

        base_rho_weight = 0.0 if self.uses_hlm_mass else self.lambda_rho
        aux_loss = (
            base_rho_weight * rho_regularization
            + self.lambda_rho_rank * rho_rank_loss
            + self.lambda_curvature * curvature_regularization
            + self.hlm_lambda_rho * hlm_rho_regularization
            + self.hlm_lambda_eff * hlm_eff_loss
            + self.hlm_lambda_shot_cov * hlm_shot_cov_loss
        )
        if self.uses_ecot and transport_probe_payload is not None:
            aux_loss = aux_loss + transport_probe_payload.get("ecot_aux_loss", logits.new_zeros(()))
        cost_margin_aux_loss = self._supervised_transport_cost_margin_loss(transport_cost, query_targets)
        aux_loss = aux_loss + float(self.cost_margin_aux_weight) * cost_margin_aux_loss

        if not needs_payload:
            return logits

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": transport_cost,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": rho,
            "rho_regularization": rho_regularization,
            "rho_rank_loss": rho_rank_loss,
            "cost_margin_aux_loss": cost_margin_aux_loss,
            "hlm_rho_regularization": hlm_rho_regularization,
            "hlm_eff_loss": hlm_eff_loss,
            "hlm_shot_cov_loss": hlm_shot_cov_loss,
            "curvature_regularization": curvature_regularization,
            "curvature": self.curvature.detach().to(dtype=logits.dtype),
            "mass_bonus": self._mass_reward_weight(logits).detach().to(dtype=logits.dtype),
            "transport_cost_threshold": self.transport_cost_threshold.detach().to(
                device=logits.device,
                dtype=logits.dtype,
            ),
            "hyperbolic_backend": logits.new_tensor(0.0 if self.hyperbolic_backend == "native" else 1.0),
            "ot_backend": self._ot_backend_code(logits),
        }
        if self.uses_shot_decomposed_transport:
            outputs.update(
                {
                    "shot_transport_cost": shot_transport_cost,
                    "shot_transported_mass": shot_transport_mass,
                    "shot_rho": shot_rho,
                }
            )
            if transport_probe_payload is not None:
                outputs.update(transport_probe_payload)
        if return_aux:
            if self.uses_ecot and self.pre_transport_shot_pool:
                payload_support_euclidean = merge_support_tokens(
                    support_euclidean,
                    merge_mode=self.pre_transport_shot_pool_mode,
                )
                payload_support_hyperbolic = merge_support_tokens(
                    support_hyperbolic,
                    merge_mode=self.pre_transport_shot_pool_mode,
                )
            else:
                payload_support_euclidean = support_euclidean if self.uses_shot_decomposed_transport else class_euclidean
                payload_support_hyperbolic = (
                    support_hyperbolic if self.uses_shot_decomposed_transport else class_hyperbolic
                )
            outputs.update(
                {
                    "transport_plan": plan,
                    "cost_matrix": cost,
                    "query_euclidean_tokens": query_euclidean,
                    "support_euclidean_tokens": payload_support_euclidean,
                    "query_hyperbolic_tokens": query_hyperbolic,
                    "support_hyperbolic_tokens": payload_support_hyperbolic,
                }
            )
            if self.uses_shot_decomposed_transport:
                outputs.update(
                    self._format_tsw_payload(
                        tsw_payload,
                        query_count=query.shape[0],
                        way_num=way_num,
                        shot_num=matching_shot_num,
                    )
                )
            else:
                outputs.update(self._format_tsw_payload(tsw_payload))
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
            "rho": torch.cat([item["rho"] for item in batch_outputs], dim=0),
            "rho_regularization": torch.stack([item["rho_regularization"] for item in batch_outputs]).mean(),
            "rho_rank_loss": torch.stack([item["rho_rank_loss"] for item in batch_outputs]).mean(),
            "cost_margin_aux_loss": torch.stack([item["cost_margin_aux_loss"] for item in batch_outputs]).mean(),
            "curvature_regularization": torch.stack([item["curvature_regularization"] for item in batch_outputs]).mean(),
            "curvature": torch.stack([item["curvature"] for item in batch_outputs]).mean(),
            "hyperbolic_backend": torch.stack([item["hyperbolic_backend"] for item in batch_outputs]).mean(),
            "ot_backend": torch.stack([item["ot_backend"] for item in batch_outputs]).mean(),
        }
        for key in ("hlm_rho_regularization", "hlm_eff_loss", "hlm_shot_cov_loss"):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "shot_transport_cost" in batch_outputs[0]:
            stacked["shot_transport_cost"] = torch.cat([item["shot_transport_cost"] for item in batch_outputs], dim=0)
            stacked["shot_transported_mass"] = torch.cat(
                [item["shot_transported_mass"] for item in batch_outputs],
                dim=0,
            )
            stacked["shot_rho"] = torch.cat([item["shot_rho"] for item in batch_outputs], dim=0)
        for key in (
            "ecot_base_score",
            "ecot_score",
            "ecot_budget_scores",
            "ecot_class_budget_scores",
            "ecot_class_budget_transport_cost",
            "ecot_class_budget_transported_mass",
            "ecot_class_base_score",
            "ecot_class_mix_score",
            "ecot_shot_transport_cost_bank",
            "ecot_shot_transported_mass_bank",
            "ecot_uot_classification_energy_bank",
            "ecot_uot_query_kl_bank",
            "ecot_uot_support_kl_bank",
            "ecot_uot_energy_shot_score",
            "ecot_uot_energy_logits",
            "elastic_ot_objective_bank",
            "elastic_ot_transported_fraction_bank",
            "dustbin_real_mass_fraction_bank",
            "dustbin_query_dustbin_fraction_bank",
            "dustbin_support_dustbin_fraction_bank",
            "dustbin_score_bank",
            "dcr_unadjusted_shot_logits",
            "dcr_unadjusted_ecot_base_score",
            "dcr_unadjusted_ecot_score",
            "dcr_unadjusted_ecot_budget_scores",
            "dcr_removed_positive_shot",
            "dcr_retained_positive_shot",
            "dcr_removed_positive_share_shot",
            "elastic_probe_uniform_uot_logits",
            "elastic_probe_uniform_uot_shot_score",
            "ecot_m2_mass_for_score_bank",
            "ecot_m2_mass_consensus_bank",
            "evidence_score_mass_bank",
            "evidence_score_cost_bank",
            "evidence_score_background_mass_bank",
            "evidence_score_mass_for_score_bank",
            "evidence_score_mass_consensus_bank",
            "ecot_noise_sink_query_mass_bank",
            "ecot_noise_sink_support_mass_bank",
            "ecot_noise_sink_self_mass_bank",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "ecot_pi_budget",
            "ecot_budget_logits",
            "ecot_rho_bank",
            "ecot_diagnostics",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean(dim=0)
        for key in (
            "ecot_lambda",
            "ecot_tau_shot",
            "ecot_threshold",
            "ecot_raw_delta_threshold",
            "ecot_identity_loss",
            "ecot_policy_entropy",
            "ecot_policy_entropy_normalized",
            "ecot_policy_entropy_loss",
            "ecot_aux_loss",
            "ecot_effective_rho",
            "ecot_peak_rho",
            "ecot_base_rho",
            "ecot_budget_shift_from_base",
            "ecot_budget_base_weight",
            "ecot_lambda_k",
            "ecot_tau_k",
            "mean_ecot_m2_swts_w_entropy",
            "ecot_m2_consensus_mass_alpha",
            "ecot_m2_mass_reward_beta_effective",
            "mean_aqm_a_entropy",
            "uot_energy/enabled",
            "uot_energy/query_kl",
            "uot_energy/support_kl",
            "uot_energy/transport_cost",
            "uot_energy/classification_energy",
            "uot_energy/marginal_penalty_share",
            "elastic_ot/enabled",
            "elastic_ot/transported_fraction",
            "elastic_ot/unmatched_fraction",
            "elastic_ot/beneficial_edge_mass_share",
            "elastic_ot/row_capacity_violation",
            "elastic_ot/column_capacity_violation",
            "elastic_ot/objective",
            "elastic_ot/score_identity_error",
            "dustbin_ot/enabled",
            "dustbin_ot/real_mass_fraction",
            "dustbin_ot/query_dustbin_fraction",
            "dustbin_ot/support_dustbin_fraction",
            "dustbin_ot/dustbin_score",
            "dustbin_ot/accepted_edge_score_mean",
            "dustbin_ot/rejected_query_score_mean",
            "dustbin_ot/row_residual",
            "dustbin_ot/column_residual",
            "dustbin_ot/mass_residual",
            "dustbin_ot/score_identity_error",
            "dcr/enabled",
            "dcr/beta",
            "dcr/tau",
            "dcr/alpha",
            "dcr/margin",
            "dcr/min_gate",
            "dcr/detach_gate",
            "dcr/gate_mean",
            "dcr/dustbin_gate_mean",
            "dcr/specificity_mean",
            "dcr/positive_mass_fraction",
            "dcr/removed_positive_share",
            "dcr/removed_positive_mean",
            "dcr/retained_positive_mean",
            "dcr/raw_positive_mean",
            "dcr/shot_score_delta",
            "dcr/shot_logit_delta",
            "dcr/shot_logit_delta_abs",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "ecot_base_idx" in batch_outputs[0]:
            stacked["ecot_base_idx"] = batch_outputs[0]["ecot_base_idx"]
        for key in (
            "nncs_support_marginal",
            "nncs_consistency",
            "nncs_sigma",
            "care_query_marginal",
            "care_query_energy",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "crs_support_marginal",
            "crs_b",
            "crs_pi",
            "crs_p_cross_ref",
            "crs_p_ssm",
            "crs_entropy",
            "crs_peak_ratio",
            "crs_uniform_kl",
            "crs_query_marginal",
            "crs_query_pi",
            "crs_query_p_cross_ref",
            "crs_query_p_ssm",
            "mea_query_marginal",
            "mea_support_marginal",
            "mea_query_pi",
            "mea_support_pi",
            "mea_query_attention",
            "mea_support_attention",
            "mea_query_entropy",
            "mea_support_entropy",
            "mea_query_peak_ratio",
            "mea_support_peak_ratio",
            "mea_query_uniform_kl",
            "mea_support_uniform_kl",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "nncs_beta",
            "nncs_eta",
            "mean_nncs_entropy",
            "mean_nncs_max_mass",
            "mean_nncs_min_mass",
            "mean_nncs_consistency",
            "std_nncs_consistency",
            "care_mdr_loss",
            "care_mdr_weighted_loss",
            "care_qesm_tau_e",
            "crs_eta",
            "crs_lambda_cr",
            "crs_tau_ssm",
            "crs_entropy_loss",
            "crs_aux_loss",
            "mea_eta",
            "mea_temperature",
            "mea_entropy_loss",
            "mea_aux_loss",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "care_fwec_weight" in batch_outputs[0]:
            stacked["care_fwec_weight"] = torch.stack([item["care_fwec_weight"] for item in batch_outputs]).mean(dim=0)
        if "transport_probe_cost" in batch_outputs[0]:
            stacked["transport_probe_cost"] = torch.cat(
                [item["transport_probe_cost"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_mass"] = torch.cat(
                [item["transport_probe_mass"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_entropy"] = torch.cat(
                [item["transport_probe_entropy"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_min_cost"] = torch.cat(
                [item["transport_probe_min_cost"] for item in batch_outputs],
                dim=0,
            )
        for key in (
            "query_token_mass",
            "support_token_mass",
            "query_token_weight",
            "support_token_weight",
            "hlm_query_weight",
            "hlm_support_weight",
            "hlm_token_gate",
            "hlm_budget_features",
            "hlm_mean_cost",
            "hlm_min_cost",
            "hlm_std_cost",
            "hlm_row_softmin",
            "hlm_col_softmin",
            "probe_query_reliability",
            "probe_support_reliability",
            "support_consensus",
            "query_hyperbolic_token_prior",
            "support_hyperbolic_token_prior",
            "egtw_query_mass",
            "egtw_class_attention",
            "egtw_class_entropy",
            "egtw_support_entropy",
            "egtw_support_relevance",
            "egtw_support_importance",
            "adaptive_threshold",
            "adaptive_transport_cost_threshold",
            "shot_logits",
            "shot_pool_weights",
            "dustbin_real_mass",
            "dustbin_score",
            "dustbin_real_mass_fraction",
            "dustbin_query_dustbin_fraction",
            "dustbin_support_dustbin_fraction",
            "dcr_logits",
            "dcr_unadjusted_logits",
            "dcr_removed_positive",
            "dcr_retained_positive",
            "dcr_removed_positive_share",
            "shot_mass_for_score",
            "shot_evidence_score_mass",
            "shot_evidence_score_cost",
            "shot_evidence_score_background_mass",
            "q_enhanced_logits",
            "h_anchor_logits",
            "h_anchor_shot_logits",
            "h_anchor_shot_transport_cost",
            "h_anchor_shot_transported_mass",
            "h_anchor_rho",
            "q_eam_cross_attention",
            "noise_sink_query_mass",
            "noise_sink_support_mass",
            "noise_sink_self_mass",
            "ecot_noise_sink_query_mass",
            "ecot_noise_sink_support_mass",
            "ecot_noise_sink_self_mass",
            "base_cost_matrix",
            "structure_cost",
            "structure_probe_mass",
            "shot_cover_weights",
            "shot_explanation_distance",
            "shot_coverage_ratio",
            "ncet_base_shot_logits",
            "ncet_refined_shot_logits",
            "ncet_gate",
            "ncet_confidence",
            "ncet_query_nuisance",
            "ncet_support_nuisance",
            "ncet_base_transport_cost",
            "ncet_base_transported_mass",
            "ncet_real_transport_cost",
            "ncet_real_transported_mass",
            "ncet_null_mass",
            "ncet_query_sink_mass",
            "ncet_support_sink_mass",
            "tsw_gate_q",
            "tsw_cost_modifier",
            "ecot_ccem_query_pi",
            "ecot_ccem_query_reliability",
            "ecot_ccem_support_pi",
            "token_marginal_query_weight",
            "token_marginal_support_weight",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "ecot_ccem_support_reliability",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        if "tsw_gate_s" in batch_outputs[0]:
            stacked["tsw_gate_s"] = torch.stack([item["tsw_gate_s"] for item in batch_outputs], dim=0)
        for key in (
            "token_temperature",
            "token_reliability_mix",
            "support_consensus_mix",
            "hyperbolic_token_prior_mix",
            "eam_cross_attention_temperature",
            "noise_sink_cost",
            "ecot_noise_sink_cost",
            "ecot_noise_sink_score_penalty",
            "shot_pool_temperature",
            "shot_pool_mix",
            "q_enhancement_mix",
            "structure_cost_weight",
            "mass_bonus",
            "transport_cost_threshold",
            "egtw_tau",
            "egtw_lambda",
            "egtw_attention_temperature",
            "egtw_support_similarity_weight",
            "egtw_uniform_mix",
            "hlm_config_token_tau",
            "ncet_real_penalty",
            "ncet_null_penalty",
            "ncet_mix",
            "tsw_mean_gate_q",
            "tsw_mean_gate_s",
            "ecot_ccem_query_entropy",
            "ecot_ccem_support_entropy",
            "ecot_ccem_uniform_mix",
            "ecot_ccem_tau_q",
            "ecot_ccem_tau_s",
            "evidence_score_mass_weight",
            "evidence_score_cost_weight",
            "evidence_score_background_penalty",
            "evidence_score_mix",
            "evidence_score_mode_id",
            "evidence_score_pair_mean",
            "evidence_score_pair_peak",
            "verified_uot/enabled",
            "verified_uot/beta",
            "verified_uot/tau",
            "verified_uot/ratio_threshold",
            "verified_uot/kernel_size",
            "verified_uot/gate_mean",
            "verified_uot/gate_positive_mean",
            "verified_uot/support_ratio_positive_mean",
            "verified_uot/positive_share",
            "verified_uot/positive_suppression_mean",
            "verified_uot/shot_logit_delta",
            "verified_uot/shot_logit_delta_abs",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        for key in sorted(
            key
            for key in batch_outputs[0]
            if str(key).startswith("token_marginal/")
        ):
            values = [item[key] for item in batch_outputs if key in item]
            if values and all(torch.is_tensor(value) for value in values):
                stacked[key] = torch.stack(values).mean()
        for key in (
            "egsm_kappa",
            "egsm_rho_adaptive",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "egsm_candidate_tau_q",
            "egsm_candidate_tau_b",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "egsm_psi" in batch_outputs[0]:
            stacked["egsm_psi"] = torch.cat([item["egsm_psi"] for item in batch_outputs], dim=0)
        if "transport_plan" in batch_outputs[0]:
            stacked["transport_plan"] = torch.cat([item["transport_plan"] for item in batch_outputs], dim=0)
            stacked["cost_matrix"] = torch.cat([item["cost_matrix"] for item in batch_outputs], dim=0)
            stacked["query_euclidean_tokens"] = torch.cat(
                [item["query_euclidean_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["support_euclidean_tokens"] = torch.stack(
                [item["support_euclidean_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["query_hyperbolic_tokens"] = torch.cat(
                [item["query_hyperbolic_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["support_hyperbolic_tokens"] = torch.stack(
                [item["support_hyperbolic_tokens"] for item in batch_outputs],
                dim=0,
            )
        return HROTFSLResult(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del support_targets
        batch_size, num_query, _, _, _, _ = self.validate_episode_inputs(query, support)
        query_targets_2d = self._reshape_query_targets(query_targets, batch_size=batch_size, num_query=num_query)

        needs_payload = bool(return_aux or self.training)
        batch_outputs = []
        batch_logits = []

        for batch_idx in range(batch_size):
            episode_outputs = self._forward_episode(
                query=query[batch_idx],
                support=support[batch_idx],
                query_targets=None if query_targets_2d is None else query_targets_2d[batch_idx],
                needs_payload=needs_payload,
                return_aux=return_aux,
            )
            if needs_payload:
                batch_outputs.append(episode_outputs)
                batch_logits.append(episode_outputs["logits"])
            else:
                batch_logits.append(episode_outputs)

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return HROTFSLResult({"logits": logits, "aux_loss": stacked["aux_loss"]})
