"""Standalone Ours model and contribution ablations.

The full design is intentionally treated as one coherent model:
J-ECOT-M2/SB-ECOT with UOT fixed to the paper-facing defaults.  Ours
``full`` enables Episode-Gated Shrinkage Marginals (EGSM) and turns off
MEA/CCDM/CRS for the ECOT token priors.
The ``ours_final`` registry entry uses a locked subclass that excludes EGSM
from the final paper setup and keeps threshold-mass scoring on.
Contribution ablations swap only high-level design choices while leaving the
full model path untouched.  The ``gap`` control uses global-average-pooled
tokens for the cost while keeping the same EGSM + UOT stack as ``full``.
"""

from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn.functional as F

from tim_2026.model._reference.fewshot_common import feature_map_to_tokens
from tim_2026.model._reference.hrot_fsl import HROTFSLResult, _compute_cross_shot_support_marginals, _inverse_softplus
from tim_2026.model._reference.hyperbolic.poincare_ops import safe_project_to_ball
from tim_2026.model._reference.modules.adaptive_region_uot import AdaptiveRegionUOTGuidance
from tim_2026.model._reference.modules.cost_evidence_marginals import CostEvidenceMarginals, _normalize_evidence_mode
from tim_2026.model._reference.modules.dustbin_contrastive_residual import DustbinContrastiveResidualScore
from tim_2026.model._reference.modules.evidence_adaptive_relaxation_uot import EvidenceAdaptiveRelaxationUOT
from tim_2026.model._reference.modules.hubness_calibrated_uot import HubnessCalibratedUOT
from tim_2026.model._reference.modules.multiscale_tokenizer import MultiScaleTokenizer
from tim_2026.model._reference.modules.mspta import MSPTATokenizer
from tim_2026.model._reference.modules.pulse_region_guidance import PulseRegionGuidance, normalize_saliency
from tim_2026.model._reference.modules.ours_final_failure_probe import OursFinalFailureProbe
from tim_2026.model._reference.modules.reciprocal_verified_transport import ReciprocalVerifiedTransport
from tim_2026.model._reference.modules.region_structural_uot import RegionStructuralUOTGuidance
from tim_2026.model._reference.modules.rival_conditional_ground_cost import (
    RivalConditionalGroundCost,
    normalize_rival_cost_mode,
    normalize_rival_cost_scope,
)
from tim_2026.model._reference.modules.spatial_context_enrichment import SpatialContextEnrichment, parse_context_kernel_sizes
from tim_2026.model._reference.modules.spatial_context_injection import SpatialContextInjection
from tim_2026.model._reference.modules.nuisance_referenced_ot import NuisanceReferencedOT
from tim_2026.model._reference.modules.structural_token_augmentation import StructuralTokenAugmentation
from tim_2026.model._reference.modules.utility_contrastive_marginals import UtilityContrastiveMarginals
from tim_2026.model._reference.modules.verified_region_matching import VerifiedRegionMatchingUOT
from tim_2026.model._reference.jecot_m2 import JECOTM2
from tim_2026.model._reference.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    solve_partial_transport,
)
from tim_2026.model._reference.modules.unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


OURS_ABLATIONS = frozenset({"full", "full_ot", "no_egsm", "gap"})
TOKEN_G_KINDS = frozenset({"none", "episode_mean_dist", "token_norm_pre_l2", "learned_attention"})
DMUOT_MARGINAL_KINDS = frozenset({"uniform", "discriminative"})
OURS_FINAL_MARGINAL_MODES = frozenset({"uniform", "score_aligned"})
DMUOT_SHOT_STRENGTH_KINDS = frozenset({"none", "inverse_sqrt", "inverse"})
PULSE_TRAIN_SCHEDULES = frozenset({"constant", "decay", "warmup_decay", "eval_only"})
PULSE_EVIDENCE_SCORE_MODES = frozenset({"replace", "mass_mix"})
GLOBAL_RESIDUAL_SCORE_MODES = frozenset({"residual", "global_only"})
MSPTA_GUIDANCE_SOURCES = frozenset({"image", "feature", "mixed"})
UCOT_ABLATIONS = frozenset({"full", "threshold_only", "marginal_only", "off"})


def normalize_ours_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "full",
        "ours": "full",
        "uot": "full",
        "balanced_ot": "full_ot",
        "balanced": "full_ot",
        "ot": "full_ot",
        "uniform_evidence": "no_egsm",
        "uniform": "no_egsm",
        "no_adaptive_evidence": "no_egsm",
        "no_evidence": "no_egsm",
        "proto": "gap",
        "prototype": "gap",
        "global_prototype": "gap",
        "gap_proto": "gap",
    }
    name = aliases.get(name, name)
    if name not in OURS_ABLATIONS:
        raise ValueError(f"Unsupported Ours ablation: {value}. Expected one of {sorted(OURS_ABLATIONS)}")
    return name


def apply_ours_design_defaults(kwargs: dict, ablation: str) -> dict:
    """Apply the standalone Ours design contract to HROT/M2 constructor kwargs."""
    kwargs = dict(kwargs)

    # Legacy Ours: local descriptors + one UOT budget, with the M2 mass score removed.
    # The ours_final registry entry passes ecot_m2_ablate_threshold_mass=False to
    # keep the threshold-mass score on.
    kwargs.setdefault("use_cata", False)
    kwargs.setdefault("cata_num_anchors", 8)
    kwargs.setdefault("cata_num_heads", 4)
    kwargs.setdefault("cata_attn_dropout", 0.0)
    kwargs.setdefault("ecot_rho_bank", "0.8")
    kwargs.setdefault("ecot_base_rho", 0.8)
    kwargs.setdefault("ecot_transport_mode", "unbalanced")
    # Default Ours score is -cost only. If the caller already enabled cost/mass
    # (CLI / run_all_experiments --m2_cost_per_mass), do not overwrite it here —
    # otherwise cost/mass appears to have no effect vs baseline.
    if kwargs.get("ecot_m2_cost_per_mass_score"):
        kwargs["ecot_m2_ablate_threshold_mass"] = False
        kwargs.setdefault("ecot_m2_cost_per_mass_detach_mass", True)
    elif kwargs.get("ecot_m2_ablate_threshold_mass") is False:
        kwargs["ecot_m2_cost_per_mass_score"] = False
    else:
        kwargs["ecot_m2_ablate_threshold_mass"] = True
        kwargs["ecot_m2_cost_per_mass_score"] = False
        kwargs["ecot_m2_cost_per_mass_detach_mass"] = False
    kwargs["ecot_m2_use_aqm"] = False
    kwargs["ecot_m2_tau_aqm"] = 1.0
    kwargs["ecot_m2_use_swts"] = False
    kwargs["ecot_m2_swts_temp"] = 1.0
    kwargs["ecot_enable_ccdm_marginal"] = False
    kwargs["ecot_enable_mea_marginal"] = False
    kwargs["ecot_enable_crs_marginal"] = False
    if ablation in {"full", "full_ot", "gap"}:
        kwargs.setdefault("ecot_enable_egsm", True)

    if ablation == "full_ot":
        kwargs["ecot_rho_bank"] = "1.0"
        kwargs["ecot_base_rho"] = 1.0
        kwargs["ecot_transport_mode"] = "balanced"
    elif ablation == "no_egsm":
        # Full Ours without episode-gated shrinkage marginals.
        kwargs["ecot_m2_use_aqm"] = False
        kwargs["ecot_m2_use_swts"] = False
        kwargs["ecot_enable_egsm"] = False
        kwargs["ecot_enable_ccdm_marginal"] = False
        kwargs["ecot_enable_mea_marginal"] = False
        kwargs["ecot_enable_crs_marginal"] = False
    return kwargs


def _bool_config(value: bool | str) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")
    return bool(value)


def _resolve_dm_debug(value: bool | str, use_differential_mode: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return bool(use_differential_mode)
    return _bool_config(value) and bool(use_differential_mode)


def _normalize_token_g_kind(value: str | None) -> str:
    name = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "mean_dist": "episode_mean_dist",
        "episode_mean_distance": "episode_mean_dist",
        "prenorm": "token_norm_pre_l2",
        "pre_l2": "token_norm_pre_l2",
        "token_norm": "token_norm_pre_l2",
        "learned": "learned_attention",
        "attention": "learned_attention",
        "learned_attn": "learned_attention",
    }
    name = aliases.get(name, name)
    if name not in TOKEN_G_KINDS:
        raise ValueError(f"Unsupported token_g_kind: {value}. Expected one of {sorted(TOKEN_G_KINDS)}")
    return name


def _normalize_dmuot_marginal_kind(value: str | None) -> str:
    name = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "uniform",
        "off": "uniform",
        "disc": "discriminative",
        "token_g": "discriminative",
    }
    name = aliases.get(name, name)
    if name not in DMUOT_MARGINAL_KINDS:
        raise ValueError(
            f"Unsupported marginal_kind: {value}. Expected one of {sorted(DMUOT_MARGINAL_KINDS)}"
        )
    return name


def _normalize_ours_final_marginal_mode(value: str | None) -> str:
    name = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "uniform",
        "none": "uniform",
        "original": "uniform",
        "sam": "score_aligned",
        "score": "score_aligned",
    }
    name = aliases.get(name, name)
    if name not in OURS_FINAL_MARGINAL_MODES:
        raise ValueError(
            "Unsupported ours_final_marginal_mode: "
            f"{value}. Expected one of {sorted(OURS_FINAL_MARGINAL_MODES)}"
        )
    return name


def _normalize_dmuot_shot_strength(value: str | None) -> str:
    name = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "sqrt": "inverse_sqrt",
        "inv_sqrt": "inverse_sqrt",
        "shot_sqrt": "inverse_sqrt",
        "inv": "inverse",
        "shot_inverse": "inverse",
    }
    name = aliases.get(name, name)
    if name not in DMUOT_SHOT_STRENGTH_KINDS:
        raise ValueError(
            f"Unsupported dmuot_shot_strength: {value}. "
            f"Expected one of {sorted(DMUOT_SHOT_STRENGTH_KINDS)}"
        )
    return name


def _normalize_pulse_train_schedule(value: str | None) -> str:
    name = "constant" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "eval_only",
        "visualize_only": "eval_only",
        "visualization_only": "eval_only",
        "cosine_decay": "decay",
        "linear_decay": "decay",
        "warmup": "warmup_decay",
        "warm_up_decay": "warmup_decay",
    }
    name = aliases.get(name, name)
    if name not in PULSE_TRAIN_SCHEDULES:
        raise ValueError(
            f"Unsupported pulse_region_train_schedule: {value}. "
            f"Expected one of {sorted(PULSE_TRAIN_SCHEDULES)}"
        )
    return name


def _normalize_pulse_evidence_score_mode(value: str | None) -> str:
    name = "replace" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "score_only": "replace",
        "masked": "replace",
        "masked_score": "replace",
        "mass": "mass_mix",
        "mixed_mass": "mass_mix",
        "mass_reward": "mass_mix",
    }
    name = aliases.get(name, name)
    if name not in PULSE_EVIDENCE_SCORE_MODES:
        raise ValueError(
            f"Unsupported pulse_evidence_score_mode: {value}. "
            f"Expected one of {sorted(PULSE_EVIDENCE_SCORE_MODES)}"
        )
    return name


def _normalize_global_residual_score_mode(value: str | None) -> str:
    name = "residual" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "add": "residual",
        "additive": "residual",
        "fuse": "residual",
        "fusion": "residual",
        "global": "global_only",
        "only_global": "global_only",
        "prototype": "global_only",
        "prototype_only": "global_only",
    }
    name = aliases.get(name, name)
    if name not in GLOBAL_RESIDUAL_SCORE_MODES:
        raise ValueError(
            f"Unsupported global_residual_mode: {value}. "
            f"Expected one of {sorted(GLOBAL_RESIDUAL_SCORE_MODES)}"
        )
    return name


def _normalize_ucot_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "true": "full",
        "1": "full",
        "threshold": "threshold_only",
        "t_only": "threshold_only",
        "marginal": "marginal_only",
        "marginals": "marginal_only",
        "quota": "marginal_only",
        "quotas": "marginal_only",
    }
    name = aliases.get(name, name)
    if name not in UCOT_ABLATIONS:
        raise ValueError(
            f"Unsupported ucot_ablation: {value}. Expected one of {sorted(UCOT_ABLATIONS)}"
        )
    return name


def _normalize_mspta_guidance_source(value: str | None) -> str:
    name = "mixed" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "both": "mixed",
        "input": "image",
        "img": "image",
        "feat": "feature",
    }
    name = aliases.get(name, name)
    if name not in MSPTA_GUIDANCE_SOURCES:
        raise ValueError(
            f"Unsupported mspta_guidance_source: {value}. "
            f"Expected one of {sorted(MSPTA_GUIDANCE_SOURCES)}"
        )
    return name


def _infer_token_hw(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    root = int(math.sqrt(num_tokens))
    best = (1, num_tokens)
    best_gap = num_tokens - 1
    for height in range(1, root + 1):
        if num_tokens % height != 0:
            continue
        width = num_tokens // height
        gap = abs(width - height)
        if gap < best_gap:
            best = (height, width)
            best_gap = gap
    return best


class OursM2(JECOTM2):
    """Paper-facing Ours entrypoint plus coarse contribution ablations."""

    def __init__(
        self,
        *args,
        ours_ablation: str = "full",
        use_differential_mode: bool = False,
        dm_alpha: float = 0.0,
        dm_debug: bool | str = "auto",
        dm_debug_dir: str = "results/dmt_debug",
        dm_debug_max_episodes: int = 5,
        token_g_kind: str = "none",
        lambda_cost: float = 0.0,
        marginal_kind: str = "uniform",
        tau_marg: float = 1.0,
        dmuot_shot_strength: str = "none",
        ours_final_marginal_mode: str = "uniform",
        score_marginal_tau: float = 0.25,
        score_marginal_mix: float = 0.65,
        score_marginal_adaptive_mix: bool = True,
        score_marginal_confidence_power: float = 1.0,
        ours_final_score_mode: str = "threshold_mass",
        elastic_ot_sigma: float = 1.0,
        enable_elastic_ot_probe: bool = False,
        dustbin_score_init: float = 0.0,
        enable_dustbin_contrastive_score: bool = False,
        dcr_beta: float = 0.50,
        dcr_tau: float = 0.25,
        dcr_alpha: float = 0.0,
        dcr_margin: float = 0.0,
        dcr_min_gate: float = 0.05,
        dcr_detach_gate: bool = True,
        enable_evidence_adaptive_relaxation_uot: bool = False,
        ear_uot_tau_min: float = 0.05,
        ear_uot_tau_max: float = 1.00,
        ear_uot_temperature: float = 0.25,
        ear_uot_margin: float = 0.0,
        ear_uot_spatial_mix: float = 0.50,
        ear_uot_kernel_size: int = 3,
        ear_uot_detach_reliability: bool = True,
        enable_ours_final_failure_probe: bool = False,
        ours_probe_common_margin: float = 0.10,
        enable_rival_conditional_cost: bool = False,
        rc_cost_weight: float = 0.50,
        rc_cost_temperature: float = 0.25,
        rc_cost_mode: str = "class_nll",
        rc_cost_scope: str = "always",
        enable_evidence_marginals: bool = False,
        evidence_tau: float = 0.1,
        evidence_tau_marginal: float = 1.0,
        evidence_mode: str = "both",
        evidence_rival_margin: float = 0.5,
        evidence_detach: bool = True,
        enable_sci: bool = False,
        sci_num_layers: int = 2,
        sci_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        enable_multiscale_ot = _bool_config(kwargs.pop("enable_multiscale_ot", False))
        multiscale_pool_sizes = kwargs.pop("multiscale_pool_sizes", "original,2x2,1x1")
        multiscale_per_scale_T = _bool_config(kwargs.pop("multiscale_per_scale_T", False))
        enable_mspta = _bool_config(kwargs.pop("enable_mspta", False))
        mspta_scales = kwargs.pop("mspta_scales", "1,2,3")
        mspta_mass_mode = str(kwargs.pop("mspta_mass_mode", "compact_area"))
        mspta_normalize = _bool_config(kwargs.pop("mspta_normalize", True))
        mspta_proj_dim = int(kwargs.pop("mspta_proj_dim", 0))
        mspta_learnable_weights = _bool_config(kwargs.pop("mspta_learnable_weights", False))
        mspta_compact_floor = float(kwargs.pop("mspta_compact_floor", 0.05))
        mspta_vertical_suppression = float(kwargs.pop("mspta_vertical_suppression", 1.0))
        mspta_saliency_cost_discount = float(kwargs.pop("mspta_saliency_cost_discount", 0.10))
        mspta_guidance_source = _normalize_mspta_guidance_source(
            kwargs.pop("mspta_guidance_source", "mixed")
        )
        enable_context_enrichment = _bool_config(kwargs.pop("enable_context_enrichment", False))
        context_kernel_sizes_raw = kwargs.pop("context_kernel_sizes", "3,5")
        context_fusion = str(kwargs.pop("context_fusion", "weighted_sum"))
        context_gate_max = float(kwargs.pop("context_gate_max", 1.0))
        context_change_max = float(kwargs.pop("context_change_max", 0.0))
        context_debug = _bool_config(kwargs.pop("context_debug", False))
        context_debug_dir = str(kwargs.pop("context_debug_dir", "results/context_debug"))
        context_debug_max_episodes = int(kwargs.pop("context_debug_max_episodes", 3))
        enable_structural_augmentation = _bool_config(kwargs.pop("enable_structural_augmentation", False))
        struct_dim = int(kwargs.pop("struct_dim", 16))
        enable_region_structural_uot = _bool_config(kwargs.pop("enable_region_structural_uot", False))
        region_uot_grid_size = kwargs.pop("region_uot_grid_size", "3x3")
        region_uot_strength = float(kwargs.pop("region_uot_strength", 0.08))
        region_uot_fgw_alpha = float(kwargs.pop("region_uot_fgw_alpha", 0.35))
        region_uot_sinkhorn_epsilon = float(kwargs.pop("region_uot_sinkhorn_epsilon", 0.08))
        region_uot_fgw_iters = int(kwargs.pop("region_uot_fgw_iters", 4))
        region_uot_sinkhorn_iters = int(kwargs.pop("region_uot_sinkhorn_iters", 40))
        region_uot_topk = int(kwargs.pop("region_uot_topk", 3))
        region_uot_fine_gate_quantile = float(kwargs.pop("region_uot_fine_gate_quantile", 0.35))
        region_uot_min_confidence = float(kwargs.pop("region_uot_min_confidence", 0.10))
        region_uot_importance_temperature = float(kwargs.pop("region_uot_importance_temperature", 0.50))
        enable_adaptive_region_uot = _bool_config(kwargs.pop("enable_adaptive_region_uot", False))
        adaptive_region_num_slots = int(kwargs.pop("adaptive_region_num_slots", 4))
        adaptive_region_context_kernels = kwargs.pop("adaptive_region_context_kernels", "1,3,5")
        adaptive_region_cost_discount = float(kwargs.pop("adaptive_region_cost_discount", 0.12))
        adaptive_region_mass_mix = float(kwargs.pop("adaptive_region_mass_mix", 0.60))
        adaptive_region_sinkhorn_epsilon = float(kwargs.pop("adaptive_region_sinkhorn_epsilon", 0.08))
        adaptive_region_sinkhorn_iters = int(kwargs.pop("adaptive_region_sinkhorn_iters", 30))
        adaptive_region_fine_gate_quantile = float(kwargs.pop("adaptive_region_fine_gate_quantile", 0.40))
        adaptive_region_temperature_min = float(kwargs.pop("adaptive_region_temperature_min", 0.35))
        adaptive_region_temperature_max = float(kwargs.pop("adaptive_region_temperature_max", 1.25))
        adaptive_region_init_gate = float(kwargs.pop("adaptive_region_init_gate", 0.05))
        enable_pulse_region_uot = _bool_config(kwargs.pop("enable_pulse_region_uot", False))
        pulse_region_kernel_size = int(kwargs.pop("pulse_region_kernel_size", 5))
        pulse_region_cost_weight = float(kwargs.pop("pulse_region_cost_weight", 0.35))
        pulse_saliency_mass_mix = float(kwargs.pop("pulse_saliency_mass_mix", 0.50))
        pulse_saliency_cost_discount = float(kwargs.pop("pulse_saliency_cost_discount", 0.10))
        pulse_saliency_image_weight = float(kwargs.pop("pulse_saliency_image_weight", 0.45))
        pulse_saliency_feature_weight = float(kwargs.pop("pulse_saliency_feature_weight", 0.35))
        pulse_saliency_contrast_weight = float(kwargs.pop("pulse_saliency_contrast_weight", 0.20))
        pulse_region_train_strength = float(kwargs.pop("pulse_region_train_strength", 1.0))
        pulse_region_eval_strength = float(kwargs.pop("pulse_region_eval_strength", 1.0))
        pulse_region_multishot_train_strength = kwargs.pop("pulse_region_multishot_train_strength", None)
        pulse_region_multishot_eval_strength = kwargs.pop("pulse_region_multishot_eval_strength", None)
        pulse_region_train_schedule = _normalize_pulse_train_schedule(
            kwargs.pop("pulse_region_train_schedule", "constant")
        )
        pulse_support_consensus_weight = float(kwargs.pop("pulse_support_consensus_weight", 0.0))
        pulse_support_consensus_beta = float(kwargs.pop("pulse_support_consensus_beta", 5.0))
        pulse_support_consensus_eta = float(kwargs.pop("pulse_support_consensus_eta", 0.05))
        pulse_evidence_score = _bool_config(kwargs.pop("pulse_evidence_score", False))
        pulse_evidence_score_mode = _normalize_pulse_evidence_score_mode(
            kwargs.pop("pulse_evidence_score_mode", "replace")
        )
        pulse_evidence_score_mix = float(kwargs.pop("pulse_evidence_score_mix", 1.0))
        pulse_evidence_mass_weight = float(kwargs.pop("pulse_evidence_mass_weight", 1.0))
        pulse_evidence_cost_weight = float(kwargs.pop("pulse_evidence_cost_weight", 1.0))
        pulse_background_penalty = float(kwargs.pop("pulse_background_penalty", 0.25))
        enable_discriminative_uot = _bool_config(kwargs.pop("enable_discriminative_uot", False))
        discriminative_uot_tau = float(kwargs.pop("discriminative_uot_tau", 0.05))
        discriminative_uot_margin = float(kwargs.pop("discriminative_uot_margin", 0.02))
        discriminative_uot_mix = float(kwargs.pop("discriminative_uot_mix", 1.0))
        discriminative_uot_background_penalty = float(kwargs.pop("discriminative_uot_background_penalty", 0.25))
        discriminative_uot_mass_weight = float(kwargs.pop("discriminative_uot_mass_weight", 1.0))
        discriminative_uot_cost_weight = float(kwargs.pop("discriminative_uot_cost_weight", 1.0))
        pulse_discriminative_evidence = _bool_config(kwargs.pop("pulse_discriminative_evidence", False))
        pulse_discriminative_tau = float(kwargs.pop("pulse_discriminative_tau", 0.05))
        pulse_discriminative_margin = float(kwargs.pop("pulse_discriminative_margin", 0.02))
        pulse_discriminative_mix = float(kwargs.pop("pulse_discriminative_mix", 1.0))
        enable_episode_contrastive_uot = _bool_config(kwargs.pop("enable_episode_contrastive_uot", False))
        ect_uot_tau = float(kwargs.pop("ect_uot_tau", 0.25))
        ect_uot_margin = float(kwargs.pop("ect_uot_margin", 0.0))
        ect_uot_cost_weight = float(kwargs.pop("ect_uot_cost_weight", 0.35))
        ect_uot_query_mass_mix = float(kwargs.pop("ect_uot_query_mass_mix", 0.50))
        ect_uot_query_tau = float(kwargs.pop("ect_uot_query_tau", 0.50))
        ect_uot_detach = _bool_config(kwargs.pop("ect_uot_detach", True))
        enable_verified_uot_score = _bool_config(kwargs.pop("enable_verified_uot_score", False))
        verified_uot_beta = float(kwargs.pop("verified_uot_beta", 0.75))
        verified_uot_tau = float(kwargs.pop("verified_uot_tau", 0.10))
        verified_uot_ratio_threshold = float(kwargs.pop("verified_uot_ratio_threshold", 0.35))
        verified_uot_kernel_size = int(kwargs.pop("verified_uot_kernel_size", 3))
        enable_reciprocal_verified_uot = _bool_config(kwargs.pop("enable_reciprocal_verified_uot", False))
        rvuot_beta = float(kwargs.pop("rvuot_beta", 0.85))
        rvuot_tau = float(kwargs.pop("rvuot_tau", 0.10))
        rvuot_ratio_threshold = float(kwargs.pop("rvuot_ratio_threshold", 0.25))
        rvuot_kernel_size = int(kwargs.pop("rvuot_kernel_size", 3))
        rvuot_cost_quantile = float(kwargs.pop("rvuot_cost_quantile", 0.35))
        rvuot_min_gate = float(kwargs.pop("rvuot_min_gate", 0.05))
        rvuot_enable_rival_gate = _bool_config(kwargs.pop("rvuot_enable_rival_gate", True))
        rvuot_rival_score_mix = float(kwargs.pop("rvuot_rival_score_mix", 0.0))
        rvuot_rival_tau = float(kwargs.pop("rvuot_rival_tau", 0.10))
        rvuot_rival_margin = float(kwargs.pop("rvuot_rival_margin", 0.0))
        rvuot_detach_gate = _bool_config(kwargs.pop("rvuot_detach_gate", True))
        enable_verified_region_matching_uot = _bool_config(
            kwargs.pop("enable_verified_region_matching_uot", False)
        )
        vrm_lambda = float(kwargs.pop("vrm_lambda", 0.20))
        vrm_concentration_tau = float(kwargs.pop("vrm_concentration_tau", 0.25))
        vrm_region_tau = float(kwargs.pop("vrm_region_tau", 0.50))
        vrm_region_kernel_size = int(kwargs.pop("vrm_region_kernel_size", 3))
        vrm_use_concentration = _bool_config(kwargs.pop("vrm_use_concentration", True))
        vrm_use_region_patch = _bool_config(kwargs.pop("vrm_use_region_patch", True))
        vrm_use_rival = _bool_config(kwargs.pop("vrm_use_rival", False))
        vrm_rival_tau = float(kwargs.pop("vrm_rival_tau", 0.25))
        vrm_rival_threshold = float(kwargs.pop("vrm_rival_threshold", 0.0))
        vrm_rival_evidence_tau = float(kwargs.pop("vrm_rival_evidence_tau", 0.10))
        vrm_rival_margin = float(kwargs.pop("vrm_rival_margin", 0.50))
        vrm_detach_gate = _bool_config(kwargs.pop("vrm_detach_gate", True))
        enable_global_residual_score = _bool_config(kwargs.pop("enable_global_residual_score", False))
        global_residual_mode = _normalize_global_residual_score_mode(
            kwargs.pop("global_residual_mode", "residual")
        )
        global_residual_weight = float(kwargs.pop("global_residual_weight", 0.10))
        enable_nr_ot = _bool_config(kwargs.pop("enable_nr_ot", False))
        nr_ot_mode = str(kwargs.pop("nr_ot_mode", "standalone")).strip().lower().replace("-", "_")
        if nr_ot_mode not in {"standalone", "residual", "gated_residual"}:
            raise ValueError(
                f"Unsupported nr_ot_mode: {nr_ot_mode}. Expected 'standalone', "
                "'residual', or 'gated_residual'."
            )
        nr_ot_weight = float(kwargs.pop("nr_ot_weight", 1.0))
        nr_ot_novelty_temp = float(kwargs.pop("nr_ot_novelty_temp", 0.5))
        nr_ot_uniform_reference = _bool_config(kwargs.pop("nr_ot_uniform_reference", True))
        nr_ot_eval_only = _bool_config(kwargs.pop("nr_ot_eval_only", False))
        nr_ot_gate_temperature = float(kwargs.pop("nr_ot_gate_temperature", 1.0))
        nr_ot_detach = _bool_config(kwargs.pop("nr_ot_detach", False))
        enable_residual_aligned_uot = _bool_config(kwargs.pop("enable_residual_aligned_uot", False))
        rauot_tau = float(kwargs.pop("rauot_tau", 0.50))
        rauot_margin = float(kwargs.pop("rauot_margin", 0.0))
        rauot_cost_weight = float(kwargs.pop("rauot_cost_weight", 0.15))
        rauot_mass_mix = float(kwargs.pop("rauot_mass_mix", 0.25))
        rauot_detach = _bool_config(kwargs.pop("rauot_detach", True))
        enable_rival_aware_evidence_uot = _bool_config(
            kwargs.pop("enable_rival_aware_evidence_uot", False)
        )
        rae_uot_tau = float(kwargs.pop("rae_uot_tau", 0.25))
        rae_uot_marginal_mix = float(kwargs.pop("rae_uot_marginal_mix", 0.65))
        enable_hubness_calibrated_uot = _bool_config(
            kwargs.pop("enable_hubness_calibrated_uot", False)
        )
        hcuot_topk = int(kwargs.pop("hcuot_topk", 3))
        hcuot_temperature = float(kwargs.pop("hcuot_temperature", 0.25))
        hcuot_cost_weight = float(kwargs.pop("hcuot_cost_weight", 0.35))
        hcuot_marginal_mix = float(kwargs.pop("hcuot_marginal_mix", 0.50))
        enable_label_ot = _bool_config(
            kwargs.pop("enable_label_ot", kwargs.pop("enable_transductive_label_ot", False))
        )
        label_ot_epsilon = float(kwargs.pop("label_ot_epsilon", 0.75))
        label_ot_iterations = int(kwargs.pop("label_ot_iterations", 30))
        label_ot_mix = float(kwargs.pop("label_ot_mix", 1.0))
        label_ot_min_queries_per_class = int(kwargs.pop("label_ot_min_queries_per_class", 2))
        label_ot_min_column_imbalance = float(kwargs.pop("label_ot_min_column_imbalance", 0.0))
        label_ot_max_bias = float(kwargs.pop("label_ot_max_bias", 2.0))
        enable_pot_guide = bool(kwargs.pop("enable_pot_guide", False))
        pot_guide_s = float(kwargs.pop("pot_guide_s", 0.5))
        pot_guide_adaptive_s = bool(kwargs.pop("pot_guide_adaptive_s", False))
        pot_guide_s_min = float(kwargs.pop("pot_guide_s_min", 0.2))
        pot_guide_s_max = float(kwargs.pop("pot_guide_s_max", 0.8))
        pot_guide_epsilon = float(kwargs.pop("pot_guide_epsilon", 0.05))
        pot_guide_max_iter = int(kwargs.pop("pot_guide_max_iter", 50))
        enable_token_attention_marginal = _bool_config(
            kwargs.get("enable_token_attention_marginal", False)
        )
        if not 0.0 <= float(dm_alpha) <= 1.0:
            raise ValueError("dm_alpha must be in [0, 1]")
        if int(dm_debug_max_episodes) < 0:
            raise ValueError("dm_debug_max_episodes must be non-negative")
        self.token_g_kind = _normalize_token_g_kind(token_g_kind)
        self.lambda_cost = float(lambda_cost)
        self.marginal_kind = _normalize_dmuot_marginal_kind(marginal_kind)
        self.tau_marg = float(tau_marg)
        self.dmuot_shot_strength = _normalize_dmuot_shot_strength(dmuot_shot_strength)
        self.ours_final_marginal_mode = _normalize_ours_final_marginal_mode(
            ours_final_marginal_mode
        )
        self.score_marginal_tau = float(score_marginal_tau)
        self.score_marginal_mix = float(score_marginal_mix)
        self.score_marginal_adaptive_mix = _bool_config(score_marginal_adaptive_mix)
        self.score_marginal_confidence_power = float(score_marginal_confidence_power)
        self.ours_final_score_mode = (
            str(ours_final_score_mode).strip().lower().replace("-", "_")
        )
        if self.ours_final_score_mode not in {
            "threshold_mass",
            "uot_energy",
            "elastic_ot",
            "dustbin_ot",
        }:
            raise ValueError(
                "ours_final_score_mode must be 'threshold_mass', 'uot_energy', "
                "'elastic_ot', or 'dustbin_ot'"
            )
        self.elastic_ot_sigma = float(elastic_ot_sigma)
        self.enable_elastic_ot_probe = _bool_config(enable_elastic_ot_probe)
        self.dustbin_score_init = float(dustbin_score_init)
        self.enable_dustbin_contrastive_score = _bool_config(
            enable_dustbin_contrastive_score
        )
        self.dcr_beta = float(dcr_beta)
        self.dcr_tau = float(dcr_tau)
        self.dcr_alpha = float(dcr_alpha)
        self.dcr_margin = float(dcr_margin)
        self.dcr_min_gate = float(dcr_min_gate)
        self.dcr_detach_gate = _bool_config(dcr_detach_gate)
        self.enable_evidence_adaptive_relaxation_uot = _bool_config(
            enable_evidence_adaptive_relaxation_uot
        )
        self.ear_uot_tau_min = float(ear_uot_tau_min)
        self.ear_uot_tau_max = float(ear_uot_tau_max)
        self.ear_uot_temperature = float(ear_uot_temperature)
        self.ear_uot_margin = float(ear_uot_margin)
        self.ear_uot_spatial_mix = float(ear_uot_spatial_mix)
        self.ear_uot_kernel_size = int(ear_uot_kernel_size)
        self.ear_uot_detach_reliability = _bool_config(
            ear_uot_detach_reliability
        )
        if (
            self.enable_dustbin_contrastive_score
            and self.ours_final_score_mode != "threshold_mass"
        ):
            raise ValueError(
                "enable_dustbin_contrastive_score is a residual correction for "
                "ours_final_score_mode='threshold_mass'"
            )
        if (
            self.ours_final_score_mode in {"elastic_ot", "dustbin_ot"}
            and self.ours_final_marginal_mode != "uniform"
        ):
            raise ValueError(
                f"ours_final_score_mode='{self.ours_final_score_mode}' cannot be combined with "
                "ours_final_marginal_mode; uniform values are capacities, not target marginals"
            )
        self.enable_ours_final_failure_probe = _bool_config(
            enable_ours_final_failure_probe
        )
        self.ours_probe_common_margin = float(ours_probe_common_margin)
        self.enable_rival_conditional_cost = _bool_config(
            enable_rival_conditional_cost
        )
        self.rc_cost_weight = float(rc_cost_weight)
        self.rc_cost_temperature = float(rc_cost_temperature)
        self.rc_cost_mode = normalize_rival_cost_mode(rc_cost_mode)
        self.rc_cost_scope = normalize_rival_cost_scope(rc_cost_scope)
        self._rc_cost_noise_test_context = False
        if self.lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        if self.lambda_cost > 0.0 and self.token_g_kind == "none":
            raise ValueError("lambda_cost > 0 requires token_g_kind != 'none'")
        if self.marginal_kind == "discriminative" and self.token_g_kind == "none":
            raise ValueError("marginal_kind='discriminative' requires token_g_kind != 'none'")
        if not (self.tau_marg > 0.0 or math.isinf(self.tau_marg)):
            raise ValueError("tau_marg must be positive or inf")
        self.ours_ablation = normalize_ours_ablation(ours_ablation)
        self.use_differential_mode = _bool_config(use_differential_mode)
        self.dm_alpha = float(dm_alpha)
        self.dm_debug = _resolve_dm_debug(dm_debug, self.use_differential_mode)
        self.dm_debug_dir = str(dm_debug_dir)
        self.dm_debug_max_episodes = int(dm_debug_max_episodes)
        self._dm_debug_count = 0
        self._last_dm_diagnostics: dict[str, torch.Tensor] | None = None
        self._token_g_prenorm_cache: list[torch.Tensor] | None = None
        if enable_rival_aware_evidence_uot:
            # Minimal RAE-UOT replaces EGSM only at the marginal level. The
            # original Ours-Final transport cost and T*M-C score stay intact.
            kwargs["ecot_enable_egsm"] = False
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        if self.ours_final_marginal_mode == "score_aligned":
            kwargs["ecot_enable_egsm"] = False
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        if enable_hubness_calibrated_uot:
            # HC-UOT is an analytic episode-local calibration of the original
            # threshold-mass UOT path, with no learned attention branch.
            kwargs["ecot_enable_egsm"] = False
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        if self.enable_evidence_adaptive_relaxation_uot:
            kwargs["ecot_enable_egsm"] = False
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        if enable_token_attention_marginal:
            kwargs["ecot_enable_egsm"] = False
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        kwargs = apply_ours_design_defaults(kwargs, self.ours_ablation)
        kwargs["ecot_m2_score_mode"] = self.ours_final_score_mode
        kwargs["ecot_m2_elastic_sigma"] = self.elastic_ot_sigma
        kwargs["ecot_m2_elastic_probe"] = self.enable_elastic_ot_probe
        kwargs["ecot_m2_dustbin_score_init"] = self.dustbin_score_init
        if self.ours_final_score_mode in {"uot_energy", "elastic_ot", "dustbin_ot"}:
            kwargs["ecot_m2_ablate_threshold_mass"] = False
            kwargs["ecot_m2_cost_per_mass_score"] = False
        if enable_mspta:
            if self.ours_ablation == "gap":
                raise ValueError("enable_mspta is not supported with ours_ablation='gap'")
            if mspta_proj_dim not in {0, int(kwargs.get("token_dim", mspta_proj_dim))}:
                raise ValueError("mspta_proj_dim must be 0 or match hrot_token_dim; use hrot_token_dim for Ours-Final")
            if enable_multiscale_ot:
                raise ValueError("enable_mspta is not supported with enable_multiscale_ot")
            if self.token_g_kind != "none":
                raise ValueError("enable_mspta currently requires token_g_kind='none'")
            if _bool_config(kwargs.get("use_cata", False)):
                raise ValueError("enable_mspta is not supported with CATA token aggregation")
            if _bool_config(kwargs.get("hrot_amp_marginals", False)):
                raise ValueError("enable_mspta is not supported with hrot_amp_marginals")
            if _bool_config(kwargs.get("ecot_episode_feature_normalize", False)):
                raise ValueError("enable_mspta is not supported with ecot_episode_feature_normalize")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_mspta is not supported with pre_transport_shot_pool")
            if enable_structural_augmentation:
                raise ValueError("enable_mspta is not supported with enable_structural_augmentation")
            if enable_region_structural_uot:
                raise ValueError("enable_mspta is not supported with enable_region_structural_uot")
            if enable_adaptive_region_uot:
                raise ValueError("enable_mspta is not supported with enable_adaptive_region_uot")
            if enable_pulse_region_uot:
                raise ValueError("enable_mspta is not supported with enable_pulse_region_uot")
            if enable_pot_guide:
                raise ValueError("enable_mspta is not supported with enable_pot_guide")
            if not 0.0 <= mspta_saliency_cost_discount < 1.0:
                raise ValueError("mspta_saliency_cost_discount must be in [0, 1)")
        if enable_multiscale_ot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_multiscale_ot is not supported with ours_ablation='gap'")
            if self.token_g_kind != "none":
                raise ValueError("enable_multiscale_ot currently requires token_g_kind='none'")
            if _bool_config(kwargs.get("hrot_amp_marginals", False)):
                raise ValueError("enable_multiscale_ot is not supported with hrot_amp_marginals")
            if _bool_config(kwargs.get("ecot_episode_feature_normalize", False)):
                raise ValueError("enable_multiscale_ot is not supported with ecot_episode_feature_normalize")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_multiscale_ot is not supported with pre_transport_shot_pool")
        if enable_region_structural_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_region_structural_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_region_structural_uot is not supported with enable_multiscale_ot")
            if enable_pulse_region_uot:
                raise ValueError("enable_region_structural_uot is not supported with enable_pulse_region_uot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_region_structural_uot is not supported with pre_transport_shot_pool")
        if enable_adaptive_region_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_adaptive_region_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_multiscale_ot")
            if enable_region_structural_uot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_region_structural_uot")
            if enable_pulse_region_uot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_pulse_region_uot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_adaptive_region_uot is not supported with pre_transport_shot_pool")
        if enable_pulse_region_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_pulse_region_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_pulse_region_uot is not supported with enable_multiscale_ot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_pulse_region_uot is not supported with pre_transport_shot_pool")
        if enable_reciprocal_verified_uot:
            if enable_verified_uot_score:
                raise ValueError("enable_reciprocal_verified_uot cannot be combined with enable_verified_uot_score")
            if self.ours_ablation == "gap":
                raise ValueError("enable_reciprocal_verified_uot is not supported with ours_ablation='gap'")
        if enable_verified_region_matching_uot:
            if self.ours_ablation != "full":
                raise ValueError(
                    "enable_verified_region_matching_uot requires ours_ablation='full'"
                )
            if self.ours_final_score_mode != "threshold_mass":
                raise ValueError(
                    "enable_verified_region_matching_uot requires "
                    "ours_final_score_mode='threshold_mass'"
                )
            conflicts = {
                "ours_final_marginal_mode": self.ours_final_marginal_mode != "uniform",
                "marginal_kind": self.marginal_kind != "uniform",
                "token_g_kind": self.token_g_kind != "none",
                "enable_evidence_marginals": _bool_config(
                    enable_evidence_marginals
                ),
                "enable_evidence_adaptive_relaxation_uot": (
                    self.enable_evidence_adaptive_relaxation_uot
                ),
                "enable_rival_conditional_cost": self.enable_rival_conditional_cost,
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_context_enrichment": enable_context_enrichment,
                "enable_structural_augmentation": enable_structural_augmentation,
                "enable_label_ot": enable_label_ot,
                "enable_pot_guide": enable_pot_guide,
                "enable_token_attention_marginal": enable_token_attention_marginal,
                "use_differential_mode": self.use_differential_mode,
                "pre_transport_shot_pool": _bool_config(
                    kwargs.get("pre_transport_shot_pool", False)
                ),
            }
            active_conflicts = [
                name for name, active in conflicts.items() if active
            ]
            if active_conflicts:
                raise ValueError(
                    "enable_verified_region_matching_uot isolates the "
                    "pre-transport verification cost and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if enable_token_attention_marginal:
            if self.ours_ablation != "full":
                raise ValueError(
                    "enable_token_attention_marginal requires ours_ablation='full'"
                )
            if self.ours_final_score_mode != "threshold_mass":
                raise ValueError(
                    "enable_token_attention_marginal requires "
                    "ours_final_score_mode='threshold_mass'"
                )
            conflicts = {
                "ours_final_marginal_mode": self.ours_final_marginal_mode
                != "uniform",
                "marginal_kind": self.marginal_kind != "uniform",
                "token_g_kind": self.token_g_kind != "none",
                "enable_evidence_marginals": _bool_config(
                    enable_evidence_marginals
                ),
                "enable_evidence_adaptive_relaxation_uot": (
                    self.enable_evidence_adaptive_relaxation_uot
                ),
                "enable_rival_conditional_cost": self.enable_rival_conditional_cost,
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_context_enrichment": enable_context_enrichment,
                "enable_structural_augmentation": enable_structural_augmentation,
                "enable_label_ot": enable_label_ot,
                "enable_pot_guide": enable_pot_guide,
                "enable_verified_region_matching_uot": enable_verified_region_matching_uot,
                "use_differential_mode": self.use_differential_mode,
                "pre_transport_shot_pool": _bool_config(
                    kwargs.get("pre_transport_shot_pool", False)
                ),
            }
            active_conflicts = [
                name for name, active in conflicts.items() if active
            ]
            if active_conflicts:
                raise ValueError(
                    "enable_token_attention_marginal isolates learned token "
                    "marginals and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if self.enable_evidence_adaptive_relaxation_uot:
            if self.ours_ablation != "full":
                raise ValueError(
                    "enable_evidence_adaptive_relaxation_uot requires ours_ablation='full'"
                )
            if self.ours_final_score_mode not in {"threshold_mass", "uot_energy"}:
                raise ValueError(
                    "EAR-UOT requires threshold_mass or uot_energy scoring"
                )
            conflicts = {
                "ours_final_marginal_mode": self.ours_final_marginal_mode != "uniform",
                "enable_dustbin_contrastive_score": self.enable_dustbin_contrastive_score,
                "enable_rival_conditional_cost": self.enable_rival_conditional_cost,
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_evidence_marginals": _bool_config(enable_evidence_marginals),
                "enable_pot_guide": enable_pot_guide,
                "token_g_kind": self.token_g_kind != "none",
            }
            active_conflicts = [name for name, active in conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "EAR-UOT isolates token-wise marginal relaxation and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if self.ours_final_marginal_mode == "score_aligned":
            if self.ours_ablation != "full":
                raise ValueError(
                    "ours_final_marginal_mode='score_aligned' requires ours_ablation='full'"
                )
            conflicts = {
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_evidence_marginals": _bool_config(enable_evidence_marginals),
                "enable_pot_guide": enable_pot_guide,
                "token_g_kind": self.token_g_kind != "none",
            }
            active_conflicts = [name for name, active in conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "ours_final_marginal_mode='score_aligned' replaces only the original "
                    "uniform marginals and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if enable_rival_aware_evidence_uot:
            if self.ours_ablation == "gap":
                raise ValueError(
                    "enable_rival_aware_evidence_uot is not supported with ours_ablation='gap'"
                )
            conflicts = {
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_evidence_marginals": _bool_config(enable_evidence_marginals),
                "token_g_kind": self.token_g_kind != "none",
            }
            active_conflicts = [name for name, active in conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "enable_rival_aware_evidence_uot defines the complete local transport path and "
                    "cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if enable_hubness_calibrated_uot:
            if self.ours_ablation == "gap":
                raise ValueError(
                    "enable_hubness_calibrated_uot is not supported with ours_ablation='gap'"
                )
            conflicts = {
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_evidence_marginals": _bool_config(enable_evidence_marginals),
                "token_g_kind": self.token_g_kind != "none",
            }
            active_conflicts = [name for name, active in conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "enable_hubness_calibrated_uot defines the complete local transport path and "
                    "cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if self.enable_rival_conditional_cost:
            if self.ours_ablation != "full":
                raise ValueError(
                    "enable_rival_conditional_cost requires ours_ablation='full'"
                )
            conflicts = {
                "ours_final_marginal_mode": self.ours_final_marginal_mode != "uniform",
                "enable_rival_aware_evidence_uot": enable_rival_aware_evidence_uot,
                "enable_hubness_calibrated_uot": enable_hubness_calibrated_uot,
                "enable_episode_contrastive_uot": enable_episode_contrastive_uot,
                "enable_residual_aligned_uot": enable_residual_aligned_uot,
                "enable_verified_uot_score": enable_verified_uot_score,
                "enable_reciprocal_verified_uot": enable_reciprocal_verified_uot,
                "enable_discriminative_uot": enable_discriminative_uot,
                "enable_region_structural_uot": enable_region_structural_uot,
                "enable_adaptive_region_uot": enable_adaptive_region_uot,
                "enable_pulse_region_uot": enable_pulse_region_uot,
                "enable_multiscale_ot": enable_multiscale_ot,
                "enable_mspta": enable_mspta,
                "enable_evidence_marginals": _bool_config(enable_evidence_marginals),
                "enable_global_residual_score": enable_global_residual_score,
                "enable_context_enrichment": enable_context_enrichment,
                "enable_structural_augmentation": enable_structural_augmentation,
                "enable_label_ot": enable_label_ot,
                "enable_pot_guide": enable_pot_guide,
                "pre_transport_shot_pool": _bool_config(
                    kwargs.get("pre_transport_shot_pool", False)
                ),
                "use_differential_mode": self.use_differential_mode,
                "token_g_kind": self.token_g_kind != "none",
            }
            active_conflicts = [name for name, active in conflicts.items() if active]
            if active_conflicts:
                raise ValueError(
                    "enable_rival_conditional_cost isolates the ground-cost change "
                    "and cannot be combined with: "
                    + ", ".join(active_conflicts)
                )
        if _bool_config(enable_evidence_marginals):
            if self.marginal_kind == "discriminative":
                raise ValueError(
                    "enable_evidence_marginals cannot be combined with marginal_kind='discriminative'; "
                    "evidence marginals replace the DMUOT discriminative marginal system"
                )
            if enable_multiscale_ot:
                raise ValueError("enable_evidence_marginals is not supported with enable_multiscale_ot")
        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )
        if self.enable_evidence_adaptive_relaxation_uot:
            if not self.uses_unbalanced_transport:
                raise ValueError("EAR-UOT requires unbalanced transport")
            if self.ot_backend != "native":
                raise ValueError("EAR-UOT currently requires ot_backend='native'")
        if self.ours_final_marginal_mode == "score_aligned":
            self.utility_contrastive_marginals = UtilityContrastiveMarginals(
                tau=self.score_marginal_tau,
                marginal_mix=self.score_marginal_mix,
                adaptive_mix=self.score_marginal_adaptive_mix,
                confidence_power=self.score_marginal_confidence_power,
                eps=self.eps,
            )
        if self.enable_ours_final_failure_probe:
            self.ours_final_failure_probe = OursFinalFailureProbe(
                common_margin=self.ours_probe_common_margin,
                eps=self.eps,
            )
        if self.enable_dustbin_contrastive_score:
            self.dustbin_contrastive_residual_score = DustbinContrastiveResidualScore(
                beta=self.dcr_beta,
                tau=self.dcr_tau,
                alpha=self.dcr_alpha,
                margin=self.dcr_margin,
                min_gate=self.dcr_min_gate,
                detach_gate=self.dcr_detach_gate,
                eps=self.eps,
            )
        if self.enable_evidence_adaptive_relaxation_uot:
            self.evidence_adaptive_relaxation_uot = EvidenceAdaptiveRelaxationUOT(
                tau_min=self.ear_uot_tau_min,
                tau_max=self.ear_uot_tau_max,
                temperature=self.ear_uot_temperature,
                margin=self.ear_uot_margin,
                spatial_mix=self.ear_uot_spatial_mix,
                kernel_size=self.ear_uot_kernel_size,
                detach_reliability=self.ear_uot_detach_reliability,
                eps=self.eps,
            )
        if self.enable_rival_conditional_cost:
            self.rival_conditional_ground_cost = RivalConditionalGroundCost(
                penalty_weight=self.rc_cost_weight,
                temperature=self.rc_cost_temperature,
                mode=self.rc_cost_mode,
                eps=self.eps,
            )
        self.enable_pulse_region_uot = bool(enable_pulse_region_uot)
        self.enable_region_structural_uot = bool(enable_region_structural_uot)
        self.enable_adaptive_region_uot = bool(enable_adaptive_region_uot)
        self._pulse_saliency_cache: list[torch.Tensor] | None = None
        self._homotopy_progress = 0.0
        self.pulse_saliency_image_weight = float(pulse_saliency_image_weight)
        self.pulse_saliency_feature_weight = float(pulse_saliency_feature_weight)
        self.pulse_saliency_contrast_weight = float(pulse_saliency_contrast_weight)
        self.pulse_region_train_strength = float(pulse_region_train_strength)
        self.pulse_region_eval_strength = float(pulse_region_eval_strength)
        self.pulse_region_multishot_train_strength = (
            None if pulse_region_multishot_train_strength is None else float(pulse_region_multishot_train_strength)
        )
        self.pulse_region_multishot_eval_strength = (
            None if pulse_region_multishot_eval_strength is None else float(pulse_region_multishot_eval_strength)
        )
        self.pulse_region_train_schedule = str(pulse_region_train_schedule)
        for name, value in (
            ("pulse_region_train_strength", self.pulse_region_train_strength),
            ("pulse_region_eval_strength", self.pulse_region_eval_strength),
            ("pulse_region_multishot_train_strength", self.pulse_region_multishot_train_strength),
            ("pulse_region_multishot_eval_strength", self.pulse_region_multishot_eval_strength),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.pulse_support_consensus_weight = float(pulse_support_consensus_weight)
        self.pulse_support_consensus_beta = float(pulse_support_consensus_beta)
        self.pulse_support_consensus_eta = float(pulse_support_consensus_eta)
        if not 0.0 <= self.pulse_support_consensus_weight <= 1.0:
            raise ValueError("pulse_support_consensus_weight must be in [0, 1]")
        if self.pulse_support_consensus_beta <= 0.0:
            raise ValueError("pulse_support_consensus_beta must be positive")
        if not 0.0 <= self.pulse_support_consensus_eta <= 1.0:
            raise ValueError("pulse_support_consensus_eta must be in [0, 1]")
        self.pulse_evidence_score = bool(pulse_evidence_score)
        self.pulse_evidence_score_mode = str(pulse_evidence_score_mode)
        self.pulse_evidence_score_mix = float(pulse_evidence_score_mix)
        self.pulse_evidence_mass_weight = float(pulse_evidence_mass_weight)
        self.pulse_evidence_cost_weight = float(pulse_evidence_cost_weight)
        self.pulse_background_penalty = float(pulse_background_penalty)
        self.enable_discriminative_uot = bool(enable_discriminative_uot)
        self.discriminative_uot_tau = float(discriminative_uot_tau)
        self.discriminative_uot_margin = float(discriminative_uot_margin)
        self.discriminative_uot_mix = float(discriminative_uot_mix)
        self.discriminative_uot_background_penalty = float(discriminative_uot_background_penalty)
        self.discriminative_uot_mass_weight = float(discriminative_uot_mass_weight)
        self.discriminative_uot_cost_weight = float(discriminative_uot_cost_weight)
        self.pulse_discriminative_evidence = bool(pulse_discriminative_evidence)
        self.pulse_discriminative_tau = float(pulse_discriminative_tau)
        self.pulse_discriminative_margin = float(pulse_discriminative_margin)
        self.pulse_discriminative_mix = float(pulse_discriminative_mix)
        self.enable_episode_contrastive_uot = bool(enable_episode_contrastive_uot)
        self.ect_uot_tau = float(ect_uot_tau)
        self.ect_uot_margin = float(ect_uot_margin)
        self.ect_uot_cost_weight = float(ect_uot_cost_weight)
        self.ect_uot_query_mass_mix = float(ect_uot_query_mass_mix)
        self.ect_uot_query_tau = float(ect_uot_query_tau)
        self.ect_uot_detach = bool(ect_uot_detach)
        self.enable_verified_uot_score = bool(enable_verified_uot_score)
        self.verified_uot_beta = float(verified_uot_beta)
        self.verified_uot_tau = float(verified_uot_tau)
        self.verified_uot_ratio_threshold = float(verified_uot_ratio_threshold)
        self.verified_uot_kernel_size = int(verified_uot_kernel_size)
        self.enable_reciprocal_verified_uot = bool(enable_reciprocal_verified_uot)
        self.rvuot_beta = float(rvuot_beta)
        self.rvuot_tau = float(rvuot_tau)
        self.rvuot_ratio_threshold = float(rvuot_ratio_threshold)
        self.rvuot_kernel_size = int(rvuot_kernel_size)
        self.rvuot_cost_quantile = float(rvuot_cost_quantile)
        self.rvuot_min_gate = float(rvuot_min_gate)
        self.rvuot_enable_rival_gate = bool(rvuot_enable_rival_gate)
        self.rvuot_rival_score_mix = float(rvuot_rival_score_mix)
        self.rvuot_rival_tau = float(rvuot_rival_tau)
        self.rvuot_rival_margin = float(rvuot_rival_margin)
        self.rvuot_detach_gate = bool(rvuot_detach_gate)
        self.enable_verified_region_matching_uot = bool(
            enable_verified_region_matching_uot
        )
        self.vrm_lambda = float(vrm_lambda)
        self.vrm_concentration_tau = float(vrm_concentration_tau)
        self.vrm_region_tau = float(vrm_region_tau)
        self.vrm_region_kernel_size = int(vrm_region_kernel_size)
        self.vrm_use_concentration = bool(vrm_use_concentration)
        self.vrm_use_region_patch = bool(vrm_use_region_patch)
        self.vrm_use_rival = bool(vrm_use_rival)
        self.vrm_rival_tau = float(vrm_rival_tau)
        self.vrm_rival_threshold = float(vrm_rival_threshold)
        self.vrm_rival_evidence_tau = float(vrm_rival_evidence_tau)
        self.vrm_rival_margin = float(vrm_rival_margin)
        self.vrm_detach_gate = bool(vrm_detach_gate)
        self.enable_global_residual_score = bool(enable_global_residual_score)
        self.global_residual_mode = str(global_residual_mode)
        self.global_residual_weight = float(global_residual_weight)
        self.enable_nr_ot = bool(enable_nr_ot)
        self.nr_ot_mode = str(nr_ot_mode)
        self.nr_ot_weight = float(nr_ot_weight)
        self.nr_ot_novelty_temp = float(nr_ot_novelty_temp)
        self.nr_ot_uniform_reference = bool(nr_ot_uniform_reference)
        self.nr_ot_eval_only = bool(nr_ot_eval_only)
        self.nr_ot_gate_temperature = float(nr_ot_gate_temperature)
        self.nr_ot_detach = bool(nr_ot_detach)
        if self.enable_nr_ot and self.enable_global_residual_score:
            raise ValueError(
                "enable_nr_ot and enable_global_residual_score are mutually exclusive: "
                "NR-OT is the principled, background-debiased replacement for the naive "
                "GAP-cosine global residual."
            )
        self.enable_residual_aligned_uot = bool(enable_residual_aligned_uot)
        self.rauot_tau = float(rauot_tau)
        self.rauot_margin = float(rauot_margin)
        self.rauot_cost_weight = float(rauot_cost_weight)
        self.rauot_mass_mix = float(rauot_mass_mix)
        self.rauot_detach = bool(rauot_detach)
        self.enable_rival_aware_evidence_uot = bool(enable_rival_aware_evidence_uot)
        self.rae_uot_tau = float(rae_uot_tau)
        self.rae_uot_marginal_mix = float(rae_uot_marginal_mix)
        self.enable_hubness_calibrated_uot = bool(enable_hubness_calibrated_uot)
        self.hcuot_topk = int(hcuot_topk)
        self.hcuot_temperature = float(hcuot_temperature)
        self.hcuot_cost_weight = float(hcuot_cost_weight)
        self.hcuot_marginal_mix = float(hcuot_marginal_mix)
        self.enable_label_ot = bool(enable_label_ot)
        self.label_ot_epsilon = float(label_ot_epsilon)
        self.label_ot_iterations = int(label_ot_iterations)
        self.label_ot_mix = float(label_ot_mix)
        self.label_ot_min_queries_per_class = int(label_ot_min_queries_per_class)
        self.label_ot_min_column_imbalance = float(label_ot_min_column_imbalance)
        self.label_ot_max_bias = float(label_ot_max_bias)
        for name, value in (
            ("pulse_evidence_mass_weight", self.pulse_evidence_mass_weight),
            ("pulse_evidence_cost_weight", self.pulse_evidence_cost_weight),
            ("pulse_background_penalty", self.pulse_background_penalty),
            ("pulse_evidence_score_mix", self.pulse_evidence_score_mix),
            ("discriminative_uot_margin", self.discriminative_uot_margin),
            ("discriminative_uot_background_penalty", self.discriminative_uot_background_penalty),
            ("discriminative_uot_mass_weight", self.discriminative_uot_mass_weight),
            ("discriminative_uot_cost_weight", self.discriminative_uot_cost_weight),
            ("pulse_discriminative_margin", self.pulse_discriminative_margin),
            ("ect_uot_margin", self.ect_uot_margin),
            ("ect_uot_cost_weight", self.ect_uot_cost_weight),
            ("verified_uot_ratio_threshold", self.verified_uot_ratio_threshold),
            ("rvuot_ratio_threshold", self.rvuot_ratio_threshold),
            ("rvuot_rival_margin", self.rvuot_rival_margin),
            ("vrm_lambda", self.vrm_lambda),
            ("vrm_rival_margin", self.vrm_rival_margin),
            ("global_residual_weight", self.global_residual_weight),
            ("nr_ot_weight", self.nr_ot_weight),
            ("nr_ot_gate_temperature", self.nr_ot_gate_temperature),
            ("rauot_margin", self.rauot_margin),
            ("rauot_cost_weight", self.rauot_cost_weight),
            ("hcuot_cost_weight", self.hcuot_cost_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.nr_ot_gate_temperature <= 0.0:
            raise ValueError("nr_ot_gate_temperature must be positive")
        if self.enable_residual_aligned_uot and not self.enable_global_residual_score:
            raise ValueError("enable_residual_aligned_uot requires enable_global_residual_score")
        if self.rauot_tau <= 0.0:
            raise ValueError("rauot_tau must be positive")
        if not 0.0 <= self.rauot_mass_mix <= 1.0:
            raise ValueError("rauot_mass_mix must be in [0, 1]")
        if self.score_marginal_tau <= 0.0:
            raise ValueError("score_marginal_tau must be positive")
        if not 0.0 <= self.score_marginal_mix <= 1.0:
            raise ValueError("score_marginal_mix must be in [0, 1]")
        if self.score_marginal_confidence_power <= 0.0:
            raise ValueError("score_marginal_confidence_power must be positive")
        if self.ours_probe_common_margin < 0.0:
            raise ValueError("ours_probe_common_margin must be non-negative")
        if self.ear_uot_tau_min <= 0.0 or self.ear_uot_tau_max < self.ear_uot_tau_min:
            raise ValueError("EAR-UOT requires 0 < tau_min <= tau_max")
        if self.ear_uot_temperature <= 0.0:
            raise ValueError("ear_uot_temperature must be positive")
        if self.ear_uot_margin < 0.0:
            raise ValueError("ear_uot_margin must be non-negative")
        if not 0.0 <= self.ear_uot_spatial_mix <= 1.0:
            raise ValueError("ear_uot_spatial_mix must be in [0, 1]")
        if self.ear_uot_kernel_size < 1 or self.ear_uot_kernel_size % 2 == 0:
            raise ValueError("ear_uot_kernel_size must be a positive odd integer")
        if self.rc_cost_weight < 0.0:
            raise ValueError("rc_cost_weight must be non-negative")
        if self.rc_cost_temperature <= 0.0:
            raise ValueError("rc_cost_temperature must be positive")
        if self.rae_uot_tau <= 0.0:
            raise ValueError("rae_uot_tau must be positive")
        if not 0.0 <= self.rae_uot_marginal_mix <= 1.0:
            raise ValueError("rae_uot_marginal_mix must be in [0, 1]")
        if self.hcuot_topk < 1:
            raise ValueError("hcuot_topk must be positive")
        if self.hcuot_temperature <= 0.0:
            raise ValueError("hcuot_temperature must be positive")
        if not 0.0 <= self.hcuot_marginal_mix <= 1.0:
            raise ValueError("hcuot_marginal_mix must be in [0, 1]")
        if self.enable_hubness_calibrated_uot:
            self.hubness_calibrated_uot = HubnessCalibratedUOT(
                topk=self.hcuot_topk,
                temperature=self.hcuot_temperature,
                cost_weight=self.hcuot_cost_weight,
                marginal_mix=self.hcuot_marginal_mix,
                eps=self.eps,
            )
        if not 0.0 <= self.rvuot_rival_score_mix <= 1.0:
            raise ValueError("rvuot_rival_score_mix must be in [0, 1]")
        if not 0.0 <= self.verified_uot_beta <= 1.0:
            raise ValueError("verified_uot_beta must be in [0, 1]")
        if self.verified_uot_tau <= 0.0:
            raise ValueError("verified_uot_tau must be positive")
        if self.verified_uot_kernel_size < 1 or self.verified_uot_kernel_size % 2 != 1:
            raise ValueError("verified_uot_kernel_size must be an odd positive integer")
        if self.enable_verified_region_matching_uot:
            self.verified_region_matching_uot = VerifiedRegionMatchingUOT(
                penalty_lambda=self.vrm_lambda,
                concentration_tau=self.vrm_concentration_tau,
                region_tau=self.vrm_region_tau,
                region_kernel_size=self.vrm_region_kernel_size,
                use_concentration=self.vrm_use_concentration,
                use_region_patch=self.vrm_use_region_patch,
                use_rival=self.vrm_use_rival,
                rival_tau=self.vrm_rival_tau,
                rival_threshold=self.vrm_rival_threshold,
                rival_evidence_tau=self.vrm_rival_evidence_tau,
                rival_margin=self.vrm_rival_margin,
                detach_gate=self.vrm_detach_gate,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if self.enable_reciprocal_verified_uot:
            self.reciprocal_verified_transport = ReciprocalVerifiedTransport(
                beta=self.rvuot_beta,
                tau=self.rvuot_tau,
                ratio_threshold=self.rvuot_ratio_threshold,
                kernel_size=self.rvuot_kernel_size,
                cost_quantile=self.rvuot_cost_quantile,
                min_gate=self.rvuot_min_gate,
                enable_rival_gate=self.rvuot_enable_rival_gate,
                rival_score_mix=self.rvuot_rival_score_mix,
                rival_tau=self.rvuot_rival_tau,
                rival_margin=self.rvuot_rival_margin,
                detach_gate=self.rvuot_detach_gate,
                eps=self.eps,
            )
        if not 0.0 <= self.pulse_evidence_score_mix <= 1.0:
            raise ValueError("pulse_evidence_score_mix must be in [0, 1]")
        if self.discriminative_uot_tau <= 0.0:
            raise ValueError("discriminative_uot_tau must be positive")
        if not 0.0 <= self.discriminative_uot_mix <= 1.0:
            raise ValueError("discriminative_uot_mix must be in [0, 1]")
        if self.pulse_discriminative_tau <= 0.0:
            raise ValueError("pulse_discriminative_tau must be positive")
        if not 0.0 <= self.pulse_discriminative_mix <= 1.0:
            raise ValueError("pulse_discriminative_mix must be in [0, 1]")
        if self.ect_uot_tau <= 0.0:
            raise ValueError("ect_uot_tau must be positive")
        if self.ect_uot_query_tau <= 0.0:
            raise ValueError("ect_uot_query_tau must be positive")
        if not 0.0 <= self.ect_uot_query_mass_mix <= 1.0:
            raise ValueError("ect_uot_query_mass_mix must be in [0, 1]")
        if self.label_ot_epsilon <= 0.0:
            raise ValueError("label_ot_epsilon must be positive")
        if self.label_ot_iterations < 1:
            raise ValueError("label_ot_iterations must be positive")
        if not 0.0 <= self.label_ot_mix <= 1.0:
            raise ValueError("label_ot_mix must be in [0, 1]")
        if self.label_ot_min_queries_per_class < 1:
            raise ValueError("label_ot_min_queries_per_class must be positive")
        if self.label_ot_min_column_imbalance < 0.0:
            raise ValueError("label_ot_min_column_imbalance must be non-negative")
        if self.label_ot_max_bias < 0.0:
            raise ValueError("label_ot_max_bias must be non-negative")
        if self.enable_pulse_region_uot:
            self.pulse_region_guidance = PulseRegionGuidance(
                kernel_size=pulse_region_kernel_size,
                region_cost_weight=pulse_region_cost_weight,
                saliency_mass_mix=pulse_saliency_mass_mix,
                saliency_cost_discount=pulse_saliency_cost_discount,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if self.enable_adaptive_region_uot:
            self.adaptive_region_uot = AdaptiveRegionUOTGuidance(
                token_dim=self.token_dim,
                num_slots=adaptive_region_num_slots,
                context_kernels=adaptive_region_context_kernels,
                cost_discount=adaptive_region_cost_discount,
                mass_mix=adaptive_region_mass_mix,
                sinkhorn_epsilon=adaptive_region_sinkhorn_epsilon,
                sinkhorn_iters=adaptive_region_sinkhorn_iters,
                sinkhorn_tol=self.sinkhorn_tolerance,
                tau=float(self.tau_q),
                fine_gate_quantile=adaptive_region_fine_gate_quantile,
                temperature_min=adaptive_region_temperature_min,
                temperature_max=adaptive_region_temperature_max,
                init_gate=adaptive_region_init_gate,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if self.enable_region_structural_uot:
            self.region_structural_uot = RegionStructuralUOTGuidance(
                grid_size=region_uot_grid_size,
                strength=region_uot_strength,
                fgw_alpha=region_uot_fgw_alpha,
                tau=float(self.tau_q),
                sinkhorn_epsilon=region_uot_sinkhorn_epsilon,
                fgw_iters=region_uot_fgw_iters,
                sinkhorn_iters=region_uot_sinkhorn_iters,
                sinkhorn_tol=self.sinkhorn_tolerance,
                ground_cost=self.ground_cost,
                topk=region_uot_topk,
                fine_gate_quantile=region_uot_fine_gate_quantile,
                min_confidence=region_uot_min_confidence,
                importance_temperature=region_uot_importance_temperature,
                eps=self.eps,
            )
        if enable_multiscale_ot:
            self.enable_multiscale_ot = True
            self.multi_scale_tokenizer = MultiScaleTokenizer(multiscale_pool_sizes)
            self.num_multiscale_scales = len(self.multi_scale_tokenizer.scale_names)
            self.scale_weights = torch.nn.Parameter(torch.zeros(self.num_multiscale_scales, dtype=torch.float32))
            self.multiscale_per_scale_T = bool(multiscale_per_scale_T)
            if self.multiscale_per_scale_T:
                if (
                    len(self.ecot_rho_bank) != 1
                    or self.ecot_m2_ablate_threshold_mass
                    or self.ecot_m2_cost_per_mass_score
                    or self.ecot_m2_per_shot_threshold
                    or self.ecot_m2_use_swts
                ):
                    raise ValueError(
                        "multiscale_per_scale_T requires single-budget active threshold-mass scoring"
                    )
                threshold_init = float(self.transport_cost_threshold.detach().float().cpu().item())
                raw_threshold = _inverse_softplus(max(threshold_init, float(self.eps)))
                self.raw_multiscale_transport_cost_thresholds = torch.nn.Parameter(
                    torch.full((self.num_multiscale_scales,), raw_threshold, dtype=torch.float32)
                )
        self.enable_mspta = bool(enable_mspta)
        self._mspta_cache: list[dict[str, torch.Tensor | tuple[str, ...] | tuple[int, ...]]] | None = None
        if self.enable_mspta:
            self.mspta_tokenizer = MSPTATokenizer(
                scales=mspta_scales,
                mass_mode=mspta_mass_mode,
                compact_floor=mspta_compact_floor,
                vertical_suppression=mspta_vertical_suppression,
            )
            self.mspta_normalize = bool(mspta_normalize)
            self.mspta_proj_dim = int(mspta_proj_dim)
            self.mspta_guidance_source = str(mspta_guidance_source)
            self.mspta_saliency_cost_discount = float(mspta_saliency_cost_discount)
            self.mspta_learnable_weights = bool(mspta_learnable_weights)
            if self.mspta_learnable_weights:
                self.mspta_scale_logits = torch.nn.Parameter(
                    torch.zeros(len(self.mspta_tokenizer.scale_names), dtype=torch.float32)
                )
        self.enable_pot_guide = bool(enable_pot_guide)
        if self.enable_pot_guide:
            from tim_2026.model._reference.modules.pot_guide import POTGuideMarginals

            self.pot_guide_module = POTGuideMarginals(
                fixed_s=pot_guide_s,
                adaptive_s=pot_guide_adaptive_s,
                s_min=pot_guide_s_min,
                s_max=pot_guide_s_max,
                epsilon=pot_guide_epsilon,
                max_iter=pot_guide_max_iter,
                eps=self.eps,
            )
            self._last_pot_guide_diagnostics: dict[str, torch.Tensor] | None = None
        self.enable_evidence_marginals = _bool_config(enable_evidence_marginals)
        if self.enable_evidence_marginals:
            if self.ours_ablation == "gap":
                raise ValueError("enable_evidence_marginals is not supported with ours_ablation='gap'")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_evidence_marginals is not supported with pre_transport_shot_pool")
            self.evidence_marginal_module = CostEvidenceMarginals(
                tau_evidence=float(evidence_tau),
                tau_marginal=float(evidence_tau_marginal),
                mode=str(evidence_mode),
                rival_margin=float(evidence_rival_margin),
                detach_cost=bool(evidence_detach),
                eps=self.eps,
            )
        self.token_attention_query = (
            torch.nn.Parameter(torch.randn(self.token_dim) * 0.01)
            if self.token_g_kind == "learned_attention"
            else None
        )
        self.register_buffer("dm_global_template", torch.empty(0), persistent=False)
        self.register_buffer("dm_template_count", torch.tensor(0.0), persistent=False)
        self.enable_context_enrichment = bool(enable_context_enrichment)
        self._context_debug = bool(context_debug)
        self._context_debug_dir = str(context_debug_dir)
        self._context_debug_max = int(context_debug_max_episodes)
        self._context_debug_count = 0
        if self.enable_context_enrichment:
            context_ks = parse_context_kernel_sizes(context_kernel_sizes_raw)
            self.context_enrichment = SpatialContextEnrichment(
                channels=self.hidden_dim,
                kernel_sizes=context_ks,
                fusion=context_fusion,
                gate_max=context_gate_max,
                change_max=context_change_max,
            )
            self._last_context_diagnostics: dict[str, torch.Tensor] | None = None

        self.enable_structural_augmentation = bool(enable_structural_augmentation)
        if self.enable_structural_augmentation:
            self.structural_augmentation = StructuralTokenAugmentation(
                token_dim=self.token_dim,
                struct_dim=struct_dim,
            )
            self._last_struct_diagnostics: dict[str, torch.Tensor] | None = None
        self.enable_sci = bool(enable_sci)
        if self.enable_sci:
            if int(sci_kernel_size) < 3 or int(sci_kernel_size) % 2 == 0:
                raise ValueError(f"sci_kernel_size must be odd >= 3, got {sci_kernel_size}")
            if int(sci_num_layers) < 1:
                raise ValueError(f"sci_num_layers must be >= 1, got {sci_num_layers}")
            self.spatial_context_injection = SpatialContextInjection(
                token_dim=self.token_dim,
                num_layers=int(sci_num_layers),
                kernel_size=int(sci_kernel_size),
            )
            self._last_sci_diagnostics: dict[str, torch.Tensor] | None = None

        self.enable_nr_ot = bool(getattr(self, "enable_nr_ot", False))
        self._nr_ot_export_evidence = False
        if self.enable_nr_ot:
            self.nuisance_referenced_ot = NuisanceReferencedOT(
                novelty_temp=self.nr_ot_novelty_temp,
                threshold_init=float(
                    torch.as_tensor(self.transport_cost_threshold).detach().clamp_min(self.eps)
                ),
                uniform_reference_marginal=self.nr_ot_uniform_reference,
            )
            self._last_nr_ot_diagnostics: dict[str, torch.Tensor] | None = None

    def set_nr_ot_evidence_export_context(self, active: bool) -> None:
        self._nr_ot_export_evidence = bool(active)

    @property
    def uses_ours_gap_control(self) -> bool:
        return self.ours_ablation == "gap"

    @staticmethod
    def _spatial_minmax(values: torch.Tensor, eps: float) -> torch.Tensor:
        flat = values.flatten(1)
        v_min = flat.amin(dim=1).reshape(-1, 1, 1, 1)
        v_max = flat.amax(dim=1).reshape(-1, 1, 1, 1)
        return (values - v_min) / (v_max - v_min).clamp_min(float(eps))

    def _compute_pulse_saliency(
        self,
        images: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """Return token-grid saliency for bright/energetic pulse regions."""
        with torch.no_grad():
            spatial_hw = feature_map.shape[-2:]
            image_energy = images.detach().float().abs().mean(dim=1, keepdim=True)
            image_energy = F.adaptive_avg_pool2d(image_energy, output_size=spatial_hw)
            feature_energy = feature_map.detach().float().pow(2).sum(dim=1, keepdim=True).sqrt()

            image_norm = self._spatial_minmax(image_energy, self.eps)
            feature_norm = self._spatial_minmax(feature_energy, self.eps)
            base = 0.5 * (image_norm + feature_norm)
            local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            contrast = self._spatial_minmax((base - local_mean).clamp_min(0.0), self.eps)

            weights = image_norm.new_tensor(
                [
                    max(self.pulse_saliency_image_weight, 0.0),
                    max(self.pulse_saliency_feature_weight, 0.0),
                    max(self.pulse_saliency_contrast_weight, 0.0),
                ]
            )
            weights = weights / weights.sum().clamp_min(self.eps)
            saliency = weights[0] * image_norm + weights[1] * feature_norm + weights[2] * contrast
            saliency = saliency.clamp_min(0.0)
            return saliency.flatten(1).to(device=feature_map.device, dtype=feature_map.dtype)

    def _pulse_region_evidence_mask(
        self,
        saliency: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Convert token saliency into a soft connected pulse-region evidence mask."""
        if saliency.dim() < 2:
            raise ValueError(f"saliency must have a token dimension, got {tuple(saliency.shape)}")
        original_shape = saliency.shape
        token_count = int(original_shape[-1])
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        flat = torch.nan_to_num(
            saliency.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0).reshape(-1, token_count)
        values_min = flat.amin(dim=-1, keepdim=True)
        values_max = flat.amax(dim=-1, keepdim=True)
        denom = (values_max - values_min).clamp_min(self.eps)
        base = (flat - values_min) / denom
        base = torch.where(
            (values_max - values_min) > self.eps,
            base,
            torch.ones_like(base),
        )
        if token_count != height * width:
            return base.reshape(original_shape).to(device=saliency.device, dtype=saliency.dtype)

        evidence_map = base.reshape(-1, 1, height, width)
        kernel = int(getattr(getattr(self, "pulse_region_guidance", None), "kernel_size", 3))
        if kernel > 1:
            padding = kernel // 2
            expanded = F.max_pool2d(evidence_map, kernel_size=kernel, stride=1, padding=padding)
            regional = F.avg_pool2d(
                expanded,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
            evidence_map = 0.5 * evidence_map + 0.5 * regional
        evidence = evidence_map.flatten(1)
        peak = evidence.amax(dim=-1, keepdim=True)
        evidence = torch.where(peak > self.eps, evidence / peak.clamp_min(self.eps), torch.ones_like(evidence))
        return evidence.clamp(0.0, 1.0).reshape(original_shape).to(device=saliency.device, dtype=saliency.dtype)

    def _pulse_discriminative_pair_evidence(
        self,
        *,
        flat_cost: torch.Tensor,
        query_evidence: torch.Tensor,
        support_evidence: torch.Tensor,
        way_num: int,
        shot_num: int,
        strength: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(query_evidence.shape) != (num_query, query_len):
            raise ValueError(
                "query_evidence must have shape (NumQuery, QueryTokens), "
                f"got {tuple(query_evidence.shape)}"
            )
        if tuple(support_evidence.shape) != (int(way_num), int(shot_num), support_len):
            raise ValueError(
                "support_evidence must have shape (Way, Shot, SupportTokens), "
                f"got {tuple(support_evidence.shape)}"
            )

        query_evidence = query_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp(0.0, 1.0)
        support_evidence = support_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp(0.0, 1.0)
        pulse_pair = torch.sqrt(
            (
                query_evidence[:, None, None, :, None]
                * support_evidence[None, :, :, None, :]
            ).clamp_min(0.0)
        )

        if int(way_num) <= 1 or float(getattr(self, "pulse_discriminative_mix", 1.0)) <= 0.0:
            rival_gate = torch.ones_like(pulse_pair)
            rival_advantage = torch.zeros_like(pulse_pair)
        else:
            cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
            tau = max(float(self.pulse_discriminative_tau), float(self.eps))
            margin = flat_cost.new_tensor(float(self.pulse_discriminative_margin))
            gates = []
            advantages = []
            all_classes = torch.arange(int(way_num), device=flat_cost.device)
            for class_idx in range(int(way_num)):
                rival_mask = all_classes != int(class_idx)
                rival_cost = cost_view[:, rival_mask]
                rival_cost = rival_cost.permute(0, 3, 1, 2, 4).reshape(num_query, query_len, -1)
                rival_count = max(int(rival_cost.shape[-1]), 1)
                rival_softmin = -tau * (
                    torch.logsumexp(-rival_cost / tau, dim=-1)
                    - math.log(float(rival_count))
                )
                advantage = rival_softmin[:, None, :, None] - cost_view[:, class_idx] - margin
                gate = torch.sigmoid(advantage / tau)
                advantages.append(advantage)
                gates.append(gate)
            rival_gate = torch.stack(gates, dim=1)
            rival_advantage = torch.stack(advantages, dim=1)

        strength_value = strength.to(device=flat_cost.device, dtype=flat_cost.dtype)
        discriminative_mix = flat_cost.new_tensor(float(self.pulse_discriminative_mix))
        effective_gate = 1.0 + strength_value * discriminative_mix * (rival_gate - 1.0)
        pair_evidence = (pulse_pair * effective_gate).clamp(0.0, 1.0)
        payload = {
            "pulse_discriminative_gate": effective_gate.detach(),
            "pulse_rival_advantage": rival_advantage.detach(),
            "pulse/discriminative_enabled": flat_cost.new_tensor(1.0),
            "pulse/discriminative_tau": flat_cost.new_tensor(float(self.pulse_discriminative_tau)),
            "pulse/discriminative_margin": flat_cost.new_tensor(float(self.pulse_discriminative_margin)),
            "pulse/discriminative_mix": discriminative_mix.detach(),
            "pulse/discriminative_gate_mean": effective_gate.mean().detach(),
            "pulse/discriminative_gate_low_share": (effective_gate < 0.5).to(flat_cost.dtype).mean().detach(),
            "pulse/rival_advantage_mean": rival_advantage.mean().detach(),
        }
        return pair_evidence, payload

    def _origin_discriminative_pair_evidence(
        self,
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")

        pair_prior = torch.ones(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if int(way_num) <= 1 or float(self.discriminative_uot_mix) <= 0.0:
            rival_gate = torch.ones_like(pair_prior)
            rival_advantage = torch.zeros_like(pair_prior)
        else:
            cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
            tau = max(float(self.discriminative_uot_tau), float(self.eps))
            margin = flat_cost.new_tensor(float(self.discriminative_uot_margin))
            gates = []
            advantages = []
            all_classes = torch.arange(int(way_num), device=flat_cost.device)
            for class_idx in range(int(way_num)):
                rival_mask = all_classes != int(class_idx)
                rival_cost = cost_view[:, rival_mask]
                rival_cost = rival_cost.permute(0, 3, 1, 2, 4).reshape(num_query, query_len, -1)
                rival_count = max(int(rival_cost.shape[-1]), 1)
                rival_softmin = -tau * (
                    torch.logsumexp(-rival_cost / tau, dim=-1)
                    - math.log(float(rival_count))
                )
                advantage = rival_softmin[:, None, :, None] - cost_view[:, class_idx] - margin
                gate = torch.sigmoid(advantage / tau)
                advantages.append(advantage)
                gates.append(gate)
            rival_gate = torch.stack(gates, dim=1)
            rival_advantage = torch.stack(advantages, dim=1)

        mix = flat_cost.new_tensor(float(self.discriminative_uot_mix))
        effective_gate = 1.0 + mix * (rival_gate - 1.0)
        pair_evidence = (pair_prior * effective_gate).clamp(0.0, 1.0)
        payload = {
            "discriminative_uot_gate": effective_gate.detach(),
            "discriminative_uot_rival_advantage": rival_advantage.detach(),
            "discriminative_uot/enabled": flat_cost.new_tensor(1.0),
            "discriminative_uot/tau": flat_cost.new_tensor(float(self.discriminative_uot_tau)),
            "discriminative_uot/margin": flat_cost.new_tensor(float(self.discriminative_uot_margin)),
            "discriminative_uot/mix": mix.detach(),
            "discriminative_uot/background_penalty": flat_cost.new_tensor(
                float(self.discriminative_uot_background_penalty)
            ),
            "discriminative_uot/gate_mean": effective_gate.mean().detach(),
            "discriminative_uot/gate_low_share": (effective_gate < 0.5).to(flat_cost.dtype).mean().detach(),
            "discriminative_uot/rival_advantage_mean": rival_advantage.mean().detach(),
        }
        return pair_evidence, payload

    def _episode_contrastive_pair_gate(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        tau: float,
        margin: float,
        detach: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")

        reference_cost = flat_cost.detach() if detach else flat_cost
        cost_view = reference_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        if int(way_num) <= 1:
            gate = torch.ones_like(cost_view)
            return gate, torch.zeros_like(cost_view)

        scale = reference_cost.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(float(self.eps))
        temperature = (float(tau) * scale).clamp_min(float(self.eps))
        margin_cost = float(margin) * scale
        temperature_pair = temperature.reshape(num_query, 1, 1, 1)
        temperature_rival = temperature.reshape(num_query, 1, 1)
        temperature_query = temperature.reshape(num_query, 1)
        margin_pair = margin_cost.reshape(num_query, 1, 1, 1)

        gates = []
        advantages = []
        all_classes = torch.arange(int(way_num), device=flat_cost.device)
        for class_idx in range(int(way_num)):
            rival_mask = all_classes != int(class_idx)
            rival_cost = cost_view[:, rival_mask]
            token_temperature = temperature.reshape(num_query, 1, 1, 1, 1)
            rival_shot_softmin = -token_temperature.squeeze(-1) * torch.logsumexp(
                -rival_cost / token_temperature,
                dim=-1,
            )
            rival_shot_softmin = rival_shot_softmin.permute(0, 3, 1, 2).reshape(
                num_query,
                query_len,
                -1,
            )
            rival_count = max(int(rival_shot_softmin.shape[-1]), 1)
            rival_softmin = -temperature_query * (
                torch.logsumexp(-rival_shot_softmin / temperature_rival, dim=-1)
                - math.log(float(rival_count))
            )
            advantage = rival_softmin[:, None, :, None] - cost_view[:, class_idx] - margin_pair
            gate = torch.sigmoid(advantage / temperature_pair)
            advantages.append(advantage)
            gates.append(gate)
        return torch.stack(gates, dim=1), torch.stack(advantages, dim=1)

    def _apply_episode_contrastive_uot(
        self,
        flat_cost: torch.Tensor,
        *,
        query_weight: torch.Tensor | None,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if not getattr(self, "enable_episode_contrastive_uot", False):
            return flat_cost, query_weight, {}

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        gate, advantage = self._episode_contrastive_pair_gate(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            tau=float(self.ect_uot_tau),
            margin=float(self.ect_uot_margin),
            detach=bool(self.ect_uot_detach),
        )
        gate_flat = gate.reshape(num_query, num_pairs, query_len, support_len)
        scale = flat_cost.detach().std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(float(self.eps))
        penalty = flat_cost.new_tensor(float(self.ect_uot_cost_weight)) * scale * (1.0 - gate_flat)
        guided_cost = flat_cost + penalty.to(device=flat_cost.device, dtype=flat_cost.dtype)

        specificity = gate_flat.amax(dim=-1).clamp_min(float(self.eps))
        specificity_prob = torch.softmax(
            specificity.clamp_min(float(self.eps)).log() / float(self.ect_uot_query_tau),
            dim=-1,
        )
        uniform = torch.full_like(specificity_prob, 1.0 / float(max(query_len, 1)))
        mix = flat_cost.new_tensor(float(self.ect_uot_query_mass_mix))
        contrast_query_weight = (1.0 - mix) * uniform + mix * specificity_prob

        if query_weight is None:
            mixed_query_weight = contrast_query_weight
        else:
            current = query_weight.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp_min(0.0)
            if tuple(current.shape) == (num_query, query_len):
                current = current[:, None, :].expand(-1, num_pairs, -1)
            elif tuple(current.shape) != (num_query, num_pairs, query_len):
                raise ValueError(
                    "query_weight must have shape (NumQuery, QueryTokens) or "
                    "(NumQuery, Way*Shot, QueryTokens), "
                    f"got {tuple(query_weight.shape)}"
                )
            mixed_query_weight = current * contrast_query_weight
            mixed_query_weight = mixed_query_weight / mixed_query_weight.sum(dim=-1, keepdim=True).clamp_min(
                float(self.eps)
            )

        normalized_query_weight = mixed_query_weight / mixed_query_weight.sum(dim=-1, keepdim=True).clamp_min(
            float(self.eps)
        )
        entropy = -(
            normalized_query_weight
            * normalized_query_weight.clamp_min(float(self.eps)).log()
        ).sum(dim=-1)
        payload = {
            "episode_contrast_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ),
            "episode_contrast_pair_gate": gate.detach(),
            "episode_contrast_query_weight": normalized_query_weight.detach(),
            "episode_contrast/enabled": flat_cost.new_tensor(1.0),
            "episode_contrast/tau": flat_cost.new_tensor(float(self.ect_uot_tau)),
            "episode_contrast/margin": flat_cost.new_tensor(float(self.ect_uot_margin)),
            "episode_contrast/cost_weight": flat_cost.new_tensor(float(self.ect_uot_cost_weight)),
            "episode_contrast/query_mass_mix": mix.detach(),
            "episode_contrast/query_tau": flat_cost.new_tensor(float(self.ect_uot_query_tau)),
            "episode_contrast/gate_mean": gate.mean().detach(),
            "episode_contrast/gate_low_share": (gate < 0.5).to(flat_cost.dtype).mean().detach(),
            "episode_contrast/rival_advantage_mean": advantage.mean().detach(),
            "episode_contrast/cost_delta_ratio": (
                penalty.detach().abs().mean() / flat_cost.detach().abs().mean().clamp_min(float(self.eps))
            ),
            "episode_contrast/query_weight_entropy": entropy.mean().detach(),
            "episode_contrast/query_specificity_mean": specificity.mean().detach(),
        }
        return guided_cost, normalized_query_weight, payload

    def _apply_rival_aware_evidence_uot(
        self,
        flat_cost: torch.Tensor,
        *,
        query_weight: torch.Tensor | None,
        support_weight: torch.Tensor | None,
        query_tokens: torch.Tensor | None,
        support_tokens: torch.Tensor | None,
        way_num: int,
        shot_num: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        if not getattr(self, "enable_rival_aware_evidence_uot", False):
            return flat_cost, query_weight, support_weight, None, {}
        if query_tokens is None or support_tokens is None:
            raise ValueError("enable_rival_aware_evidence_uot requires query_tokens and support_tokens")
        if query_weight is not None or support_weight is not None:
            raise ValueError(
                "rival-aware evidence UOT replaces other explicit query/support marginals"
            )
        if flat_cost.dim() != 4:
            raise ValueError(
                f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}"
            )

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        expected_pairs = int(way_num) * int(shot_num)
        if num_pairs != expected_pairs:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={expected_pairs}"
            )
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

        query_unit = F.normalize(query_tokens.detach(), p=2, dim=-1, eps=self.eps)
        support_unit = F.normalize(support_tokens.detach(), p=2, dim=-1, eps=self.eps)
        query_global = F.normalize(
            query_tokens.detach().mean(dim=1),
            p=2,
            dim=-1,
            eps=self.eps,
        )
        support_global = F.normalize(
            support_tokens.detach().mean(dim=2),
            p=2,
            dim=-1,
            eps=self.eps,
        )

        query_cross_reference = F.relu(
            torch.einsum("qld,wsd->qwsl", query_unit, support_global)
        )
        support_cross_reference = F.relu(
            torch.einsum("wsld,qd->qwsl", support_unit, query_global)
        )
        rival_gate, rival_advantage = self._episode_contrastive_pair_gate(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            tau=float(self.rae_uot_tau),
            margin=0.0,
            detach=True,
        )
        rival_specificity = (2.0 * rival_gate - 1.0).clamp_min(0.0)
        query_raw = query_cross_reference * rival_specificity.amax(dim=-1)
        support_raw = support_cross_reference * rival_specificity.amax(dim=-2)

        def normalize_with_uniform_fallback(raw: torch.Tensor) -> torch.Tensor:
            total = raw.sum(dim=-1, keepdim=True)
            normalized = raw / total.clamp_min(float(self.eps))
            uniform = torch.full_like(raw, 1.0 / float(max(raw.shape[-1], 1)))
            return torch.where(total > float(self.eps), normalized, uniform)

        query_prob = normalize_with_uniform_fallback(query_raw)
        support_prob = normalize_with_uniform_fallback(support_raw)
        mix = flat_cost.new_tensor(float(self.rae_uot_marginal_mix))
        query_prob = (
            (1.0 - mix) * torch.full_like(query_prob, 1.0 / float(max(query_len, 1)))
            + mix * query_prob
        )
        support_prob = (
            (1.0 - mix) * torch.full_like(support_prob, 1.0 / float(max(support_len, 1)))
            + mix * support_prob
        )
        query_weight_out = query_prob.reshape(num_query, num_pairs, query_len)
        support_weight_out = support_prob.reshape(num_query, num_pairs, support_len)

        query_peak = query_prob / query_prob.amax(dim=-1, keepdim=True).clamp_min(float(self.eps))
        support_peak = support_prob / support_prob.amax(dim=-1, keepdim=True).clamp_min(float(self.eps))
        pair_evidence = (
            rival_specificity
            * torch.sqrt(
                (
                    query_peak[..., :, None]
                    * support_peak[..., None, :]
                ).clamp_min(0.0)
            )
        ).clamp(0.0, 1.0)

        query_entropy = -(
            query_weight_out * query_weight_out.clamp_min(float(self.eps)).log()
        ).sum(dim=-1)
        support_entropy = -(
            support_weight_out * support_weight_out.clamp_min(float(self.eps)).log()
        ).sum(dim=-1)
        payload = {
            "rae_uot_pair_evidence": pair_evidence.detach(),
            "rae_uot_rival_gate": rival_gate.detach(),
            "rae_uot_rival_specificity": rival_specificity.detach(),
            "rae_uot_query_weight": query_weight_out.detach(),
            "rae_uot_support_weight": support_weight_out.detach(),
            "rae_uot/enabled": flat_cost.new_tensor(1.0),
            "rae_uot/tau": flat_cost.new_tensor(float(self.rae_uot_tau)),
            "rae_uot/marginal_mix": mix.detach(),
            "rae_uot/rival_gate_mean": rival_gate.mean().detach(),
            "rae_uot/specificity_mean": rival_specificity.mean().detach(),
            "rae_uot/evidence_mean": pair_evidence.mean().detach(),
            "rae_uot/evidence_std": pair_evidence.std(unbiased=False).detach(),
            "rae_uot/uniform_fallback_query_share": (
                query_raw.sum(dim=-1) <= float(self.eps)
            ).to(flat_cost.dtype).mean().detach(),
            "rae_uot/uniform_fallback_support_share": (
                support_raw.sum(dim=-1) <= float(self.eps)
            ).to(flat_cost.dtype).mean().detach(),
            "rae_uot/query_weight_entropy": query_entropy.mean().detach(),
            "rae_uot/support_weight_entropy": support_entropy.mean().detach(),
            "rae_uot/query_weight_peak": query_weight_out.amax(dim=-1).mean().detach(),
            "rae_uot/support_weight_peak": support_weight_out.amax(dim=-1).mean().detach(),
            "rae_uot/rival_advantage_mean": rival_advantage.mean().detach(),
        }
        return flat_cost, query_weight_out, support_weight_out, pair_evidence, payload

    def _apply_score_aligned_marginals(
        self,
        flat_cost: torch.Tensor,
        *,
        query_weight: torch.Tensor | None,
        support_weight: torch.Tensor | None,
        way_num: int,
        shot_num: int,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        if self.ours_final_marginal_mode != "score_aligned":
            return query_weight, support_weight, {}
        if query_weight is not None or support_weight is not None:
            raise ValueError(
                "score-aligned Ours-Final marginals replace other explicit marginals"
            )
        return self.utility_contrastive_marginals(
            flat_cost,
            threshold=self.transport_cost_threshold.detach(),
            way_num=way_num,
            shot_num=shot_num,
        )

    def _apply_hubness_calibrated_uot(
        self,
        flat_cost: torch.Tensor,
        *,
        query_weight: torch.Tensor | None,
        support_weight: torch.Tensor | None,
        query_tokens: torch.Tensor | None,
        support_tokens: torch.Tensor | None,
        way_num: int,
        shot_num: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        if not getattr(self, "enable_hubness_calibrated_uot", False):
            return flat_cost, query_weight, support_weight, None, {}
        if query_tokens is None or support_tokens is None:
            raise ValueError("enable_hubness_calibrated_uot requires query_tokens and support_tokens")
        if query_weight is not None or support_weight is not None:
            raise ValueError(
                "hubness-calibrated UOT replaces other explicit query/support marginals"
            )

        (
            guided_cost,
            calibrated_query_weight,
            calibrated_support_weight,
            pair_specificity,
            diagnostics,
        ) = self.hubness_calibrated_uot(
            flat_cost,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=way_num,
            shot_num=shot_num,
        )
        num_query, _, query_len, support_len = flat_cost.shape
        payload = {
            "hcuot_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ).detach(),
            "hcuot_pair_specificity": pair_specificity.detach(),
            "hcuot_query_weight": calibrated_query_weight.detach(),
            "hcuot_support_weight": calibrated_support_weight.detach(),
            **diagnostics,
        }
        return (
            guided_cost,
            calibrated_query_weight,
            calibrated_support_weight,
            pair_specificity,
            payload,
        )

    def _residual_aligned_evidence(
        self,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if tuple(support_tokens.shape[:2]) != (int(way_num), int(shot_num)):
            raise ValueError(
                f"support_tokens shape {tuple(support_tokens.shape)} does not match way/shot={way_num}/{shot_num}"
            )
        query_ref = query_tokens.detach() if getattr(self, "rauot_detach", True) else query_tokens
        support_ref = support_tokens.detach() if getattr(self, "rauot_detach", True) else support_tokens
        query_unit = F.normalize(query_ref, p=2, dim=-1, eps=self.eps)
        support_unit = F.normalize(support_ref, p=2, dim=-1, eps=self.eps)

        support_shot_global = support_ref.mean(dim=2)
        class_proto = F.normalize(support_shot_global.mean(dim=1), p=2, dim=-1, eps=self.eps)
        if int(way_num) > 1:
            residual_anchor = class_proto - class_proto.mean(dim=0, keepdim=True)
            residual_anchor = F.normalize(residual_anchor, p=2, dim=-1, eps=self.eps)
        else:
            residual_anchor = class_proto

        query_align = torch.einsum("qld,wd->qwl", query_unit, residual_anchor)
        support_align = torch.einsum("wsld,wd->wsl", support_unit, residual_anchor)

        def standardized_gate(values: torch.Tensor) -> torch.Tensor:
            center = values - values.mean(dim=-1, keepdim=True)
            scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(self.eps))
            margin = float(self.rauot_margin) * scale
            return torch.sigmoid((center - margin) / (float(self.rauot_tau) * scale).clamp_min(float(self.eps)))

        query_gate = standardized_gate(query_align)
        support_gate = standardized_gate(support_align)
        pair_evidence = torch.sqrt(
            (
                query_gate[:, :, None, :, None]
                * support_gate[None, :, :, None, :]
            ).clamp_min(0.0)
        )
        return query_gate, support_gate, pair_evidence, residual_anchor

    def _mix_residual_aligned_weight(
        self,
        current: torch.Tensor | None,
        target_prob: torch.Tensor,
    ) -> torch.Tensor:
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(float(self.eps))
        uniform = target_prob.new_full(target_prob.shape, 1.0 / float(target_prob.shape[-1]))
        mix = target_prob.new_tensor(float(self.rauot_mass_mix))
        mixed = (1.0 - mix) * uniform + mix * target_prob
        if current is None:
            return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(float(self.eps))
        active = current.to(device=target_prob.device, dtype=target_prob.dtype).clamp_min(0.0)
        if (
            active.dim() == mixed.dim() - 1
            and tuple(active.shape[:-1]) == tuple(mixed.shape[:-2])
            and active.shape[-1] == mixed.shape[-1]
        ):
            active = active.unsqueeze(-2).expand_as(mixed)
        elif active.dim() == 2 and mixed.dim() == 3 and active.shape[0] == mixed.shape[0] * mixed.shape[1]:
            active = active.reshape(mixed.shape[0], mixed.shape[1], mixed.shape[2])
        if tuple(active.shape) != tuple(mixed.shape):
            raise ValueError(
                "residual-aligned UOT weight shape mismatch: "
                f"current={tuple(current.shape)} target={tuple(mixed.shape)}"
            )
        merged = active * mixed
        return merged / merged.sum(dim=-1, keepdim=True).clamp_min(float(self.eps))

    def _apply_residual_aligned_uot(
        self,
        flat_cost: torch.Tensor,
        *,
        query_weight: torch.Tensor | None,
        support_weight: torch.Tensor | None,
        query_tokens: torch.Tensor | None,
        support_tokens: torch.Tensor | None,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        if not getattr(self, "enable_residual_aligned_uot", False):
            return flat_cost, query_weight, support_weight, {}
        if query_tokens is None or support_tokens is None:
            raise ValueError("enable_residual_aligned_uot requires query_tokens and support_tokens")
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        query_gate, support_gate, pair_evidence, residual_anchor = self._residual_aligned_evidence(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=way_num,
            shot_num=shot_num,
        )
        if tuple(query_gate.shape) != (num_query, int(way_num), query_len):
            raise ValueError(
                f"residual query gate shape {tuple(query_gate.shape)} does not match {(num_query, way_num, query_len)}"
            )
        if tuple(support_gate.shape) != (int(way_num), int(shot_num), support_len):
            raise ValueError(
                "residual support gate shape "
                f"{tuple(support_gate.shape)} does not match {(way_num, shot_num, support_len)}"
            )
        cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        evidence_centered = pair_evidence - pair_evidence.mean(dim=(-1, -2), keepdim=True)
        cost_scale = cost_view.detach().std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(float(self.eps))
        delta = -float(self.rauot_cost_weight) * cost_scale * evidence_centered
        guided_cost = (cost_view + delta).clamp_min(0.0).reshape_as(flat_cost)

        pair_query_prob = torch.softmax(
            query_gate[:, :, None, :].expand(num_query, int(way_num), int(shot_num), query_len).reshape(
                num_query,
                num_pairs,
                query_len,
            )
            .clamp_min(float(self.eps))
            .log()
            / float(self.rauot_tau),
            dim=-1,
        )
        support_prob = torch.softmax(support_gate.clamp_min(float(self.eps)).log() / float(self.rauot_tau), dim=-1)
        if float(self.rauot_mass_mix) > 0.0:
            query_weight_out = self._mix_residual_aligned_weight(query_weight, pair_query_prob)
            support_weight_out = self._mix_residual_aligned_weight(support_weight, support_prob)
        else:
            query_weight_out = query_weight
            support_weight_out = support_weight

        q_entropy = -(pair_query_prob * pair_query_prob.clamp_min(float(self.eps)).log()).sum(dim=-1)
        s_entropy = -(support_prob * support_prob.clamp_min(float(self.eps)).log()).sum(dim=-1)
        cost_delta_ratio = delta.abs().mean() / cost_view.detach().abs().mean().clamp_min(float(self.eps))
        payload = {
            "residual_aligned_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ).detach(),
            "residual_aligned_pair_evidence": pair_evidence.detach(),
            "residual_aligned_query_weight": (
                query_weight_out.detach() if torch.is_tensor(query_weight_out) else pair_query_prob.detach()
            ),
            "residual_aligned_support_weight": (
                support_weight_out.detach() if torch.is_tensor(support_weight_out) else support_prob.detach()
            ),
            "residual_aligned/enabled": flat_cost.new_tensor(1.0),
            "residual_aligned/tau": flat_cost.new_tensor(float(self.rauot_tau)),
            "residual_aligned/margin": flat_cost.new_tensor(float(self.rauot_margin)),
            "residual_aligned/cost_weight": flat_cost.new_tensor(float(self.rauot_cost_weight)),
            "residual_aligned/mass_mix": flat_cost.new_tensor(float(self.rauot_mass_mix)),
            "residual_aligned/query_gate_mean": query_gate.mean().detach(),
            "residual_aligned/support_gate_mean": support_gate.mean().detach(),
            "residual_aligned/pair_evidence_mean": pair_evidence.mean().detach(),
            "residual_aligned/pair_evidence_std": pair_evidence.std(unbiased=False).detach(),
            "residual_aligned/cost_delta_ratio": cost_delta_ratio.detach(),
            "residual_aligned/query_weight_entropy": q_entropy.mean().detach(),
            "residual_aligned/support_weight_entropy": s_entropy.mean().detach(),
            "residual_aligned/anchor_norm": residual_anchor.norm(dim=-1).mean().detach(),
        }
        return guided_cost, query_weight_out, support_weight_out, payload

    def _transport_specificity_audit(
        self,
        *,
        flat_cost: torch.Tensor,
        plan: torch.Tensor,
        way_num: int,
        shot_num: int,
        prefix: str,
        threshold: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if flat_cost.dim() != 4 or plan.dim() != 5:
            return {}
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if tuple(plan.shape) != (num_query, int(way_num), int(shot_num), query_len, support_len):
            return {}
        gate, advantage = self._episode_contrastive_pair_gate(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            tau=float(getattr(self, "ect_uot_tau", 0.25)),
            margin=0.0,
            detach=True,
        )
        mass = plan.detach().clamp_min(0.0)
        total_mass = mass.sum(dim=(-1, -2)).clamp_min(float(self.eps))
        common_mask = (gate < 0.5).to(mass.dtype)
        specific_mask = (gate > 0.75).to(mass.dtype)
        common_mass = (mass * common_mask).sum(dim=(-1, -2))
        specific_mass = (mass * specific_mask).sum(dim=(-1, -2))
        common_mass_ratio = common_mass / total_mass
        specific_mass_ratio = specific_mass / total_mass
        cost_view = flat_cost.detach().reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        total_cost = (mass * cost_view).sum(dim=(-1, -2))
        common_cost = (mass * cost_view * common_mask).sum(dim=(-1, -2))
        specific_cost = (mass * cost_view * specific_mask).sum(dim=(-1, -2))
        cost_per_mass = total_cost / total_mass
        if threshold is None or not torch.is_tensor(threshold):
            threshold = self.transport_cost_threshold
        threshold_shot = self._threshold_like_shot_scores(threshold, total_mass)
        common_score_term = threshold_shot * common_mass - common_cost
        specific_score_term = threshold_shot * specific_mass - specific_cost
        score_denom = common_score_term.abs() + specific_score_term.abs() + float(self.eps)
        row_mass = mass.sum(dim=-1)
        row_prob = row_mass / row_mass.sum(dim=-1, keepdim=True).clamp_min(float(self.eps))
        row_entropy = -(row_prob * row_prob.clamp_min(float(self.eps)).log()).sum(dim=-1)
        return {
            f"{prefix}/common_mass_ratio": common_mass_ratio.mean().detach(),
            f"{prefix}/specific_mass_ratio": specific_mass_ratio.mean().detach(),
            f"{prefix}/common_score_term": common_score_term.mean().detach(),
            f"{prefix}/specific_score_term": specific_score_term.mean().detach(),
            f"{prefix}/common_score_positive_rate": (common_score_term > 0.0).to(flat_cost.dtype).mean().detach(),
            f"{prefix}/common_score_abs_share": (common_score_term.abs() / score_denom).mean().detach(),
            f"{prefix}/gate_mean": gate.mean().detach(),
            f"{prefix}/gate_low_share": (gate < 0.5).to(flat_cost.dtype).mean().detach(),
            f"{prefix}/rival_advantage_mean": advantage.mean().detach(),
            f"{prefix}/cost_per_mass": cost_per_mass.mean().detach(),
            f"{prefix}/query_mass_entropy": row_entropy.mean().detach(),
        }

    def _record_pulse_saliency(self, images: torch.Tensor, feature_map: torch.Tensor) -> None:
        if not getattr(self, "enable_pulse_region_uot", False):
            return
        cache = getattr(self, "_pulse_saliency_cache", None)
        if cache is None:
            return
        cache.append(self._compute_pulse_saliency(images, feature_map).detach())

    def set_homotopy_progress(self, progress: float) -> None:
        parent_setter = getattr(super(), "set_homotopy_progress", None)
        if parent_setter is not None:
            parent_setter(progress)
        self._homotopy_progress = float(min(max(float(progress), 0.0), 1.0))

    def _pulse_region_effective_strength(
        self,
        reference: torch.Tensor,
        *,
        shot_num: int,
    ) -> torch.Tensor:
        if not getattr(self, "enable_pulse_region_uot", False):
            return reference.new_tensor(0.0)
        train_strength = self.pulse_region_train_strength
        eval_strength = self.pulse_region_eval_strength
        if int(shot_num) > 1:
            if self.pulse_region_multishot_train_strength is not None:
                train_strength = self.pulse_region_multishot_train_strength
            if self.pulse_region_multishot_eval_strength is not None:
                eval_strength = self.pulse_region_multishot_eval_strength
        if not self.training:
            return reference.new_tensor(eval_strength)

        progress = float(min(max(getattr(self, "_homotopy_progress", 0.0), 0.0), 1.0))
        schedule = getattr(self, "pulse_region_train_schedule", "constant")
        if schedule == "eval_only":
            factor = 0.0
        elif schedule == "decay":
            factor = 1.0 - progress
        elif schedule == "warmup_decay":
            warmup = 0.10
            if progress <= warmup:
                factor = progress / warmup
            else:
                factor = max(0.0, (1.0 - progress) / (1.0 - warmup))
        else:
            factor = 1.0
        return reference.new_tensor(train_strength * factor)

    def _apply_pulse_support_consensus(
        self,
        support_tokens: torch.Tensor,
        support_saliency: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weight = float(getattr(self, "pulse_support_consensus_weight", 0.0))
        if support_tokens.dim() != 4 or support_tokens.shape[1] <= 1 or weight <= 0.0:
            return support_saliency, {}

        consensus_mass, consistency, sigma = _compute_cross_shot_support_marginals(
            support_tokens.unsqueeze(0),
            rho=1.0,
            beta=float(self.pulse_support_consensus_beta),
            eta=float(self.pulse_support_consensus_eta),
            detach=True,
            distance=self.ground_cost,
            eps=float(self.eps),
            min_sigma=float(self.eps),
        )
        consensus_prob = normalize_saliency(
            consensus_mass.squeeze(0).to(device=support_saliency.device, dtype=support_saliency.dtype),
            eps=float(self.eps),
        )
        saliency_prob = normalize_saliency(support_saliency, eps=float(self.eps))
        mixed = (1.0 - weight) * saliency_prob + weight * consensus_prob
        entropy = -(consensus_prob * consensus_prob.clamp_min(self.eps).log()).sum(dim=-1).mean()
        payload = {
            "pulse/support_consensus_weight": support_saliency.new_tensor(weight),
            "pulse/support_consensus_peak": consensus_prob.max(dim=-1).values.mean().detach(),
            "pulse/support_consensus_entropy": entropy.detach(),
            "pulse/support_consistency_mean": consistency.to(
                device=support_saliency.device,
                dtype=support_saliency.dtype,
            ).mean().detach(),
            "pulse/support_consistency_sigma_peak": sigma.to(
                device=support_saliency.device,
                dtype=support_saliency.dtype,
            ).max(dim=-1).values.mean().detach(),
        }
        return mixed, payload

    @staticmethod
    def _combine_token_weight(
        current: torch.Tensor | None,
        pulse_weight: torch.Tensor,
    ) -> torch.Tensor:
        if current is None:
            return pulse_weight
        return current.to(device=pulse_weight.device, dtype=pulse_weight.dtype) * pulse_weight

    def _pulse_saliency_pair(
        self,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_pulse_saliency_cache", None)
        if cache is None or len(cache) < 2:
            query_saliency = query_tokens.detach().norm(dim=-1)
            support_saliency = support_tokens.detach().norm(dim=-1)
            return query_saliency, support_saliency
        query_saliency = cache[0].to(device=query_tokens.device, dtype=query_tokens.dtype)
        support_saliency = cache[1].to(device=support_tokens.device, dtype=support_tokens.dtype)
        if tuple(query_saliency.shape) != tuple(query_tokens.shape[:2]):
            query_saliency = query_tokens.detach().norm(dim=-1)
        if support_saliency.dim() == 2 and support_saliency.shape[0] == way_num * shot_num:
            support_saliency = support_saliency.reshape(way_num, shot_num, support_saliency.shape[-1])
        if tuple(support_saliency.shape) != tuple(support_tokens.shape[:3]):
            support_saliency = support_tokens.detach().norm(dim=-1)
        return query_saliency, support_saliency

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = super().encode(x)
        if self._context_debug and hasattr(self, "_context_debug_images"):
            self._context_debug_images.append(x.detach().cpu())
        if self.enable_context_enrichment:
            enriched = self.context_enrichment(feature_map)
            self._last_context_diagnostics = self.context_enrichment.diagnostics(
                feature_map, enriched,
            )
            if self._context_debug and hasattr(self, "_context_debug_fmaps"):
                self._context_debug_fmaps.append(feature_map.detach().cpu())
            return enriched
        if self._context_debug and hasattr(self, "_context_debug_fmaps"):
            self._context_debug_fmaps.append(feature_map.detach().cpu())
        return feature_map

    def _apply_structural_augmentation(
        self,
        projected: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        if not getattr(self, "enable_structural_augmentation", False):
            return projected
        before = projected
        projected = self.structural_augmentation(projected, spatial_hw)
        self._last_struct_diagnostics = self.structural_augmentation.diagnostics(before, projected)
        return projected

    def _apply_sci(
        self,
        projected: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        if not getattr(self, "enable_sci", False):
            return projected
        before = projected
        enriched = self.spatial_context_injection(projected, spatial_hw)
        # Re-normalize back to unit sphere so cost = 1 - dot(q_sci, s_sci) is a
        # proper cosine distance.  Without this the cross-terms dot(Δq, s) and
        # dot(q, Δs) are O(||Δ||) and corrupt the cost matrix with unstructured
        # noise.  Re-normalization preserves the directional shift from SCI
        # (cosine_shift is unchanged) while eliminating magnitude artefacts.
        enriched = F.normalize(enriched, p=2, dim=-1, eps=self.eps)
        self._last_sci_diagnostics = self.spatial_context_injection.diagnostics(before, enriched)
        return enriched

    @property
    def multiscale_transport_cost_thresholds(self) -> torch.Tensor:
        raw_thresholds = getattr(self, "raw_multiscale_transport_cost_thresholds", None)
        if raw_thresholds is None:
            return self.transport_cost_threshold.reshape(1)
        return F.softplus(raw_thresholds).clamp_min(self.eps)

    def _mspta_scale_multipliers(self, reference: torch.Tensor) -> torch.Tensor:
        scale_names = getattr(getattr(self, "mspta_tokenizer", None), "scale_names", ())
        logits = getattr(self, "mspta_scale_logits", None)
        if logits is None:
            return reference.new_ones((len(scale_names),))
        weights = torch.softmax(logits.to(device=reference.device, dtype=reference.dtype), dim=0)
        return weights * float(len(scale_names))

    @staticmethod
    def _normalize_spatial_guidance(values: torch.Tensor, eps: float) -> torch.Tensor:
        values = values.detach().float()
        values = values - values.amin(dim=(-2, -1), keepdim=True)
        denom = values.amax(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        return values / denom

    @staticmethod
    def _pulse_guidance_kernel_size(height: int, width: int) -> int:
        base = max(5, min(int(height), int(width)) // 10)
        if base % 2 == 0:
            base += 1
        return min(base, 31)

    def _image_pulse_guidance(
        self,
        images: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        image = images.detach().float()
        energy = image.abs().mean(dim=1, keepdim=True)
        color_change = image.std(dim=1, keepdim=True, unbiased=False)
        base = self._normalize_spatial_guidance(0.7 * energy + 0.3 * color_change, self.eps)
        kernel_size = self._pulse_guidance_kernel_size(int(base.shape[-2]), int(base.shape[-1]))
        local_mean = F.avg_pool2d(
            base,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        local_deviation = (base - local_mean).abs()
        dx = F.pad(base[..., :, 1:] - base[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(base[..., 1:, :] - base[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx.pow(2) + dy.pow(2) + float(self.eps))
        local_anomaly = self._normalize_spatial_guidance(local_deviation + 0.5 * gradient, self.eps)
        column_profile = local_anomaly.mean(dim=2, keepdim=True)
        compact = (local_anomaly - column_profile).clamp_min(0.0)
        compact = self._normalize_spatial_guidance(compact, self.eps)
        pooled_avg = F.adaptive_avg_pool2d(compact, output_size=spatial_hw)
        pooled_max = F.adaptive_max_pool2d(compact, output_size=spatial_hw)
        return self._normalize_spatial_guidance(0.5 * pooled_avg + 0.5 * pooled_max, self.eps)

    def _feature_pulse_guidance(self, feature_map: torch.Tensor) -> torch.Tensor:
        feature_energy = feature_map.detach().float().norm(dim=1, keepdim=True)
        feature_norm = self._normalize_spatial_guidance(feature_energy, self.eps)
        local_mean = F.avg_pool2d(
            feature_norm,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        feature_contrast = self._normalize_spatial_guidance((feature_norm - local_mean).abs(), self.eps)
        column_profile = feature_contrast.mean(dim=2, keepdim=True)
        return self._normalize_spatial_guidance((feature_contrast - column_profile).clamp_min(0.0), self.eps)

    def _build_mspta_guidance_map(
        self,
        images: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        source = getattr(self, "mspta_guidance_source", "mixed")
        feature_guidance = self._feature_pulse_guidance(feature_map)
        if source == "feature":
            return feature_guidance.to(device=feature_map.device, dtype=feature_map.dtype)

        image_guidance = self._image_pulse_guidance(images, feature_map.shape[-2:])
        if source == "image":
            return image_guidance.to(device=feature_map.device, dtype=feature_map.dtype)
        guidance = 0.65 * image_guidance + 0.35 * feature_guidance
        return guidance.to(device=feature_map.device, dtype=feature_map.dtype)

    def _encode_mspta_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        base_spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        guidance_map = self._build_mspta_guidance_map(images, feature_map)
        scale_records = self.mspta_tokenizer(feature_map, guidance_map=guidance_map)
        if not scale_records:
            raise RuntimeError("MSPTA tokenizer returned no scales")

        scale_multipliers = self._mspta_scale_multipliers(feature_map)
        euclidean_chunks = []
        hyperbolic_chunks = []
        weight_chunks = []
        scale_lengths = []
        scale_names = []
        for scale_idx, record in enumerate(scale_records):
            projected = self._project_backbone_tokens(record.tokens)
            euclidean_tokens = (
                F.normalize(projected, p=2, dim=-1, eps=self.eps)
                if self.mspta_normalize
                else projected
            )
            ball = self._build_ball(projected)
            hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
            multiplier = scale_multipliers[scale_idx]
            euclidean_tokens = euclidean_tokens * multiplier
            euclidean_chunks.append(euclidean_tokens)
            hyperbolic_chunks.append(hyperbolic_tokens)
            weight_chunks.append(record.weights.to(device=euclidean_tokens.device, dtype=euclidean_tokens.dtype))
            scale_lengths.append(int(record.tokens.shape[-2]))
            scale_names.append(str(record.name))

        euclidean = torch.cat(euclidean_chunks, dim=-2)
        hyperbolic = torch.cat(hyperbolic_chunks, dim=-2)
        token_weight = torch.cat(weight_chunks, dim=-1)
        token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        cache = getattr(self, "_mspta_cache", None)
        if cache is not None:
            cache.append(
                {
                    "weights": token_weight,
                    "scale_multipliers": scale_multipliers,
                    "scale_lengths": tuple(scale_lengths),
                    "scale_names": tuple(scale_names),
                }
            )
        return euclidean, hyperbolic, base_spatial_hw

    def _encode_multiscale_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        scale_records = []
        for scale_name, spatial_hw, tokens in self.multi_scale_tokenizer(feature_map):
            projected = self._project_backbone_tokens(tokens)
            projected = self._apply_cata(projected)
            projected = self._apply_structural_augmentation(projected, spatial_hw)
            euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
            euclidean_tokens = self._apply_sci(euclidean_tokens, spatial_hw)
            scale_records.append(
                {
                    "name": scale_name,
                    "spatial_hw": spatial_hw,
                    "euclidean": euclidean_tokens,
                    "hyperbolic": hyperbolic_tokens,
                }
            )
        if not scale_records:
            raise RuntimeError("Multi-scale tokenizer returned no scales")
        cache = getattr(self, "_multiscale_ot_cache", None)
        if cache is not None:
            cache.append(scale_records)
        first = scale_records[0]
        return first["euclidean"], first["hyperbolic"], first["spatial_hw"]

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if getattr(self, "enable_mspta", False):
            return self._encode_mspta_images(images)
        if getattr(self, "enable_multiscale_ot", False):
            return self._encode_multiscale_images(images)

        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or self.uses_ours_gap_control
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_sci", False)
            or getattr(self, "enable_pulse_region_uot", False)
            or getattr(self, "enable_adaptive_region_uot", False)
        )
        if not need_local:
            return super()._encode_images(images)

        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        if self.uses_ours_gap_control:
            tokens = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1).unsqueeze(1)
            spatial_hw = (1, 1)
        else:
            tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        if not self.uses_ours_gap_control:
            projected = self._apply_cata(projected)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
        self._record_token_g_prenorm(projected)
        euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
        euclidean_tokens = self._apply_sci(euclidean_tokens, spatial_hw)
        return euclidean_tokens, hyperbolic_tokens, spatial_hw

    def _project_tokens_from_images(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_sci", False)
            or getattr(self, "enable_pulse_region_uot", False)
            or getattr(self, "enable_adaptive_region_uot", False)
        )
        if not need_local:
            return super()._project_tokens_from_images(images)
        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        projected = self._apply_cata(projected)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
        self._record_token_g_prenorm(projected)
        return projected, spatial_hw

    def _encode_images_with_amp_norms(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_sci", False)
            or getattr(self, "enable_pulse_region_uot", False)
            or getattr(self, "enable_adaptive_region_uot", False)
        )
        if not need_local:
            return super()._encode_images_with_amp_norms(images)
        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected, pre_proj_norms = self._project_backbone_tokens_with_prenorm(tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            pre_proj_norms = projected.norm(p=2, dim=-1).clamp(min=1e-6)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
        self._record_token_g_prenorm(projected)
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        euclidean_tokens = self._apply_sci(euclidean_tokens, spatial_hw)
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, pre_proj_norms, spatial_hw

    def _record_token_g_prenorm(self, projected: torch.Tensor) -> None:
        cache = getattr(self, "_token_g_prenorm_cache", None)
        if cache is not None:
            cache.append(projected.norm(p=2, dim=-1).clamp_min(float(self.eps)))

    @staticmethod
    def _minmax_normalize_per_image(values: torch.Tensor, eps: float) -> torch.Tensor:
        min_value = values.amin(dim=-1, keepdim=True)
        max_value = values.amax(dim=-1, keepdim=True)
        return (values - min_value) / (max_value - min_value).clamp_min(float(eps))

    def _episode_mean_dist_token_g(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        mean_token = tokens.mean(dim=-2, keepdim=True)
        distance = torch.linalg.vector_norm(tokens - mean_token, dim=-1)
        return self._minmax_normalize_per_image(distance, float(self.eps)).to(dtype=tokens.dtype)

    def _prenorm_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_token_g_prenorm_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError(
                "token_g_kind='token_norm_pre_l2' requires cached projector-output norms. "
                "Use the standard Ours-Final local token path or token_g_kind='episode_mean_dist'."
            )
        query_norm = cache[0].to(device=query_tokens.device, dtype=query_tokens.dtype)
        support_norm = cache[1].to(device=support_tokens.device, dtype=support_tokens.dtype)
        if tuple(query_norm.shape) != tuple(query_tokens.shape[:-1]):
            raise ValueError(
                "Cached query token norms do not match query token shape: "
                f"{tuple(query_norm.shape)} vs {tuple(query_tokens.shape[:-1])}"
            )
        expected_support_flat = int(support_tokens.shape[0] * support_tokens.shape[1])
        if support_norm.dim() != 2 or support_norm.shape[0] != expected_support_flat:
            raise ValueError(
                "Cached support token norms do not match shot-decomposed support shape: "
                f"{tuple(support_norm.shape)} vs flat support count {expected_support_flat}"
            )
        support_norm = support_norm.reshape(
            support_tokens.shape[0],
            support_tokens.shape[1],
            support_tokens.shape[-2],
        )
        return (
            self._minmax_normalize_per_image(query_norm, float(self.eps)),
            self._minmax_normalize_per_image(support_norm, float(self.eps)),
        )

    def _learned_attention_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_attention_query is None:
            raise RuntimeError("token_attention_query not initialized for learned_attention")
        v = self.token_attention_query.to(device=query_tokens.device, dtype=query_tokens.dtype)
        d_sqrt = math.sqrt(float(v.shape[-1]))
        query_scores = (query_tokens @ v) / d_sqrt
        support_scores = (support_tokens @ v) / d_sqrt
        return (
            self._minmax_normalize_per_image(query_scores, float(self.eps)),
            self._minmax_normalize_per_image(support_scores, float(self.eps)),
        )

    def _compute_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_g_kind == "episode_mean_dist":
            return (
                self._episode_mean_dist_token_g(query_tokens),
                self._episode_mean_dist_token_g(support_tokens),
            )
        if self.token_g_kind == "token_norm_pre_l2":
            return self._prenorm_token_g(query_tokens, support_tokens)
        if self.token_g_kind == "learned_attention":
            return self._learned_attention_token_g(query_tokens, support_tokens)
        raise RuntimeError("token_g_kind='none' does not compute token-g diagnostics")

    def _dmuot_marginal_probs(
        self,
        token_g_query: torch.Tensor,
        token_g_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if math.isinf(float(self.tau_marg)):
            query_prob = torch.full_like(token_g_query, 1.0 / float(token_g_query.shape[-1]))
            support_prob = torch.full_like(token_g_support, 1.0 / float(token_g_support.shape[-1]))
            return query_prob, support_prob
        tau = max(float(self.tau_marg), float(self.eps))
        return torch.softmax(token_g_query / tau, dim=-1), torch.softmax(token_g_support / tau, dim=-1)

    def _dmuot_shot_strength_value(self, shot_num: int, reference: torch.Tensor) -> torch.Tensor:
        if self.dmuot_shot_strength == "inverse_sqrt":
            value = 1.0 / math.sqrt(float(max(1, shot_num)))
        elif self.dmuot_shot_strength == "inverse":
            value = 1.0 / float(max(1, shot_num))
        else:
            value = 1.0
        return reference.new_tensor(value)

    def _blend_dmuot_prob_with_uniform(
        self,
        prob: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        if self.dmuot_shot_strength == "none":
            return prob
        uniform = torch.full_like(prob, 1.0 / float(prob.shape[-1]))
        return strength.to(device=prob.device, dtype=prob.dtype) * prob + (
            1.0 - strength.to(device=prob.device, dtype=prob.dtype)
        ) * uniform

    def _prepare_multiscale_cost_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cost_query_tokens = query_tokens
        cost_support_tokens = support_tokens
        if self.hrot_token_center:
            support_for_center = cost_support_tokens.reshape(
                way_num,
                shot_num,
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
                way_num * shot_num,
                cost_support_tokens.shape[-2],
                cost_support_tokens.shape[-1],
            )
            if self.hrot_token_center_query:
                query_mean = cost_query_tokens.mean(dim=1, keepdim=True)
                cost_query_tokens = cost_query_tokens - query_mean
        return cost_query_tokens, cost_support_tokens

    @staticmethod
    def _weighted_multiscale_value(
        payloads: list[dict[str, torch.Tensor]],
        key: str,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        values = [payload.get(key) for payload in payloads]
        if not values or any(not torch.is_tensor(value) for value in values):
            return None
        first = values[0]
        if any(tuple(value.shape) != tuple(first.shape) for value in values):
            return None
        stacked = torch.stack(values, dim=0)
        view_shape = (weights.shape[0],) + (1,) * first.dim()
        return (weights.reshape(view_shape).to(device=stacked.device, dtype=stacked.dtype) * stacked).sum(dim=0)

    def _forward_multiscale_ecot_budget_bank(
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
        threshold_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del support_tokens, query_tokens, spatial_hw
        if support_weight is not None or query_weight is not None:
            raise ValueError("enable_multiscale_ot does not support explicit query/support weights")
        cache = getattr(self, "_multiscale_ot_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError("Multi-scale OT cache was not populated before ECOT scoring")
        query_scales = cache[0]
        support_scales = cache[1]
        if len(query_scales) != len(support_scales):
            raise ValueError(
                f"Query/support scale count mismatch: {len(query_scales)} vs {len(support_scales)}"
            )
        if len(query_scales) != self.num_multiscale_scales:
            raise ValueError(
                f"Expected {self.num_multiscale_scales} multi-scale token sets, got {len(query_scales)}"
            )

        scale_weights = torch.softmax(self.scale_weights.to(device=flat_cost.device, dtype=flat_cost.dtype), dim=0)
        threshold_per_scale = self.multiscale_transport_cost_thresholds.to(
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )

        payloads: list[dict[str, torch.Tensor]] = []
        scale_shot_logits = []
        scale_shot_cost = []
        scale_shot_mass = []
        for scale_idx, (query_record, support_record) in enumerate(zip(query_scales, support_scales)):
            if query_record["spatial_hw"] != support_record["spatial_hw"]:
                raise ValueError(
                    "Query/support multi-scale token grids must match for "
                    f"{query_record['name']}: {query_record['spatial_hw']} vs {support_record['spatial_hw']}"
                )
            query_euclidean = query_record["euclidean"].to(device=flat_cost.device, dtype=flat_cost.dtype)
            flat_support_euclidean = support_record["euclidean"].to(
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            )
            expected_support = int(way_num) * int(shot_num)
            if flat_support_euclidean.shape[0] != expected_support:
                raise ValueError(
                    "Multi-scale support token cache does not match Way*Shot: "
                    f"{flat_support_euclidean.shape[0]} vs {expected_support}"
                )
            cost_query_tokens, cost_support_tokens = self._prepare_multiscale_cost_tokens(
                query_euclidean,
                flat_support_euclidean,
                way_num=way_num,
                shot_num=shot_num,
            )
            scale_cost = self._ground_cost(cost_query_tokens, cost_support_tokens)
            scale_cost, _tsw_payload = self._apply_tsw_to_cost(
                scale_cost,
                cost_query_tokens,
                cost_support_tokens,
            )
            scale_support_tokens = flat_support_euclidean.reshape(
                way_num,
                shot_num,
                flat_support_euclidean.shape[-2],
                flat_support_euclidean.shape[-1],
            )
            payload = super()._forward_ecot_budget_bank(
                scale_cost,
                way_num=way_num,
                shot_num=shot_num,
                support_weight=None,
                query_weight=None,
                support_tokens=scale_support_tokens,
                query_tokens=query_euclidean,
                spatial_hw=support_record["spatial_hw"],
            )
            if self.multiscale_per_scale_T:
                threshold = threshold_per_scale[scale_idx]
                shot_logits = self.score_scale * (
                    threshold * payload["shot_transported_mass"] - payload["shot_transport_cost"]
                )
                payload = dict(payload)
                payload["shot_logits"] = shot_logits
                payload["ecot_base_score"] = shot_logits
                if "ecot_score" in payload:
                    payload["ecot_score"] = shot_logits
                if "ecot_budget_scores" in payload and payload["ecot_budget_scores"].shape[-1] == 1:
                    payload["ecot_budget_scores"] = shot_logits.unsqueeze(-1)
            payloads.append(payload)
            scale_shot_logits.append(payload["shot_logits"])
            scale_shot_cost.append(payload["shot_transport_cost"])
            scale_shot_mass.append(payload["shot_transported_mass"])

        def fuse_stack(values: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(values, dim=0)
            view_shape = (scale_weights.shape[0],) + (1,) * values[0].dim()
            return (scale_weights.reshape(view_shape) * stacked).sum(dim=0)

        fused_shot_logits = fuse_stack(scale_shot_logits)
        fused_shot_cost = fuse_stack(scale_shot_cost)
        fused_shot_mass = fuse_stack(scale_shot_mass)

        raw_tau_shot = None
        fused_budget_scores = self._weighted_multiscale_value(payloads, "ecot_budget_scores", scale_weights)
        fused_cost_bank = self._weighted_multiscale_value(
            payloads,
            "ecot_shot_transport_cost_bank",
            scale_weights,
        )
        fused_mass_bank = self._weighted_multiscale_value(
            payloads,
            "ecot_shot_transported_mass_bank",
            scale_weights,
        )
        if (
            self.episode_controller is not None
            and fused_budget_scores is not None
            and fused_cost_bank is not None
            and fused_mass_bank is not None
        ):
            diagnostics = self._compute_ecot_diagnostics(fused_budget_scores, fused_cost_bank, fused_mass_bank)
            controller_outputs = self.episode_controller(diagnostics)
            raw_tau_shot = controller_outputs.get("raw_tau_shot")

        logits, transport_cost, transport_mass, shot_pool_weights, tau_shot = self._pool_ecot_shot_scores(
            fused_shot_logits,
            fused_shot_cost,
            fused_shot_mass,
            raw_tau_shot,
        )

        fused_payload = dict(payloads[0])
        fused_payload.update(
            {
                "logits": logits,
                "transport_cost": transport_cost,
                "transported_mass": transport_mass,
                "shot_transport_cost": fused_shot_cost,
                "shot_transported_mass": fused_shot_mass,
                "shot_logits": fused_shot_logits,
                "shot_pool_weights": shot_pool_weights,
            }
        )
        if tau_shot is not None:
            fused_payload["ecot_tau_shot"] = tau_shot
        for key in (
            "rho",
            "shot_rho",
            "ecot_base_score",
            "ecot_score",
            "ecot_budget_scores",
            "ecot_shot_transport_cost_bank",
            "ecot_shot_transported_mass_bank",
            "ecot_m2_mass_for_score_bank",
            "ecot_m2_mass_consensus_bank",
            "ecot_aux_loss",
            "ecot_identity_loss",
            "ecot_policy_entropy",
            "ecot_policy_entropy_loss",
        ):
            fused_value = self._weighted_multiscale_value(payloads, key, scale_weights)
            if fused_value is not None:
                fused_payload[key] = fused_value

        fused_payload["multiscale/scale_weights"] = scale_weights
        for scale_idx, query_record in enumerate(query_scales):
            name = str(query_record["name"])
            fused_payload[f"multiscale/scale_weight_{name}"] = scale_weights[scale_idx]
            fused_payload[f"multiscale/score_{name}_mean"] = scale_shot_logits[scale_idx].mean()
            fused_payload[f"multiscale/mass_{name}_mean"] = scale_shot_mass[scale_idx].mean()
            fused_payload[f"multiscale/cost_{name}_mean"] = scale_shot_cost[scale_idx].mean()
        if self.multiscale_per_scale_T:
            for scale_idx, query_record in enumerate(query_scales):
                name = str(query_record["name"])
                fused_payload[f"multiscale/T_{name}"] = threshold_per_scale[scale_idx]
        return fused_payload

    def _transport_match(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        tau_q: torch.Tensor | None = None,
        tau_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tau_q is not None or tau_c is not None:
            if self.enable_pot_guide:
                raise ValueError("POT guide cannot be combined with token-wise UOT relaxation")
            return super()._transport_match(
                cost,
                rho,
                a=a,
                b=b,
                tau_q=tau_q,
                tau_c=tau_c,
            )
        num_query, num_way, query_tokens, class_tokens = cost.shape
        partial_transport_mass = None
        if self.enable_pot_guide and a is None and b is None:
            a, b, pot_guide_diagnostics = self.pot_guide_module(cost, rho)
            self._last_pot_guide_diagnostics = pot_guide_diagnostics
        else:
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

        if self.uses_partial_transport:
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

    def _local_average_excluding_center(
        self,
        values: torch.Tensor,
        *,
        spatial_hw: tuple[int, int],
        kernel_size: int,
    ) -> torch.Tensor:
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if values.shape[-1] != height * width:
            raise ValueError(
                f"spatial_hw={spatial_hw} does not match token count {values.shape[-1]}"
            )
        if kernel_size <= 1:
            return torch.zeros_like(values)
        pad = kernel_size // 2
        original_shape = values.shape
        grid = values.reshape(-1, 1, height, width)
        kernel = grid.new_ones((1, 1, kernel_size, kernel_size))
        summed = F.conv2d(grid, kernel, padding=pad) - grid
        counts = F.conv2d(torch.ones_like(grid), kernel, padding=pad) - 1.0
        averaged = summed / counts.clamp_min(1.0)
        return averaged.reshape(original_shape)

    def _verified_uot_shot_logits(
        self,
        *,
        flat_cost: torch.Tensor,
        transport_plan: torch.Tensor,
        threshold: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(transport_plan.shape) != (num_query, int(way_num), int(shot_num), query_len, support_len):
            raise ValueError(
                "transport_plan must have shape (Nq, Way, Shot, Lq, Ls), "
                f"got {tuple(transport_plan.shape)}"
            )

        query_hw = spatial_hw if spatial_hw is not None and query_len == int(spatial_hw[0]) * int(spatial_hw[1]) else _infer_token_hw(query_len)
        support_hw = (
            spatial_hw
            if spatial_hw is not None and support_len == int(spatial_hw[0]) * int(spatial_hw[1])
            else _infer_token_hw(support_len)
        )
        cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        active_threshold = threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
        while active_threshold.dim() < cost_view.dim():
            active_threshold = active_threshold.unsqueeze(-1)

        affinity = (active_threshold - cost_view).clamp_min(0.0)
        detached_affinity = affinity.detach()
        q_neighbors = self._local_average_excluding_center(
            detached_affinity.permute(0, 1, 2, 4, 3).reshape(-1, query_len),
            spatial_hw=query_hw,
            kernel_size=self.verified_uot_kernel_size,
        ).reshape(num_query, int(way_num), int(shot_num), support_len, query_len).permute(0, 1, 2, 4, 3)
        s_neighbors = self._local_average_excluding_center(
            detached_affinity.reshape(-1, support_len),
            spatial_hw=support_hw,
            kernel_size=self.verified_uot_kernel_size,
        ).reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        neighborhood_support = torch.sqrt((q_neighbors * s_neighbors).clamp_min(0.0))
        support_ratio = neighborhood_support / detached_affinity.clamp_min(float(self.eps))
        gate = torch.sigmoid(
            (support_ratio - float(self.verified_uot_ratio_threshold))
            / max(float(self.verified_uot_tau), float(self.eps))
        )
        beta = flat_cost.new_tensor(float(self.verified_uot_beta))
        positive_multiplier = (1.0 - beta) + beta * gate

        raw_terms = transport_plan.to(device=flat_cost.device, dtype=flat_cost.dtype) * (
            active_threshold - cost_view
        )
        positive_mask = raw_terms > 0.0
        verified_terms = torch.where(positive_mask, raw_terms * positive_multiplier, raw_terms)
        verified_shot_logits = self.score_scale * verified_terms.sum(dim=(-1, -2))

        positive_multiplier_det = positive_multiplier.detach()
        positive_mask_f = positive_mask.to(dtype=flat_cost.dtype)
        positive_count = positive_mask_f.sum().clamp_min(1.0)
        payload = {
            "verified_uot/enabled": flat_cost.new_tensor(1.0),
            "verified_uot/beta": flat_cost.new_tensor(float(self.verified_uot_beta)),
            "verified_uot/tau": flat_cost.new_tensor(float(self.verified_uot_tau)),
            "verified_uot/ratio_threshold": flat_cost.new_tensor(float(self.verified_uot_ratio_threshold)),
            "verified_uot/kernel_size": flat_cost.new_tensor(float(self.verified_uot_kernel_size)),
            "verified_uot/gate_mean": gate.detach().mean(),
            "verified_uot/gate_positive_mean": (gate.detach() * positive_mask_f).sum() / positive_count,
            "verified_uot/support_ratio_positive_mean": (
                support_ratio.detach() * positive_mask_f
            ).sum() / positive_count,
            "verified_uot/positive_share": positive_mask_f.mean(),
            "verified_uot/positive_suppression_mean": (
                (1.0 - positive_multiplier_det) * positive_mask_f
            ).sum() / positive_count,
            "verified_uot/raw_positive_evidence": raw_terms.clamp_min(0.0).sum(dim=(-1, -2)).detach(),
            "verified_uot/verified_positive_evidence": verified_terms.clamp_min(0.0).sum(dim=(-1, -2)).detach(),
        }
        return verified_shot_logits, payload

    def _apply_verified_uot_score(
        self,
        payload: dict[str, torch.Tensor],
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None,
    ) -> dict[str, torch.Tensor]:
        if not getattr(self, "enable_verified_uot_score", False):
            return payload
        transport_plan = payload.get("transport_plan")
        if not torch.is_tensor(transport_plan):
            return payload
        threshold = payload.get("ecot_threshold")
        if not torch.is_tensor(threshold):
            threshold = self.transport_cost_threshold
        verified_shot_logits, verified_payload = self._verified_uot_shot_logits(
            flat_cost=flat_cost,
            transport_plan=transport_plan,
            threshold=threshold,
            way_num=way_num,
            shot_num=shot_num,
            spatial_hw=spatial_hw,
        )

        old_shot_logits = payload.get("shot_logits")
        if torch.is_tensor(old_shot_logits):
            payload["unverified_shot_logits"] = old_shot_logits.detach()
            verified_payload["verified_uot/shot_logit_delta"] = (
                verified_shot_logits.detach() - old_shot_logits.detach()
            ).mean()
            verified_payload["verified_uot/shot_logit_delta_abs"] = (
                verified_shot_logits.detach() - old_shot_logits.detach()
            ).abs().mean()

        shot_cost = payload.get("shot_transport_cost")
        shot_mass = payload.get("shot_transported_mass")
        if not torch.is_tensor(shot_cost) or not torch.is_tensor(shot_mass):
            raise RuntimeError("Verified-UOT requires shot_transport_cost and shot_transported_mass in ECOT payload")
        tau_shot = payload.get("ecot_tau_shot")
        if torch.is_tensor(tau_shot):
            scaled = verified_shot_logits / tau_shot.clamp_min(float(self.eps))
            if verified_shot_logits.shape[-1] == 1:
                shot_pool_weights = torch.ones_like(verified_shot_logits)
                logits = verified_shot_logits.squeeze(-1)
            else:
                shot_pool_weights = torch.softmax(scaled, dim=-1)
                logits = tau_shot * (
                    torch.logsumexp(scaled, dim=-1) - math.log(float(verified_shot_logits.shape[-1]))
                )
            transport_cost = (shot_pool_weights * shot_cost).sum(dim=-1)
            transport_mass = (shot_pool_weights * shot_mass).sum(dim=-1)
        else:
            logits, transport_cost, transport_mass, shot_pool_weights = self._pool_j_shot_scores(
                verified_shot_logits,
                shot_cost,
                shot_mass,
            )

        payload["shot_logits"] = verified_shot_logits
        payload["logits"] = logits
        payload["class_scores"] = logits
        payload["transport_cost"] = transport_cost
        payload["transported_mass"] = transport_mass
        payload["shot_pool_weights"] = shot_pool_weights
        payload["verified_uot_logits"] = logits.detach()
        if torch.is_tensor(payload.get("ecot_base_score")):
            payload["unverified_ecot_base_score"] = payload["ecot_base_score"].detach()
            payload["ecot_base_score"] = verified_shot_logits
        if torch.is_tensor(payload.get("ecot_score")):
            payload["unverified_ecot_score"] = payload["ecot_score"].detach()
            payload["ecot_score"] = verified_shot_logits
        budget_scores = payload.get("ecot_budget_scores")
        if torch.is_tensor(budget_scores) and budget_scores.shape[-1] == 1:
            payload["unverified_ecot_budget_scores"] = budget_scores.detach()
            payload["ecot_budget_scores"] = verified_shot_logits.unsqueeze(-1)
        payload.update(verified_payload)
        return payload

    def _threshold_like_shot_scores(
        self,
        threshold: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        active = threshold.to(device=reference.device, dtype=reference.dtype)
        if active.dim() == 0:
            return active.expand_as(reference)
        while active.dim() < reference.dim():
            active = active.unsqueeze(0)
        return active.expand_as(reference)

    def _apply_reciprocal_verified_uot(
        self,
        payload: dict[str, torch.Tensor],
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None,
    ) -> dict[str, torch.Tensor]:
        if not getattr(self, "enable_reciprocal_verified_uot", False):
            return payload
        transport_plan = payload.get("transport_plan")
        if not torch.is_tensor(transport_plan):
            return payload
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(transport_plan.shape) != (num_query, int(way_num), int(shot_num), query_len, support_len):
            raise ValueError(
                "transport_plan must have shape (Nq, Way, Shot, Lq, Ls), "
                f"got {tuple(transport_plan.shape)}"
            )

        cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        verified_plan, rvuot_payload = self.reciprocal_verified_transport(
            cost=cost_view,
            plan=transport_plan.to(device=flat_cost.device, dtype=flat_cost.dtype),
            spatial_hw=spatial_hw,
        )
        verified_cost = (verified_plan * cost_view).sum(dim=(-1, -2))
        verified_mass = verified_plan.sum(dim=(-1, -2))
        threshold = payload.get("ecot_threshold")
        if not torch.is_tensor(threshold):
            threshold = self.transport_cost_threshold
        threshold_shot = self._threshold_like_shot_scores(threshold, verified_mass)

        if self.ecot_m2_cost_per_mass_score:
            mass_denom = verified_mass.detach() if self.ecot_m2_cost_per_mass_detach_mass else verified_mass
            verified_shot_logits = (
                -float(self.ecot_m2_cost_per_mass_alpha)
                * self.score_scale
                * verified_cost
                / mass_denom.clamp_min(float(self.eps))
            )
        else:
            mass_for_score = verified_mass
            if self.ecot_m2_mass_score_mode == "shot_consensus":
                alpha_mass = verified_mass.new_tensor(float(self.ecot_m2_consensus_mass_alpha))
                mass_consensus = verified_mass.mean(dim=2, keepdim=True)
                mass_for_score = (1.0 - alpha_mass) * verified_mass + alpha_mass * mass_consensus
                payload["rvuot_shot_mass_for_score"] = mass_for_score.detach()
            threshold_term = (
                torch.zeros_like(threshold_shot)
                if self.ecot_m2_ablate_threshold_mass
                else threshold_shot
            )
            verified_shot_logits = self.score_scale * (threshold_term * mass_for_score - verified_cost)

        old_shot_logits = payload.get("shot_logits")
        if torch.is_tensor(old_shot_logits):
            payload["rvuot_unverified_shot_logits"] = old_shot_logits.detach()
            rvuot_payload["rvuot/shot_logit_delta"] = (
                verified_shot_logits.detach() - old_shot_logits.detach()
            ).mean()
            rvuot_payload["rvuot/shot_logit_delta_abs"] = (
                verified_shot_logits.detach() - old_shot_logits.detach()
            ).abs().mean()
        payload["rvuot_unverified_transport_plan"] = transport_plan.detach()
        if torch.is_tensor(payload.get("shot_transport_cost")):
            payload["rvuot_unverified_shot_transport_cost"] = payload["shot_transport_cost"].detach()
        if torch.is_tensor(payload.get("shot_transported_mass")):
            payload["rvuot_unverified_shot_transported_mass"] = payload["shot_transported_mass"].detach()

        tau_shot = payload.get("ecot_tau_shot")
        if torch.is_tensor(tau_shot):
            scaled = verified_shot_logits / tau_shot.clamp_min(float(self.eps))
            if verified_shot_logits.shape[-1] == 1:
                shot_pool_weights = torch.ones_like(verified_shot_logits)
                logits = verified_shot_logits.squeeze(-1)
            else:
                shot_pool_weights = torch.softmax(scaled, dim=-1)
                logits = tau_shot * (
                    torch.logsumexp(scaled, dim=-1) - math.log(float(verified_shot_logits.shape[-1]))
                )
            transport_cost = (shot_pool_weights * verified_cost).sum(dim=-1)
            transport_mass = (shot_pool_weights * verified_mass).sum(dim=-1)
        else:
            logits, transport_cost, transport_mass, shot_pool_weights = self._pool_j_shot_scores(
                verified_shot_logits,
                verified_cost,
                verified_mass,
            )

        payload["transport_plan"] = verified_plan
        payload["shot_transport_cost"] = verified_cost
        payload["shot_transported_mass"] = verified_mass
        payload["shot_logits"] = verified_shot_logits
        payload["logits"] = logits
        payload["class_scores"] = logits
        payload["transport_cost"] = transport_cost
        payload["transported_mass"] = transport_mass
        payload["shot_pool_weights"] = shot_pool_weights
        payload["rvuot_logits"] = logits.detach()
        if torch.is_tensor(payload.get("ecot_base_score")):
            payload["rvuot_unverified_ecot_base_score"] = payload["ecot_base_score"].detach()
            payload["ecot_base_score"] = verified_shot_logits
        if torch.is_tensor(payload.get("ecot_score")):
            payload["rvuot_unverified_ecot_score"] = payload["ecot_score"].detach()
            payload["ecot_score"] = verified_shot_logits
        budget_scores = payload.get("ecot_budget_scores")
        if torch.is_tensor(budget_scores) and budget_scores.shape[-1] == 1:
            payload["rvuot_unverified_ecot_budget_scores"] = budget_scores.detach()
            payload["ecot_budget_scores"] = verified_shot_logits.unsqueeze(-1)
        payload.update(rvuot_payload)
        return payload

    def _apply_dustbin_contrastive_residual_score(
        self,
        payload: dict[str, torch.Tensor],
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> dict[str, torch.Tensor]:
        if not getattr(self, "enable_dustbin_contrastive_score", False):
            return payload
        transport_plan = payload.get("transport_plan")
        if not torch.is_tensor(transport_plan):
            return payload
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(transport_plan.shape) != (num_query, int(way_num), int(shot_num), query_len, support_len):
            raise ValueError(
                "transport_plan must have shape (Nq, Way, Shot, Lq, Ls), "
                f"got {tuple(transport_plan.shape)}"
            )

        threshold = payload.get("ecot_threshold")
        if not torch.is_tensor(threshold):
            threshold = self.transport_cost_threshold
        cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
        calibrated_unscaled, dcr_payload = self.dustbin_contrastive_residual_score(
            cost=cost_view,
            plan=transport_plan.to(device=flat_cost.device, dtype=flat_cost.dtype),
            threshold=threshold,
        )
        calibrated_shot_logits = self.score_scale * calibrated_unscaled
        old_shot_logits = payload.get("shot_logits")
        if torch.is_tensor(old_shot_logits):
            payload["dcr_unadjusted_shot_logits"] = old_shot_logits.detach()
            dcr_payload["dcr/shot_logit_delta"] = (
                calibrated_shot_logits.detach() - old_shot_logits.detach()
            ).mean()
            dcr_payload["dcr/shot_logit_delta_abs"] = (
                calibrated_shot_logits.detach() - old_shot_logits.detach()
            ).abs().mean()

        shot_cost = payload.get("shot_transport_cost")
        shot_mass = payload.get("shot_transported_mass")
        if not torch.is_tensor(shot_cost) or not torch.is_tensor(shot_mass):
            raise RuntimeError("DCR score requires shot_transport_cost and shot_transported_mass")
        tau_shot = payload.get("ecot_tau_shot")
        if torch.is_tensor(tau_shot):
            scaled = calibrated_shot_logits / tau_shot.clamp_min(float(self.eps))
            if calibrated_shot_logits.shape[-1] == 1:
                shot_pool_weights = torch.ones_like(calibrated_shot_logits)
                logits = calibrated_shot_logits.squeeze(-1)
            else:
                shot_pool_weights = torch.softmax(scaled, dim=-1)
                logits = tau_shot * (
                    torch.logsumexp(scaled, dim=-1) - math.log(float(calibrated_shot_logits.shape[-1]))
                )
            transport_cost = (shot_pool_weights * shot_cost).sum(dim=-1)
            transport_mass = (shot_pool_weights * shot_mass).sum(dim=-1)
        else:
            logits, transport_cost, transport_mass, shot_pool_weights = self._pool_j_shot_scores(
                calibrated_shot_logits,
                shot_cost,
                shot_mass,
            )

        old_logits = payload.get("logits")
        if torch.is_tensor(old_logits):
            payload["dcr_unadjusted_logits"] = old_logits.detach()
        removed_shot = dcr_payload.get("dcr_removed_positive_shot")
        retained_shot = dcr_payload.get("dcr_retained_positive_shot")
        removed_share_shot = dcr_payload.get("dcr_removed_positive_share_shot")
        if torch.is_tensor(removed_shot):
            dcr_payload["dcr_removed_positive"] = (shot_pool_weights * removed_shot).sum(dim=-1)
        if torch.is_tensor(retained_shot):
            dcr_payload["dcr_retained_positive"] = (shot_pool_weights * retained_shot).sum(dim=-1)
        if torch.is_tensor(removed_share_shot):
            dcr_payload["dcr_removed_positive_share"] = (shot_pool_weights * removed_share_shot).sum(dim=-1)

        payload["shot_logits"] = calibrated_shot_logits
        payload["logits"] = logits
        payload["class_scores"] = logits
        payload["transport_cost"] = transport_cost
        payload["transported_mass"] = transport_mass
        payload["shot_pool_weights"] = shot_pool_weights
        payload["dcr_logits"] = logits.detach()
        if torch.is_tensor(payload.get("ecot_base_score")):
            payload["dcr_unadjusted_ecot_base_score"] = payload["ecot_base_score"].detach()
            payload["ecot_base_score"] = calibrated_shot_logits
        if torch.is_tensor(payload.get("ecot_score")):
            payload["dcr_unadjusted_ecot_score"] = payload["ecot_score"].detach()
            payload["ecot_score"] = calibrated_shot_logits
        budget_scores = payload.get("ecot_budget_scores")
        if torch.is_tensor(budget_scores) and budget_scores.shape[-1] == 1:
            payload["dcr_unadjusted_ecot_budget_scores"] = budget_scores.detach()
            payload["ecot_budget_scores"] = calibrated_shot_logits.unsqueeze(-1)
        payload.update(dcr_payload)
        return payload

    def _apply_evidence_adaptive_relaxation_uot(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int] | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        if not getattr(self, "enable_evidence_adaptive_relaxation_uot", False):
            return None, None, {}
        tau_q, tau_c, diagnostics = self.evidence_adaptive_relaxation_uot(
            flat_cost,
            threshold=self.transport_cost_threshold,
            way_num=way_num,
            shot_num=shot_num,
            spatial_hw=spatial_hw,
        )
        return tau_q, tau_c, diagnostics

    def _audit_evidence_adaptive_relaxation_uot(
        self,
        payload: dict[str, torch.Tensor],
        ear_payload: dict[str, torch.Tensor],
        *,
        way_num: int,
        shot_num: int,
    ) -> dict[str, torch.Tensor]:
        if not ear_payload:
            return {}
        plan = payload.get("transport_plan")
        query_reliability = ear_payload.get("ear_uot_query_reliability")
        support_reliability = ear_payload.get("ear_uot_support_reliability")
        if not all(torch.is_tensor(value) for value in (plan, query_reliability, support_reliability)):
            return {}

        num_query, _, _, query_len, support_len = plan.shape
        query_reliability = query_reliability.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
        ).to(device=plan.device, dtype=plan.dtype)
        support_reliability = support_reliability.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            support_len,
        ).to(device=plan.device, dtype=plan.dtype)
        row_mass = plan.sum(dim=-1)
        col_mass = plan.sum(dim=-2)
        shot_mass = plan.sum(dim=(-1, -2)).clamp_min(self.eps)

        reliable_mass_shot = 0.5 * (
            (row_mass * query_reliability).sum(dim=-1)
            + (col_mass * support_reliability).sum(dim=-1)
        )
        low_reliability_mass_shot = 0.5 * (
            (row_mass * (1.0 - query_reliability)).sum(dim=-1)
            + (col_mass * (1.0 - support_reliability)).sum(dim=-1)
        )
        high_reliability_mass_shot = 0.5 * (
            (row_mass * (query_reliability >= 0.5).to(plan.dtype)).sum(dim=-1)
            + (col_mass * (support_reliability >= 0.5).to(plan.dtype)).sum(dim=-1)
        )
        high_reliability_mass_fraction = high_reliability_mass_shot / shot_mass

        base_rho = self._ecot_base_rho_tensor(plan).to(device=plan.device, dtype=plan.dtype)
        target_query = base_rho / float(max(query_len, 1))
        target_support = base_rho / float(max(support_len, 1))
        target_low_mass = 0.5 * (
            target_query * (1.0 - query_reliability).sum(dim=-1)
            + target_support * (1.0 - support_reliability).sum(dim=-1)
        )
        low_reliability_destroyed_fraction = (
            (target_low_mass - low_reliability_mass_shot).clamp_min(0.0)
            / target_low_mass.clamp_min(self.eps)
        )

        mass_field = torch.cat([row_mass, col_mass], dim=-1)
        reliability_field = torch.cat(
            [query_reliability, support_reliability],
            dim=-1,
        )
        mass_centered = mass_field - mass_field.mean(dim=-1, keepdim=True)
        reliability_centered = (
            reliability_field - reliability_field.mean(dim=-1, keepdim=True)
        )
        correlation = (
            (mass_centered * reliability_centered).mean(dim=-1)
            / (
                mass_centered.square().mean(dim=-1).sqrt()
                * reliability_centered.square().mean(dim=-1).sqrt()
            ).clamp_min(self.eps)
        )

        return {
            "ear_uot_reliable_mass": reliable_mass_shot.mean(dim=2).detach(),
            "ear_uot_low_reliability_mass": low_reliability_mass_shot.mean(dim=2).detach(),
            "ear_uot_high_reliability_mass_fraction": (
                high_reliability_mass_fraction.mean(dim=2).detach()
            ),
            "ear_uot_low_reliability_destroyed_fraction": (
                low_reliability_destroyed_fraction.mean(dim=2).detach()
            ),
            "ear_uot/reliable_mass_fraction": (
                reliable_mass_shot / shot_mass
            ).mean().detach(),
            "ear_uot/high_reliability_mass_fraction": (
                high_reliability_mass_fraction.mean().detach()
            ),
            "ear_uot/low_reliability_destroyed_fraction": (
                low_reliability_destroyed_fraction.mean().detach()
            ),
            "ear_uot/mass_reliability_correlation": correlation.mean().detach(),
        }

    def _mspta_weights_for_ecot(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cache = getattr(self, "_mspta_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError("MSPTA cache was not populated before ECOT scoring")
        query_record = cache[0]
        support_record = cache[1]
        query_weight = query_record["weights"]
        support_weight = support_record["weights"]
        if not torch.is_tensor(query_weight) or not torch.is_tensor(support_weight):
            raise RuntimeError("MSPTA cache is missing query/support token weights")

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        expected_support = int(way_num) * int(shot_num)
        if tuple(query_weight.shape) != (num_query, query_len):
            raise ValueError(
                "MSPTA query weights do not match ECOT cost shape: "
                f"{tuple(query_weight.shape)} vs {(num_query, query_len)}"
            )
        if tuple(support_weight.shape) != (expected_support, support_len):
            raise ValueError(
                "MSPTA support weights do not match ECOT cost shape: "
                f"{tuple(support_weight.shape)} vs {(expected_support, support_len)}"
            )
        if num_pairs != expected_support:
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={expected_support}")

        query_weight = query_weight.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_weight = support_weight.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_weight = support_weight.reshape(way_num, shot_num, support_len)

        scale_lengths = query_record.get("scale_lengths", ())
        scale_names = query_record.get("scale_names", ())
        scale_multipliers = query_record.get("scale_multipliers", None)
        if not torch.is_tensor(scale_multipliers):
            scale_multipliers = flat_cost.new_ones((len(scale_names),))
        else:
            scale_multipliers = scale_multipliers.to(device=flat_cost.device, dtype=flat_cost.dtype)

        payload: dict[str, torch.Tensor] = {
            "mspta/token_count": flat_cost.new_tensor(float(query_len)),
            "mspta/query_weight_sum": query_weight.sum(dim=-1).mean(),
            "mspta/support_weight_sum": support_weight.sum(dim=-1).mean(),
            "mspta/scale_multipliers": scale_multipliers,
        }
        for idx, name in enumerate(scale_names):
            length = int(scale_lengths[idx])
            start = sum(int(item) for item in scale_lengths[:idx])
            stop = start + length
            payload[f"mspta/token_count_{name}"] = flat_cost.new_tensor(float(length))
            payload[f"mspta/query_weight_share_{name}"] = query_weight[:, start:stop].sum(dim=-1).mean()
            payload[f"mspta/support_weight_share_{name}"] = support_weight[..., start:stop].sum(dim=-1).mean()
            payload[f"mspta/query_weight_mean_{name}"] = query_weight[:, start:stop].mean()
            payload[f"mspta/support_weight_mean_{name}"] = support_weight[..., start:stop].mean()
            payload[f"mspta/scale_multiplier_{name}"] = scale_multipliers[idx]
        return query_weight, support_weight, payload

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
        threshold_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rc_cost_payload: dict[str, torch.Tensor] = {}
        if self._rival_conditional_cost_is_active():
            flat_cost, rc_cost_payload = self.rival_conditional_ground_cost(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
            )
            rc_cost_payload["rc_cost_guided_cost_matrix"] = flat_cost.reshape(
                flat_cost.shape[0],
                int(way_num),
                int(shot_num),
                flat_cost.shape[-2],
                flat_cost.shape[-1],
            )
        if getattr(self, "enable_multiscale_ot", False):
            return self._forward_multiscale_ecot_budget_bank(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                support_weight=support_weight,
                query_weight=query_weight,
                support_tokens=support_tokens,
                query_tokens=query_tokens,
                spatial_hw=spatial_hw,
                threshold_override=threshold_override,
            )
        dmuot_payload: dict[str, torch.Tensor] = {}
        pulse_payload: dict[str, torch.Tensor] = {}
        discriminative_payload: dict[str, torch.Tensor] = {}
        region_uot_payload: dict[str, torch.Tensor] = {}
        adaptive_region_payload: dict[str, torch.Tensor] = {}
        verified_region_payload: dict[str, torch.Tensor] = {}
        token_g_query = None
        token_g_support = None
        cost_for_transport = flat_cost
        score_query_evidence = None
        score_support_evidence = None
        score_pair_evidence = None
        score_evidence_mass_weight = float(self.pulse_evidence_mass_weight)
        score_evidence_cost_weight = float(self.pulse_evidence_cost_weight)
        score_background_penalty = float(self.pulse_background_penalty)
        score_evidence_mode = self.pulse_evidence_score_mode
        score_evidence_mix = float(self.pulse_evidence_score_mix)
        mspta_payload: dict[str, torch.Tensor] = {}
        if getattr(self, "enable_mspta", False):
            if query_weight is not None or support_weight is not None:
                raise ValueError("enable_mspta cannot be combined with explicit query/support weights")
            query_weight, support_weight, mspta_payload = self._mspta_weights_for_ecot(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
            )
            discount = float(getattr(self, "mspta_saliency_cost_discount", 0.0))
            if discount > 0.0:
                support_weight_flat = support_weight.reshape(way_num * shot_num, support_weight.shape[-1])
                q_peak = query_weight / query_weight.amax(dim=-1, keepdim=True).clamp_min(self.eps)
                s_peak = support_weight_flat / support_weight_flat.amax(dim=-1, keepdim=True).clamp_min(self.eps)
                saliency_pair = torch.sqrt(
                    (q_peak[:, None, :, None] * s_peak[None, :, None, :]).clamp_min(0.0)
                )
                cost_for_transport = cost_for_transport * (1.0 - discount * saliency_pair)
                mspta_payload["mspta_guided_cost_matrix"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )
                mspta_payload["mspta/saliency_cost_discount"] = flat_cost.new_tensor(discount)
                mspta_payload["mspta/cost_delta_ratio"] = (
                    (cost_for_transport.detach() - flat_cost.detach()).abs().mean()
                    / flat_cost.detach().abs().mean().clamp_min(self.eps)
                )
        if self.enable_pot_guide:
            self._last_pot_guide_diagnostics = None

        if getattr(self, "enable_region_structural_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_region_structural_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            cost_for_transport, region_uot_payload = self.region_structural_uot(
                flat_cost=cost_for_transport,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )

        if getattr(self, "enable_adaptive_region_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_adaptive_region_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            (
                adaptive_guided_cost,
                adaptive_query_weight,
                adaptive_support_weight,
                adaptive_region_payload,
            ) = self.adaptive_region_uot(
                flat_cost=cost_for_transport,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )
            cost_for_transport = adaptive_guided_cost
            query_weight = self._combine_token_weight(query_weight, adaptive_query_weight)
            support_weight = self._combine_token_weight(support_weight, adaptive_support_weight)

        if getattr(self, "enable_verified_region_matching_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_verified_region_matching_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            cost_for_transport, verified_region_payload = (
                self.verified_region_matching_uot(
                    flat_cost=cost_for_transport,
                    query_tokens=query_tokens,
                    support_tokens=support_tokens,
                    way_num=way_num,
                    shot_num=shot_num,
                    spatial_hw=spatial_hw,
                )
            )

        if (
            getattr(self, "enable_discriminative_uot", False)
            and not getattr(self, "enable_pulse_region_uot", False)
        ):
            score_pair_evidence, discriminative_payload = self._origin_discriminative_pair_evidence(
                flat_cost=cost_for_transport,
                way_num=way_num,
                shot_num=shot_num,
            )
            score_evidence_mass_weight = float(self.discriminative_uot_mass_weight)
            score_evidence_cost_weight = float(self.discriminative_uot_cost_weight)
            score_background_penalty = float(self.discriminative_uot_background_penalty)

        if getattr(self, "enable_pulse_region_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_pulse_region_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            query_saliency, support_saliency = self._pulse_saliency_pair(
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
            )
            support_saliency, pulse_consensus_payload = self._apply_pulse_support_consensus(
                support_tokens,
                support_saliency,
            )
            pulse_base_cost = cost_for_transport
            pulse_guided_cost, pulse_query_weight, pulse_support_weight, pulse_payload = (
                self.pulse_region_guidance(
                    flat_cost=pulse_base_cost,
                    query_tokens=query_tokens,
                    support_tokens=support_tokens,
                    query_saliency=query_saliency,
                    support_saliency=support_saliency,
                    way_num=way_num,
                    shot_num=shot_num,
                    spatial_hw=spatial_hw,
                )
            )
            pulse_payload.update(pulse_consensus_payload)
            pulse_strength = self._pulse_region_effective_strength(pulse_base_cost, shot_num=shot_num)
            cost_for_transport = pulse_base_cost + pulse_strength * (pulse_guided_cost - pulse_base_cost)
            if pulse_strength.item() < 1.0:
                q_uniform = pulse_query_weight.new_full(
                    pulse_query_weight.shape,
                    1.0 / float(pulse_query_weight.shape[-1]),
                )
                s_uniform = pulse_support_weight.new_full(
                    pulse_support_weight.shape,
                    1.0 / float(pulse_support_weight.shape[-1]),
                )
                pulse_query_weight = q_uniform + pulse_strength * (pulse_query_weight - q_uniform)
                pulse_support_weight = s_uniform + pulse_strength * (pulse_support_weight - s_uniform)
            pulse_payload["pulse/effective_strength"] = pulse_strength.detach()
            if getattr(self, "pulse_evidence_score", True):
                score_query_evidence = self._pulse_region_evidence_mask(query_saliency, spatial_hw)
                score_support_evidence = self._pulse_region_evidence_mask(support_saliency, spatial_hw)
                pulse_strength_value = float(pulse_strength.detach().cpu().item())
                score_evidence_mass_weight = 1.0 + pulse_strength_value * (
                    float(self.pulse_evidence_mass_weight) - 1.0
                )
                score_evidence_cost_weight = 1.0 + pulse_strength_value * (
                    float(self.pulse_evidence_cost_weight) - 1.0
                )
                score_background_penalty = pulse_strength_value * float(self.pulse_background_penalty)
                if pulse_strength.item() < 1.0:
                    q_ones = torch.ones_like(score_query_evidence)
                    s_ones = torch.ones_like(score_support_evidence)
                    score_query_evidence = q_ones + pulse_strength * (score_query_evidence - q_ones)
                    score_support_evidence = s_ones + pulse_strength * (score_support_evidence - s_ones)
                if getattr(self, "pulse_discriminative_evidence", True):
                    score_pair_evidence, discriminative_payload = self._pulse_discriminative_pair_evidence(
                        flat_cost=cost_for_transport,
                        query_evidence=score_query_evidence,
                        support_evidence=score_support_evidence,
                        way_num=way_num,
                        shot_num=shot_num,
                        strength=pulse_strength,
                    )
                    pulse_payload.update(discriminative_payload)
                pulse_payload.update(
                    {
                        "pulse_query_evidence": score_query_evidence.detach(),
                        "pulse_support_evidence": score_support_evidence.detach(),
                        "pulse/evidence_score_enabled": flat_cost.new_tensor(1.0),
                        "pulse/evidence_score_mode_id": flat_cost.new_tensor(
                            0.0 if self.pulse_evidence_score_mode == "replace" else 1.0
                        ),
                        "pulse/evidence_score_mix": flat_cost.new_tensor(
                            float(self.pulse_evidence_score_mix)
                        ),
                        "pulse/evidence_mass_weight": flat_cost.new_tensor(score_evidence_mass_weight),
                        "pulse/evidence_cost_weight": flat_cost.new_tensor(score_evidence_cost_weight),
                        "pulse/background_penalty": flat_cost.new_tensor(score_background_penalty),
                        "pulse/background_penalty_config": flat_cost.new_tensor(
                            float(self.pulse_background_penalty)
                        ),
                        "pulse/query_evidence_area50": (score_query_evidence > 0.5).to(flat_cost.dtype).mean().detach(),
                        "pulse/support_evidence_area50": (
                            score_support_evidence > 0.5
                        ).to(flat_cost.dtype).mean().detach(),
                    }
                )
            if pulse_strength.item() < 1.0:
                pulse_payload["pulse_guided_cost_matrix"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )
            query_weight = self._combine_token_weight(query_weight, pulse_query_weight)
            support_weight = self._combine_token_weight(support_weight, pulse_support_weight)

        if self.token_g_kind != "none":
            if query_tokens is None or support_tokens is None:
                raise ValueError("token_g_kind != 'none' requires query_tokens and support_tokens")
            token_g_query, token_g_support = self._compute_token_g(query_tokens, support_tokens)
            dmuot_payload["token_g_query"] = token_g_query
            dmuot_payload["token_g_support"] = token_g_support
            dmuot_strength = self._dmuot_shot_strength_value(shot_num, flat_cost)
            lambda_cost_effective = dmuot_strength * float(self.lambda_cost)
            dmuot_payload["dmuot_shot_strength"] = dmuot_strength.detach()
            dmuot_payload["lambda_cost_effective"] = lambda_cost_effective.detach()

            flat_support_g = token_g_support.reshape(way_num * shot_num, token_g_support.shape[-1])
            cost_modulator = 1.0 + lambda_cost_effective * (
                1.0 - token_g_query[:, None, :, None] * flat_support_g[None, :, None, :]
            )
            cost_modulator = cost_modulator.to(device=flat_cost.device, dtype=flat_cost.dtype)
            dmuot_payload["cost_modulator"] = cost_modulator.reshape(
                flat_cost.shape[0],
                way_num,
                shot_num,
                flat_cost.shape[-2],
                flat_cost.shape[-1],
            )
            if self.lambda_cost > 0.0:
                cost_for_transport = cost_for_transport * cost_modulator
                dmuot_payload["cost_matrix_modulated"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )

            base_rho = self._ecot_base_rho_tensor(flat_cost)
            if self.marginal_kind == "discriminative":
                query_prob, support_prob = self._dmuot_marginal_probs(token_g_query, token_g_support)
                query_prob = self._blend_dmuot_prob_with_uniform(query_prob, dmuot_strength)
                support_prob = self._blend_dmuot_prob_with_uniform(support_prob, dmuot_strength)
                if query_weight is not None:
                    query_prob = query_prob * query_weight.to(device=query_prob.device, dtype=query_prob.dtype)
                    query_prob = query_prob / query_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                if support_weight is not None:
                    support_prob = support_prob * support_weight.to(
                        device=support_prob.device,
                        dtype=support_prob.dtype,
                    )
                    support_prob = support_prob / support_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                query_weight = query_prob
                support_weight = support_prob
                dmuot_payload["marginal_query"] = query_prob * base_rho
                dmuot_payload["marginal_support"] = support_prob * base_rho
            else:
                base_rho = base_rho.to(device=token_g_query.device, dtype=token_g_query.dtype)
                dmuot_payload["marginal_query"] = token_g_query.new_full(
                    token_g_query.shape,
                    1.0 / float(token_g_query.shape[-1]),
                ) * base_rho
                dmuot_payload["marginal_support"] = token_g_support.new_full(
                    token_g_support.shape,
                    1.0 / float(token_g_support.shape[-1]),
                ) * base_rho.to(device=token_g_support.device, dtype=token_g_support.dtype)

        evidence_payload: dict[str, torch.Tensor] = {}
        if getattr(self, "enable_evidence_marginals", False):
            base_rho_ev = self._ecot_base_rho_tensor(flat_cost)
            ev_query_marginal, ev_support_marginal, evidence_payload = (
                self.evidence_marginal_module(
                    cost_for_transport,
                    rho=base_rho_ev,
                    way_num=way_num,
                    shot_num=shot_num,
                )
            )
            if ev_query_marginal is not None:
                query_weight = ev_query_marginal
            if ev_support_marginal is not None:
                support_weight = ev_support_marginal

        score_marginal_payload: dict[str, torch.Tensor] = {}
        query_weight, support_weight, score_marginal_payload = (
            self._apply_score_aligned_marginals(
                cost_for_transport,
                query_weight=query_weight,
                support_weight=support_weight,
                way_num=way_num,
                shot_num=shot_num,
            )
        )

        hcuot_payload: dict[str, torch.Tensor] = {}
        (
            cost_for_transport,
            query_weight,
            support_weight,
            _hcuot_pair_specificity,
            hcuot_payload,
        ) = self._apply_hubness_calibrated_uot(
            cost_for_transport,
            query_weight=query_weight,
            support_weight=support_weight,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=way_num,
            shot_num=shot_num,
        )

        rae_uot_payload: dict[str, torch.Tensor] = {}
        (
            cost_for_transport,
            query_weight,
            support_weight,
            rae_pair_evidence,
            rae_uot_payload,
        ) = self._apply_rival_aware_evidence_uot(
            cost_for_transport,
            query_weight=query_weight,
            support_weight=support_weight,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=way_num,
            shot_num=shot_num,
        )

        episode_contrast_payload: dict[str, torch.Tensor] = {}
        cost_for_transport, query_weight, episode_contrast_payload = self._apply_episode_contrastive_uot(
            cost_for_transport,
            query_weight=query_weight,
            way_num=way_num,
            shot_num=shot_num,
        )
        residual_aligned_payload: dict[str, torch.Tensor] = {}
        cost_for_transport, query_weight, support_weight, residual_aligned_payload = (
            self._apply_residual_aligned_uot(
                cost_for_transport,
                query_weight=query_weight,
                support_weight=support_weight,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
            )
        )
        transport_tau_q, transport_tau_c, ear_uot_payload = (
            self._apply_evidence_adaptive_relaxation_uot(
                cost_for_transport,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
            )
        )

        payload = super()._forward_ecot_budget_bank(
            cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
            support_weight=support_weight,
            query_weight=query_weight,
            support_tokens=support_tokens,
            query_tokens=query_tokens,
            spatial_hw=spatial_hw,
            score_query_evidence=score_query_evidence,
            score_support_evidence=score_support_evidence,
            score_pair_evidence=score_pair_evidence,
            score_evidence_mass_weight=score_evidence_mass_weight,
            score_evidence_cost_weight=score_evidence_cost_weight,
            score_background_penalty=score_background_penalty,
            score_evidence_mode=score_evidence_mode,
            score_evidence_mix=score_evidence_mix,
            transport_tau_q=transport_tau_q,
            transport_tau_c=transport_tau_c,
            threshold_override=threshold_override,
        )
        payload = self._apply_verified_uot_score(
            payload,
            flat_cost=cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
            spatial_hw=spatial_hw,
        )
        payload = self._apply_reciprocal_verified_uot(
            payload,
            flat_cost=cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
            spatial_hw=spatial_hw,
        )
        payload = self._apply_dustbin_contrastive_residual_score(
            payload,
            flat_cost=cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
        )
        if ear_uot_payload:
            payload.update(
                self._audit_evidence_adaptive_relaxation_uot(
                    payload,
                    ear_uot_payload,
                    way_num=way_num,
                    shot_num=shot_num,
                )
            )
            payload.update(ear_uot_payload)
        if torch.is_tensor(payload.get("transport_plan")):
            payload.update(
                self._transport_specificity_audit(
                    flat_cost=cost_for_transport,
                    plan=payload["transport_plan"],
                    way_num=way_num,
                    shot_num=shot_num,
                    prefix="transport_audit",
                    threshold=payload.get("ecot_threshold"),
                )
            )
            if verified_region_payload and torch.is_tensor(
                verified_region_payload.get("verified_region_pair_gate")
            ):
                payload.update(
                    self._verified_region_plan_audit(
                        plan=payload["transport_plan"],
                        gate=verified_region_payload["verified_region_pair_gate"],
                    )
                )
        if torch.is_tensor(payload.get("rvuot_unverified_transport_plan")):
            payload.update(
                self._transport_specificity_audit(
                    flat_cost=cost_for_transport,
                    plan=payload["rvuot_unverified_transport_plan"],
                    way_num=way_num,
                    shot_num=shot_num,
                    prefix="transport_audit_unverified",
                    threshold=payload.get("ecot_threshold"),
                )
            )
        if (
            self.enable_ours_final_failure_probe
            and torch.is_tensor(payload.get("transport_plan"))
        ):
            payload.update(
                self.ours_final_failure_probe(
                    cost_for_transport,
                    payload["transport_plan"],
                    threshold=payload.get(
                        "ecot_threshold",
                        self.transport_cost_threshold,
                    ),
                    way_num=way_num,
                    shot_num=shot_num,
                    logits=payload.get("logits"),
                )
            )
        if score_marginal_payload and torch.is_tensor(payload.get("transport_plan")):
            plan = payload["transport_plan"]
            row_mass = plan.sum(dim=-1)
            col_mass = plan.sum(dim=-2)
            base_rho = self._ecot_base_rho_tensor(plan).to(
                device=plan.device,
                dtype=plan.dtype,
            )
            query_target = (
                score_marginal_payload["score_marginal_query_weight"]
                .to(device=plan.device, dtype=plan.dtype)
                [:, None, None, :]
                * base_rho
            )
            support_target = (
                score_marginal_payload["score_marginal_support_weight"]
                .to(device=plan.device, dtype=plan.dtype)
                .reshape(
                    plan.shape[0],
                    int(way_num),
                    int(shot_num),
                    plan.shape[-1],
                )
                * base_rho
            )
            query_l1 = (row_mass - query_target.expand_as(row_mass)).abs().sum(dim=-1)
            support_l1 = (col_mass - support_target).abs().sum(dim=-1)
            score_marginal_payload.update(
                {
                    "score_marginal/query_l1_drift": query_l1.mean().detach(),
                    "score_marginal/support_l1_drift": support_l1.mean().detach(),
                    "score_marginal/l1_drift": (
                        0.5 * (query_l1.mean() + support_l1.mean())
                    ).detach(),
                }
            )

        if dmuot_payload:
            plan = payload["transport_plan"]
            dmuot_payload["transport_plan_modulated"] = plan
            if "marginal_query" in dmuot_payload and "marginal_support" in dmuot_payload:
                marginal_query = dmuot_payload["marginal_query"].to(device=plan.device, dtype=plan.dtype)
                marginal_support = dmuot_payload["marginal_support"].to(device=plan.device, dtype=plan.dtype)
                row_mass = plan.sum(dim=-1)
                col_mass = plan.sum(dim=-2)
                query_l1 = (
                    row_mass
                    - marginal_query[:, None, None, :].expand_as(row_mass)
                ).abs().sum(dim=-1)
                support_l1 = (col_mass - marginal_support[None, :, :, :].expand_as(col_mass)).abs().sum(dim=-1)
                dmuot_payload["marginal_query_l1_drift"] = query_l1.mean()
                dmuot_payload["marginal_support_l1_drift"] = support_l1.mean()
                dmuot_payload["marginal_l1_drift"] = 0.5 * (
                    dmuot_payload["marginal_query_l1_drift"]
                    + dmuot_payload["marginal_support_l1_drift"]
                )
            payload.update(dmuot_payload)
        if pulse_payload:
            payload.update(pulse_payload)
        if discriminative_payload:
            payload.update(discriminative_payload)
        if region_uot_payload:
            payload.update(region_uot_payload)
        if adaptive_region_payload:
            payload.update(adaptive_region_payload)
        if verified_region_payload:
            payload.update(verified_region_payload)
        if mspta_payload:
            payload.update(mspta_payload)
        if evidence_payload:
            payload.update(evidence_payload)
        if score_marginal_payload:
            payload.update(score_marginal_payload)
        if hcuot_payload:
            payload.update(hcuot_payload)
        if episode_contrast_payload:
            payload.update(episode_contrast_payload)
        if residual_aligned_payload:
            payload.update(residual_aligned_payload)
        if rae_uot_payload:
            payload.update(rae_uot_payload)
        if rc_cost_payload:
            payload.update(rc_cost_payload)
        if self.enable_pot_guide and self._last_pot_guide_diagnostics is not None:
            payload.update(self._last_pot_guide_diagnostics)
        if (
            getattr(self, "enable_global_residual_score", False)
            and query_tokens is not None
            and support_tokens is not None
        ):
            self._apply_global_residual_score(
                payload,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
            )
        if (
            getattr(self, "enable_nr_ot", False)
            and query_tokens is not None
            and support_tokens is not None
        ):
            self._apply_nr_ot_score(
                payload,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
            )
        return payload

    def _global_prototype_logits(
        self,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> torch.Tensor:
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if tuple(support_tokens.shape[:2]) != (int(way_num), int(shot_num)):
            raise ValueError(
                f"support_tokens shape {tuple(support_tokens.shape)} does not match way/shot={way_num}/{shot_num}"
            )
        query_global = F.normalize(query_tokens.mean(dim=1), p=2, dim=-1, eps=self.eps)
        support_shot_global = support_tokens.mean(dim=2)
        support_global = F.normalize(support_shot_global.mean(dim=1), p=2, dim=-1, eps=self.eps)
        logits = self.score_scale * torch.einsum("qd,wd->qw", query_global, support_global)
        return logits - logits.mean(dim=1, keepdim=True)

    def _apply_global_residual_score(
        self,
        payload: dict[str, torch.Tensor],
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> None:
        base_logits = payload.get("logits")
        if not torch.is_tensor(base_logits):
            return
        global_logits = self._global_prototype_logits(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=way_num,
            shot_num=shot_num,
        ).to(device=base_logits.device, dtype=base_logits.dtype)
        if tuple(global_logits.shape) != tuple(base_logits.shape):
            raise ValueError(
                f"global residual logits shape {tuple(global_logits.shape)} does not match base logits "
                f"{tuple(base_logits.shape)}"
            )
        weight = base_logits.new_tensor(float(self.global_residual_weight))
        if self.global_residual_mode == "global_only":
            fused_logits = global_logits
            effective_weight = base_logits.new_tensor(1.0)
            mode_id = base_logits.new_tensor(1.0)
        else:
            fused_logits = base_logits + weight * global_logits
            effective_weight = weight
            mode_id = base_logits.new_tensor(0.0)
        payload["local_scores"] = base_logits
        payload["global_scores"] = global_logits
        payload["logits"] = fused_logits
        payload["class_scores"] = fused_logits
        payload["global_residual_weight"] = effective_weight.detach()
        payload["global_residual_mode_id"] = mode_id.detach()

    def _apply_nr_ot_score(
        self,
        payload: dict[str, torch.Tensor],
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> None:
        """Fuse the Nuisance-Referenced OT score into the payload.

        ``standalone`` mode replaces the base Ours-Final logits with the
        background-debiased NR-OT logits; ``residual`` mode adds them, scaled by
        ``nr_ot_weight``, on top of the base local UOT logits.  ``gated_residual``
        preserves the base decision when the local UOT margin is already high
        and uses NR-OT as an uncertainty-time verifier.
        """
        base_logits = payload.get("logits")
        if not torch.is_tensor(base_logits):
            return
        if self.training and getattr(self, "nr_ot_eval_only", False):
            payload["local_scores"] = base_logits
            diagnostics = {
                "nr_ot/enabled": base_logits.new_tensor(1.0).detach(),
                "nr_ot/mode_id": base_logits.new_tensor(
                    2.0 if self.nr_ot_mode == "gated_residual" else (1.0 if self.nr_ot_mode == "residual" else 0.0)
                ).detach(),
                "nr_ot/eval_only": base_logits.new_tensor(1.0).detach(),
                "nr_ot/skipped_train": base_logits.new_tensor(1.0).detach(),
                "nr_ot/weight": base_logits.new_tensor(float(self.nr_ot_weight)).detach(),
            }
            self._last_nr_ot_diagnostics = diagnostics
            payload.update(diagnostics)
            return
        nr_logits, diagnostics = self.nuisance_referenced_ot(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=int(way_num),
            shot_num=int(shot_num),
            score_scale=float(self.score_scale),
            eps=float(self.sinkhorn_epsilon),
            tau_q=float(self.tau_q),
            tau_c=float(self.tau_c),
            rho=float(self.fixed_mass),
            sinkhorn_iters=int(self.sinkhorn_iterations),
            sinkhorn_tol=float(self.sinkhorn_tolerance),
            return_evidence_payload=bool(getattr(self, "_nr_ot_export_evidence", False)),
        )
        nr_logits = nr_logits.to(device=base_logits.device, dtype=base_logits.dtype)
        if tuple(nr_logits.shape) != tuple(base_logits.shape):
            raise ValueError(
                f"NR-OT logits shape {tuple(nr_logits.shape)} does not match base logits "
                f"{tuple(base_logits.shape)}"
            )
        nr_term = nr_logits.detach() if getattr(self, "nr_ot_detach", False) else nr_logits
        weight = base_logits.new_tensor(float(self.nr_ot_weight))
        gate = base_logits.new_ones((base_logits.shape[0], 1))
        base_margin = base_logits.new_zeros((base_logits.shape[0],))
        if self.nr_ot_mode == "gated_residual":
            if base_logits.shape[1] >= 2:
                top2 = base_logits.detach().topk(2, dim=1).values
                base_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
            gate = torch.exp(-base_margin / float(self.nr_ot_gate_temperature)).unsqueeze(1)
            gate = gate.clamp(0.0, 1.0).to(device=base_logits.device, dtype=base_logits.dtype)
            fused_logits = base_logits + weight * gate * nr_term
        elif self.nr_ot_mode == "residual":
            fused_logits = base_logits + weight * nr_term
        else:  # standalone
            fused_logits = nr_logits
        payload["local_scores"] = base_logits
        payload["nr_ot_scores"] = nr_logits
        payload["logits"] = fused_logits
        payload["class_scores"] = fused_logits
        diagnostics["nr_ot/enabled"] = base_logits.new_tensor(1.0).detach()
        diagnostics["nr_ot/mode_id"] = base_logits.new_tensor(
            2.0 if self.nr_ot_mode == "gated_residual" else (1.0 if self.nr_ot_mode == "residual" else 0.0)
        ).detach()
        diagnostics["nr_ot/eval_only"] = base_logits.new_tensor(
            1.0 if getattr(self, "nr_ot_eval_only", False) else 0.0
        ).detach()
        diagnostics["nr_ot/skipped_train"] = base_logits.new_tensor(0.0).detach()
        diagnostics["nr_ot/detached"] = base_logits.new_tensor(
            1.0 if getattr(self, "nr_ot_detach", False) else 0.0
        ).detach()
        diagnostics["nr_ot/weight"] = weight.detach()
        diagnostics["nr_ot/gate_mean"] = gate.detach().mean()
        diagnostics["nr_ot/gate_active_frac"] = (gate.detach().squeeze(1) > 0.5).to(base_logits.dtype).mean()
        diagnostics["nr_ot/base_margin_mean"] = base_margin.detach().mean()
        diagnostics["nr_ot/logit_delta_abs"] = (nr_logits - base_logits).abs().mean().detach()
        diagnostics["nr_ot/fused_delta_abs"] = (fused_logits - base_logits).abs().mean().detach()
        diagnostics["nr_ot/fused_pred_change_rate"] = (
            fused_logits.detach().argmax(dim=1).ne(base_logits.detach().argmax(dim=1)).to(base_logits.dtype).mean()
        )
        self._last_nr_ot_diagnostics = diagnostics
        payload.update(diagnostics)

    def _update_differential_mode_template(self, episode_template: torch.Tensor) -> None:
        if not self.training:
            return
        with torch.no_grad():
            template = episode_template.detach()
            if tuple(self.dm_global_template.shape) != tuple(template.shape):
                self.dm_global_template = template.clone()
                self.dm_template_count = template.new_tensor(1.0)
                return
            count = self.dm_template_count.to(device=template.device, dtype=template.dtype)
            next_count = count + 1.0
            updated = self.dm_global_template.to(device=template.device, dtype=template.dtype)
            updated = updated + (template - updated) / next_count
            self.dm_global_template = updated.detach()
            self.dm_template_count = next_count.detach()

    def _differential_mode_template(self, support_tokens: torch.Tensor) -> torch.Tensor:
        episode_template = support_tokens.mean(dim=0)
        self._update_differential_mode_template(episode_template)
        if (
            self.training
            or self.dm_alpha <= 0.0
            or self.dm_global_template.numel() == 0
            or tuple(self.dm_global_template.shape) != tuple(episode_template.shape)
        ):
            return episode_template
        global_template = self.dm_global_template.to(
            device=episode_template.device,
            dtype=episode_template.dtype,
        )
        alpha = float(self.dm_alpha)
        return alpha * global_template + (1.0 - alpha) * episode_template

    def _apply_differential_mode_to_cost_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_differential_mode:
            return query_tokens, support_tokens
        if query_tokens.dim() != 3 or support_tokens.dim() != 3:
            raise ValueError(
                "Ours differential mode expects query/support cost tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumSupport, Tokens, Dim)"
            )
        if query_tokens.shape[-2:] != support_tokens.shape[-2:]:
            raise ValueError(
                "Ours differential mode requires aligned spatial token grids, "
                f"got {tuple(query_tokens.shape[-2:])} vs {tuple(support_tokens.shape[-2:])}"
            )

        template = self._differential_mode_template(support_tokens)
        query_diff = query_tokens - template.unsqueeze(0)
        support_diff = support_tokens - template.unsqueeze(0)
        torch._assert(query_diff.shape == query_tokens.shape, "DMT changed query token shape")
        torch._assert(support_diff.shape == support_tokens.shape, "DMT changed support token shape")
        self._last_dm_diagnostics = self._build_differential_mode_diagnostics(
            template,
            query_tokens,
            support_tokens,
            query_diff,
            support_diff,
        )
        self._maybe_export_differential_mode_debug(template, self._last_dm_diagnostics)
        return query_diff, support_diff

    def _build_differential_mode_diagnostics(
        self,
        template: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_diff: torch.Tensor,
        support_diff: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = float(self.eps)
        query_residual_ratio = query_diff.norm() / query_tokens.norm().clamp_min(eps)
        support_residual_ratio = support_diff.norm() / support_tokens.norm().clamp_min(eps)
        residual_ratio = 0.5 * (query_residual_ratio + support_residual_ratio)
        token_norm = template.norm(dim=-1)
        return {
            "dm_mu_norm": token_norm.mean(),
            "dm_mu_max_norm": token_norm.max(),
            "dm_mu_support_ratio": template.norm() / support_tokens.norm().clamp_min(eps),
            "dm_query_residual_ratio": query_residual_ratio,
            "dm_support_residual_ratio": support_residual_ratio,
            "dm_residual_ratio": residual_ratio,
            "dm_alpha": template.new_tensor(float(self.dm_alpha)),
            "dm_template_count": self.dm_template_count.to(device=template.device, dtype=template.dtype),
            "dm_mu_token_norm": token_norm.detach(),
        }

    def _maybe_export_differential_mode_debug(
        self,
        template: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        if not self.dm_debug:
            return
        if self.dm_debug_max_episodes > 0 and self._dm_debug_count >= self.dm_debug_max_episodes:
            return
        self._dm_debug_count += 1
        debug_idx = self._dm_debug_count
        try:
            os.makedirs(self.dm_debug_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug disabled: cannot create {self.dm_debug_dir}: {exc}")
            return

        token_norm = diagnostics["dm_mu_token_norm"].detach().float().cpu()
        token_hw = _infer_token_hw(int(token_norm.numel()))
        heatmap_path = os.path.join(self.dm_debug_dir, f"dmt_mu_heatmap_{debug_idx:04d}.png")

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            heatmap = token_norm.reshape(*token_hw).numpy()
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            image = ax.imshow(heatmap, cmap="magma")
            ax.set_title(
                "Ours DMT common component | "
                f"mu_norm={float(diagnostics['dm_mu_norm'].detach().cpu()):.4f}"
            )
            ax.set_xlabel("token x")
            ax.set_ylabel("token y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - defensive debug path
            heatmap_path = f"failed: {exc}"

        csv_path = os.path.join(self.dm_debug_dir, "dmt_debug_metrics.csv")
        write_header = not os.path.exists(csv_path)
        row = {
            "episode": debug_idx,
            "training": int(self.training),
            "dm_alpha": float(diagnostics["dm_alpha"].detach().cpu()),
            "dm_template_count": float(diagnostics["dm_template_count"].detach().cpu()),
            "dm_mu_norm": float(diagnostics["dm_mu_norm"].detach().cpu()),
            "dm_mu_max_norm": float(diagnostics["dm_mu_max_norm"].detach().cpu()),
            "dm_mu_support_ratio": float(diagnostics["dm_mu_support_ratio"].detach().cpu()),
            "dm_query_residual_ratio": float(diagnostics["dm_query_residual_ratio"].detach().cpu()),
            "dm_support_residual_ratio": float(diagnostics["dm_support_residual_ratio"].detach().cpu()),
            "dm_residual_ratio": float(diagnostics["dm_residual_ratio"].detach().cpu()),
            "heatmap_path": heatmap_path,
        }
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug metrics skipped: cannot write {csv_path}: {exc}")
            return

        print(
            "[Ours-DMT] "
            f"episode={debug_idx} "
            f"mu_norm={row['dm_mu_norm']:.6f} "
            f"mu_support_ratio={row['dm_mu_support_ratio']:.6f} "
            f"residual_ratio={row['dm_residual_ratio']:.6f} "
            f"q_residual={row['dm_query_residual_ratio']:.6f} "
            f"s_residual={row['dm_support_residual_ratio']:.6f} "
            f"heatmap={heatmap_path}"
        )

    def _ground_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens, class_tokens = self._apply_differential_mode_to_cost_tokens(
            query_tokens,
            class_tokens,
        )
        return super()._ground_cost(query_tokens, class_tokens)

    def _label_ot_inactive_payload(
        self,
        logits: torch.Tensor,
        *,
        reason_code: float,
        imbalance_before: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = logits.new_zeros(())
        return {
            "label_ot/enabled": logits.new_tensor(1.0),
            "label_ot/active": zero,
            "label_ot/reason": logits.new_tensor(float(reason_code)),
            "label_ot/epsilon": logits.new_tensor(float(self.label_ot_epsilon)),
            "label_ot/iterations": logits.new_tensor(float(self.label_ot_iterations)),
            "label_ot/mix": logits.new_tensor(float(self.label_ot_mix)),
            "label_ot/column_imbalance_before": zero if imbalance_before is None else imbalance_before.detach(),
            "label_ot/column_imbalance_after": zero if imbalance_before is None else imbalance_before.detach(),
            "label_ot/bias_abs_mean": zero,
            "label_ot/bias_abs_max": zero,
            "label_ot/logit_delta_abs_mean": zero,
            "label_ot/assignment_entropy": zero,
        }

    def _apply_label_ot_to_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not getattr(self, "enable_label_ot", False):
            return logits, {}
        if logits.dim() != 2:
            return logits, self._label_ot_inactive_payload(logits, reason_code=4.0)
        if self.training:
            return logits, self._label_ot_inactive_payload(logits, reason_code=1.0)

        num_query, way_num = logits.shape
        min_queries = int(self.label_ot_min_queries_per_class) * int(way_num)
        if way_num <= 1 or num_query < min_queries:
            return logits, self._label_ot_inactive_payload(logits, reason_code=2.0)

        eps = max(float(self.label_ot_epsilon), float(self.eps))
        row_target = logits.new_ones(num_query)
        col_target = logits.new_full((way_num,), float(num_query) / float(way_num))
        soft_before = torch.softmax(logits / eps, dim=-1)
        col_before = soft_before.sum(dim=0)
        imbalance_before = (col_before - col_target).abs().sum() / float(max(1, num_query))
        if imbalance_before < float(self.label_ot_min_column_imbalance):
            return logits, self._label_ot_inactive_payload(
                logits,
                reason_code=3.0,
                imbalance_before=imbalance_before,
            )

        stabilized = (logits - logits.amax(dim=1, keepdim=True)) / eps
        kernel = stabilized.clamp(min=-60.0, max=60.0).exp().clamp_min(float(self.eps))
        u = logits.new_ones(num_query)
        v = logits.new_ones(way_num)
        for _ in range(int(self.label_ot_iterations)):
            u = row_target / (kernel @ v).clamp_min(float(self.eps))
            v = col_target / (kernel.transpose(0, 1) @ u).clamp_min(float(self.eps))

        bias = eps * v.clamp_min(float(self.eps)).log()
        bias = bias - bias.mean()
        if self.label_ot_max_bias > 0.0:
            bias = bias.clamp(min=-float(self.label_ot_max_bias), max=float(self.label_ot_max_bias))
        delta = float(self.label_ot_mix) * bias.unsqueeze(0)
        adjusted = logits + delta

        assignment = torch.softmax(adjusted / eps, dim=-1)
        col_after = assignment.sum(dim=0)
        imbalance_after = (col_after - col_target).abs().sum() / float(max(1, num_query))
        entropy = -(assignment * assignment.clamp_min(float(self.eps)).log()).sum(dim=-1).mean()
        payload = {
            "label_ot/enabled": logits.new_tensor(1.0),
            "label_ot/active": logits.new_tensor(1.0),
            "label_ot/reason": logits.new_tensor(0.0),
            "label_ot/epsilon": logits.new_tensor(float(self.label_ot_epsilon)),
            "label_ot/iterations": logits.new_tensor(float(self.label_ot_iterations)),
            "label_ot/mix": logits.new_tensor(float(self.label_ot_mix)),
            "label_ot/column_imbalance_before": imbalance_before.detach(),
            "label_ot/column_imbalance_after": imbalance_after.detach(),
            "label_ot/bias_abs_mean": bias.detach().abs().mean(),
            "label_ot/bias_abs_max": bias.detach().abs().amax(),
            "label_ot/logit_delta_abs_mean": delta.detach().abs().mean(),
            "label_ot/assignment_entropy": entropy.detach(),
        }
        return adjusted, payload

    def _apply_label_ot_to_outputs(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        adjusted, payload = self._apply_label_ot_to_logits(logits)
        if not payload:
            return outputs, payload
        if isinstance(outputs, dict):
            outputs = dict(outputs)
            outputs["logits"] = adjusted
            outputs["class_scores"] = adjusted
            outputs.update(payload)
            return outputs, payload
        return adjusted, payload

    def _forward_episode(self, *args, **kwargs) -> torch.Tensor | dict[str, torch.Tensor]:
        self._last_dm_diagnostics = None
        use_context_debug = (
            self._context_debug
            and self._context_debug_count < self._context_debug_max
        )
        if self.enable_context_enrichment:
            self._last_context_diagnostics = None
        if getattr(self, "enable_sci", False):
            self._last_sci_diagnostics = None
        if getattr(self, "enable_nr_ot", False):
            self._last_nr_ot_diagnostics = None
        if use_context_debug:
            self._context_debug_images = []
            self._context_debug_fmaps = []
            if self.enable_context_enrichment:
                self.context_enrichment._debug_cache = []
        use_prenorm_cache = self.token_g_kind == "token_norm_pre_l2"
        use_multiscale_cache = getattr(self, "enable_multiscale_ot", False)
        use_mspta_cache = getattr(self, "enable_mspta", False)
        use_pulse_cache = getattr(self, "enable_pulse_region_uot", False)
        if use_prenorm_cache:
            self._token_g_prenorm_cache = []
        if use_multiscale_cache:
            self._multiscale_ot_cache = []
        if use_mspta_cache:
            self._mspta_cache = []
        if use_pulse_cache:
            self._pulse_saliency_cache = []
        try:
            outputs = super()._forward_episode(*args, **kwargs)
        finally:
            if use_prenorm_cache:
                self._token_g_prenorm_cache = None
            if use_multiscale_cache:
                self._multiscale_ot_cache = None
            if use_mspta_cache:
                self._mspta_cache = None
            if use_pulse_cache:
                self._pulse_saliency_cache = None
            if use_context_debug:
                self._export_context_debug()
                if self.enable_context_enrichment:
                    self.context_enrichment._debug_cache = None
                self._context_debug_images = []
                self._context_debug_fmaps = []
        outputs, _label_ot_payload = self._apply_label_ot_to_outputs(outputs)
        if isinstance(outputs, dict):
            if "mspta_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["mspta_guided_cost_matrix"]
            if "region_uot_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["region_uot_guided_cost_matrix"]
            if "pulse_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["pulse_guided_cost_matrix"]
            if "adaptive_region_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["adaptive_region_guided_cost_matrix"]
            if "episode_contrast_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["episode_contrast_guided_cost_matrix"]
            if "rc_cost_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["rc_cost_guided_cost_matrix"]
            if "hcuot_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["hcuot_guided_cost_matrix"]
            if "verified_region_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["verified_region_guided_cost_matrix"]
            if "residual_aligned_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["residual_aligned_guided_cost_matrix"]
            if self.lambda_cost > 0.0 and "cost_matrix_modulated" in outputs:
                if "base_cost_matrix" not in outputs and "cost_matrix" in outputs:
                    outputs["base_cost_matrix"] = outputs["cost_matrix"]
                outputs["cost_matrix"] = outputs["cost_matrix_modulated"]
            if self._last_dm_diagnostics is not None:
                outputs.update(self._last_dm_diagnostics)
            if self.enable_context_enrichment and self._last_context_diagnostics is not None:
                outputs.update(self._last_context_diagnostics)
            if getattr(self, "enable_structural_augmentation", False) and self._last_struct_diagnostics is not None:
                outputs.update(self._last_struct_diagnostics)
            if getattr(self, "enable_sci", False) and self._last_sci_diagnostics is not None:
                outputs.update(self._last_sci_diagnostics)
            if getattr(self, "enable_nr_ot", False) and self._last_nr_ot_diagnostics is not None:
                outputs.update(self._last_nr_ot_diagnostics)
        return outputs

    def set_rival_conditional_cost_test_context(self, *, is_noise: bool) -> None:
        self._rc_cost_noise_test_context = bool(is_noise)

    def _rival_conditional_cost_is_active(self) -> bool:
        if not self.enable_rival_conditional_cost:
            return False
        if self.rc_cost_scope == "always":
            return True
        if self.training:
            return False
        if self.rc_cost_scope == "eval_only":
            return True
        return self._rc_cost_noise_test_context

    def _verified_region_plan_audit(
        self,
        *,
        plan: torch.Tensor,
        gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gate = gate.to(device=plan.device, dtype=plan.dtype)
        if tuple(gate.shape) != tuple(plan.shape):
            raise ValueError(
                f"VRM gate shape {tuple(gate.shape)} does not match plan shape {tuple(plan.shape)}"
            )
        total_mass = plan.sum().clamp_min(float(self.eps))
        mass_weighted_gate = (plan * gate).sum() / total_mass
        low_gate_mask = gate < 0.5
        rejected_mass_ratio = (
            plan.masked_select(low_gate_mask).sum() / total_mass
            if low_gate_mask.any()
            else plan.new_tensor(0.0)
        )
        high_gate_mask = gate >= 0.5
        accepted_mass_ratio = (
            plan.masked_select(high_gate_mask).sum() / total_mass
            if high_gate_mask.any()
            else plan.new_tensor(0.0)
        )
        plan_flat = plan.reshape(-1)
        gate_flat = gate.reshape(-1)
        flat_count = int(gate_flat.numel())
        top10_k = max(1, int(math.ceil(0.10 * flat_count)))
        top20_k = max(1, int(math.ceil(0.20 * flat_count)))
        top10_threshold = torch.topk(gate_flat, k=top10_k).values.min()
        top20_threshold = torch.topk(gate_flat, k=top20_k).values.min()
        top10_mask = gate >= top10_threshold
        top20_mask = gate >= top20_threshold
        top10_mass_ratio = plan.masked_select(top10_mask).sum() / total_mass
        top20_mass_ratio = plan.masked_select(top20_mask).sum() / total_mass
        top10_gate_mean = gate_flat.topk(top10_k).values.mean()
        top20_gate_mean = gate_flat.topk(top20_k).values.mean()
        plan_centered = plan_flat - plan_flat.mean()
        gate_centered = gate_flat - gate_flat.mean()
        denom = (
            plan_centered.norm() * gate_centered.norm()
        ).clamp_min(float(self.eps))
        correlation = (plan_centered * gate_centered).sum() / denom
        return {
            "verified/accepted_mass_ratio": accepted_mass_ratio.detach(),
            "verified/rejected_mass_ratio": rejected_mass_ratio.detach(),
            "verified/plan_gate_mean": mass_weighted_gate.detach(),
            "verified/plan_gate_correlation": correlation.detach(),
            "verified/top10_gate_mass_ratio": top10_mass_ratio.detach(),
            "verified/top20_gate_mass_ratio": top20_mass_ratio.detach(),
            "verified/top10_gate_mean": top10_gate_mean.detach(),
            "verified/top20_gate_mean": top20_gate_mean.detach(),
        }

    def _export_context_debug(self) -> None:
        from tim_2026.model._reference.modules.spatial_context_enrichment import (
            export_context_debug_figure,
            export_baseline_debug_figure,
        )

        images = getattr(self, "_context_debug_images", [])
        fmaps = getattr(self, "_context_debug_fmaps", [])

        if self.enable_context_enrichment:
            cache = getattr(self.context_enrichment, "_debug_cache", None)
            if not cache:
                return
        else:
            if not fmaps:
                return
            cache = None

        self._context_debug_count += 1
        ep_idx = self._context_debug_count

        if cache is not None:
            bw = None
            gv = None
            if self.context_enrichment.fusion_mode == "weighted_sum":
                bw = (
                    torch.softmax(self.context_enrichment.branch_weights.detach().float(), dim=0)
                    .cpu()
                    .numpy()
                )
                gv = float(self.context_enrichment.gate_value.detach().float().cpu().item())

            for call_idx, record in enumerate(cache):
                batch_size = record["original"].shape[0]
                flat_images = images[call_idx] if call_idx < len(images) else None
                for img_idx in range(min(batch_size, 8)):
                    inp = None
                    if flat_images is not None and img_idx < flat_images.shape[0]:
                        inp = flat_images[img_idx]
                    path = os.path.join(
                        self._context_debug_dir,
                        f"ep{ep_idx:03d}_call{call_idx}_img{img_idx:03d}.png",
                    )
                    try:
                        export_context_debug_figure(
                            debug_record={
                                "original": record["original"][img_idx : img_idx + 1],
                                "enriched": record["enriched"][img_idx : img_idx + 1],
                                "delta": record["delta"][img_idx : img_idx + 1],
                                "branches": [b[img_idx : img_idx + 1] for b in record["branches"]],
                            },
                            input_image=inp,
                            kernel_sizes=self.context_enrichment.kernel_sizes,
                            image_index=0,
                            save_path=path,
                            branch_weights=bw,
                            gate_value=gv,
                        )
                    except Exception as exc:
                        print(f"[context-debug] failed to export {path}: {exc}")
                        return
            print(
                f"[context-debug] episode {ep_idx}: exported {len(cache)} calls, "
                f"gate={gv:.4f}, dir={self._context_debug_dir}"
            )
        else:
            for call_idx, fmap in enumerate(fmaps):
                batch_size = fmap.shape[0]
                flat_images = images[call_idx] if call_idx < len(images) else None
                for img_idx in range(min(batch_size, 8)):
                    inp = None
                    if flat_images is not None and img_idx < flat_images.shape[0]:
                        inp = flat_images[img_idx]
                    path = os.path.join(
                        self._context_debug_dir,
                        f"ep{ep_idx:03d}_call{call_idx}_img{img_idx:03d}_baseline.png",
                    )
                    try:
                        export_baseline_debug_figure(
                            feature_map=fmap[img_idx : img_idx + 1],
                            input_image=inp,
                            save_path=path,
                        )
                    except Exception as exc:
                        print(f"[context-debug] failed to export {path}: {exc}")
                        return
            print(
                f"[context-debug] episode {ep_idx} (baseline): exported {len(fmaps)} calls, "
                f"dir={self._context_debug_dir}"
            )

    def _stack_outputs(self, batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = super()._stack_outputs(batch_outputs)
        for key in (
            "token_g_query",
            "token_g_support",
            "cost_modulator",
            "transport_plan_modulated",
            "marginal_query",
            "marginal_support",
            "cost_matrix_modulated",
            "mspta_guided_cost_matrix",
            "episode_contrast_guided_cost_matrix",
            "hcuot_guided_cost_matrix",
            "residual_aligned_guided_cost_matrix",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "pulse_query_saliency",
            "pulse_query_evidence",
            "pulse_discriminative_gate",
            "pulse_rival_advantage",
            "discriminative_uot_gate",
            "discriminative_uot_rival_advantage",
            "pulse_query_marginal_weight",
            "pulse_region_cost_matrix",
            "pulse_guided_cost_matrix",
            "region_uot_coarse_plan",
            "region_uot_sparse_coarse_plan",
            "region_uot_coarse_cost_matrix",
            "region_uot_guided_cost_matrix",
            "region_uot_coarse_transport_cost",
            "region_uot_coarse_transported_mass",
            "adaptive_region_query_masks",
            "adaptive_region_guided_cost_matrix",
            "adaptive_region_plan",
            "adaptive_region_cost_matrix",
            "adaptive_region_transport_cost",
            "adaptive_region_transported_mass",
            "adaptive_region_query_weight",
            "episode_contrast_pair_gate",
            "episode_contrast_query_weight",
            "ucot_pair_evidence",
            "ucot_query_weight",
            "ucot_support_weight",
            "ucot_query_specificity",
            "score_marginal_pair_evidence",
            "score_marginal_query_weight",
            "score_marginal_support_weight",
            "score_marginal_query_advantage",
            "ear_uot_query_reliability",
            "ear_uot_support_reliability",
            "ear_uot_query_tau",
            "ear_uot_support_tau",
            "ear_uot_rival_advantage",
            "ear_uot_reliable_mass",
            "ear_uot_low_reliability_mass",
            "ear_uot_high_reliability_mass_fraction",
            "ear_uot_low_reliability_destroyed_fraction",
            "ours_probe_negative_utility_mass_ratio",
            "ours_probe_common_mass_ratio",
            "ours_probe_harm_share",
            "ours_probe_dead_query_ratio",
            "ours_probe_query_mass_entropy",
            "ours_probe_effective_query_fraction",
            "ours_probe_utility_score",
            "ours_probe_mass_score",
            "ours_probe_cost_distance",
            "ours_probe_cost_only_score",
            "ours_probe_cost_per_mass_score",
            "ours_probe_threshold_x0_score",
            "ours_probe_threshold_x0p5_score",
            "ours_probe_threshold_x2_score",
            "ours_probe_threshold_x4_score",
            "rc_cost_class_probability",
            "rc_cost_penalty",
            "rc_cost_adjustment",
            "rc_cost_class_energy",
            "rc_cost_edge_advantage",
            "rc_cost_guided_cost_matrix",
            "hcuot_pair_specificity",
            "hcuot_query_weight",
            "hcuot_support_weight",
            "verified_region_pair_gate",
            "verified_region_cost_matrix",
            "verified_region_guided_cost_matrix",
            "residual_aligned_pair_evidence",
            "residual_aligned_query_weight",
            "rae_uot_pair_evidence",
            "rae_uot_rival_gate",
            "rae_uot_rival_specificity",
            "rae_uot_query_weight",
            "rae_uot_support_weight",
            "rvuot_verifier_gate",
            "rvuot_evidence_verifier_gate",
            "rvuot_evidence_transport_plan",
            "rvuot_reciprocal_affinity",
            "rvuot_unverified_transport_plan",
            "rvuot_unverified_shot_logits",
            "rvuot_unverified_shot_transport_cost",
            "rvuot_unverified_shot_transported_mass",
            "rvuot_shot_mass_for_score",
            "local_scores",
            "global_scores",
            "nr_ot_scores",
            "nr_ot_class_transport_plan",
            "nr_ot_class_cost_matrix",
            "nr_ot_ref_transport_plan",
            "nr_ot_ref_cost_matrix",
            "nr_ot_query_marginal",
            "nr_ot_support_marginal",
            "nr_ot_class_shot_transported_mass",
            "nr_ot_class_shot_expected_mass",
            "nr_ot_class_evidence",
            "nr_ot_ref_evidence",
            "nr_ot_debias_gap",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "pulse_support_saliency",
            "pulse_support_evidence",
            "pulse_support_marginal_weight",
            "adaptive_region_support_masks",
            "adaptive_region_support_weight",
            "residual_aligned_support_weight",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "marginal_query_l1_drift",
            "marginal_support_l1_drift",
            "marginal_l1_drift",
            "dmuot_shot_strength",
            "lambda_cost_effective",
            "pot_guide/alpha",
            "pot_guide/temperature",
            "pot_guide/s_mean",
            "pot_guide/pot_sparsity",
            "pot_guide/marginal_q_max",
            "pot_guide/marginal_q_min",
            "global_residual_weight",
            "global_residual_mode_id",
            "nr_ot_transport_cost_threshold",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        for key in (
            "dm_mu_norm",
            "dm_mu_max_norm",
            "dm_mu_support_ratio",
            "dm_query_residual_ratio",
            "dm_support_residual_ratio",
            "dm_residual_ratio",
            "dm_alpha",
            "dm_template_count",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "dm_mu_token_norm" in batch_outputs[0]:
            stacked["dm_mu_token_norm"] = torch.stack(
                [item["dm_mu_token_norm"] for item in batch_outputs],
                dim=0,
            )
        for key in sorted(k for k in batch_outputs[0] if str(k).startswith("multiscale/")):
            values = [item[key] for item in batch_outputs]
            if all(torch.is_tensor(value) for value in values):
                if values[0].dim() == 0:
                    stacked[key] = torch.stack(values).mean()
                else:
                    stacked[key] = torch.stack(values, dim=0).mean(dim=0)
        for prefix in (
            "context/",
            "sci/",
            "struct/",
            "pulse/",
            "discriminative_uot/",
            "label_ot/",
            "region_uot/",
            "adaptive_region/",
            "mspta/",
            "rvuot/",
            "episode_contrast/",
            "ucot/",
            "score_marginal/",
            "ours_probe/",
            "rc_cost/",
            "hcuot/",
            "verified/",
            "ear_uot/",
            "residual_aligned/",
            "rae_uot/",
            "nr_ot/",
            "transport_audit/",
            "transport_audit_unverified/",
        ):
            for key in sorted(k for k in batch_outputs[0] if str(k).startswith(prefix)):
                values = [item[key] for item in batch_outputs if key in item]
                if values and all(torch.is_tensor(value) for value in values):
                    stacked[key] = torch.stack(values).mean()
        return stacked


class OursFinalM2(OursM2):
    """Ours-Final entrypoint with EGSM removed from its architecture contract."""

    _REMOVED_EGSM_KWARGS = frozenset(
        {
            "ecot_egsm_hidden_dim",
            "ecot_egsm_candidate_tau_q",
            "ecot_egsm_candidate_tau_b",
            "ecot_egsm_kappa_min",
            "ecot_egsm_kappa_max",
            "ecot_egsm_rho_delta_max",
            "ecot_egsm_rho_grad_clip",
            "ecot_egsm_rho_reg_lambda",
        }
    )

    def __init__(self, *args, **kwargs) -> None:
        if normalize_ours_ablation(kwargs.get("ours_ablation", "full")) == "no_egsm":
            raise ValueError(
                "ours_ablation='no_egsm' is unavailable for Ours-Final because "
                "its token marginals are permanently uniform"
            )
        if _bool_config(kwargs.get("ecot_enable_egsm", False)):
            raise ValueError("EGSM has been removed from Ours-Final")
        if _bool_config(kwargs.get("ecot_egsm_adaptive_rho", False)):
            raise ValueError("EGSM adaptive rho has been removed from Ours-Final")
        removed = sorted(self._REMOVED_EGSM_KWARGS.intersection(kwargs))
        if removed:
            raise ValueError(
                "EGSM configuration has been removed from Ours-Final: "
                + ", ".join(removed)
            )
        kwargs["ecot_enable_egsm"] = False
        kwargs["ecot_egsm_adaptive_rho"] = False
        super().__init__(*args, **kwargs)


class OursFinalUCOT(OursFinalM2):
    """Standalone utility-calibrated Ours-Final with global residual and support-strict UOT."""

    _ABLATION_IDS = {
        "off": 0.0,
        "threshold_only": 1.0,
        "marginal_only": 2.0,
        "full": 3.0,
    }

    def __init__(self, *args, **kwargs) -> None:
        enable_ucot_calibration = _bool_config(kwargs.pop("enable_ucot_calibration", True))
        ucot_threshold_quantile = float(kwargs.pop("ucot_threshold_quantile", 0.70))
        ucot_threshold_mix = float(kwargs.pop("ucot_threshold_mix", 1.0))
        ucot_threshold_max_ratio = float(kwargs.pop("ucot_threshold_max_ratio", 2.0))
        ucot_threshold_detach = _bool_config(kwargs.pop("ucot_threshold_detach", True))
        ucot_temperature = float(kwargs.pop("ucot_temperature", 0.25))
        ucot_marginal_mix = float(kwargs.pop("ucot_marginal_mix", 0.65))
        ucot_adaptive_mix = _bool_config(kwargs.pop("ucot_adaptive_mix", True))
        ucot_confidence_power = float(kwargs.pop("ucot_confidence_power", 1.0))
        ucot_uniform_floor = float(kwargs.pop("ucot_uniform_floor", 0.05))
        ucot_rival_margin = float(kwargs.pop("ucot_rival_margin", 0.0))
        ucot_ablation = _normalize_ucot_ablation(kwargs.pop("ucot_ablation", "full"))

        if _normalize_ours_final_marginal_mode(kwargs.get("ours_final_marginal_mode", "uniform")) != "uniform":
            raise ValueError("ours_final_ucot owns its utility-calibrated marginals; do not use score_aligned")
        if str(kwargs.get("ours_final_score_mode", "threshold_mass")).strip().lower().replace("-", "_") != "threshold_mass":
            raise ValueError("ours_final_ucot requires ours_final_score_mode='threshold_mass'")
        kwargs["ours_final_marginal_mode"] = "uniform"
        kwargs["ours_final_score_mode"] = "threshold_mass"
        kwargs["enable_global_residual_score"] = True
        kwargs["global_residual_mode"] = "residual"
        kwargs["global_residual_weight"] = 0.1
        kwargs.setdefault("tau_q", 0.5)
        kwargs.setdefault("tau_c", 0.8)
        kwargs.setdefault("ecot_enable_egsm", False)
        kwargs.setdefault("ecot_m2_ablate_threshold_mass", False)
        kwargs.setdefault("ecot_m2_cost_per_mass_score", False)

        super().__init__(*args, **kwargs)

        if not 0.0 <= ucot_threshold_quantile <= 1.0:
            raise ValueError("ucot_threshold_quantile must be in [0, 1]")
        if not 0.0 <= ucot_threshold_mix <= 1.0:
            raise ValueError("ucot_threshold_mix must be in [0, 1]")
        if ucot_threshold_max_ratio < 1.0:
            raise ValueError("ucot_threshold_max_ratio must be at least 1")
        if ucot_temperature <= 0.0:
            raise ValueError("ucot_temperature must be positive")
        if not 0.0 <= ucot_marginal_mix <= 1.0:
            raise ValueError("ucot_marginal_mix must be in [0, 1]")
        if ucot_confidence_power <= 0.0:
            raise ValueError("ucot_confidence_power must be positive")
        if not 0.0 <= ucot_uniform_floor <= 1.0:
            raise ValueError("ucot_uniform_floor must be in [0, 1]")
        if ucot_rival_margin < 0.0:
            raise ValueError("ucot_rival_margin must be non-negative")

        self.enable_ucot_calibration = bool(enable_ucot_calibration)
        self.ucot_threshold_quantile = float(ucot_threshold_quantile)
        self.ucot_threshold_mix = float(ucot_threshold_mix)
        self.ucot_threshold_max_ratio = float(ucot_threshold_max_ratio)
        self.ucot_threshold_detach = bool(ucot_threshold_detach)
        self.ucot_temperature = float(ucot_temperature)
        self.ucot_marginal_mix = float(ucot_marginal_mix)
        self.ucot_adaptive_mix = bool(ucot_adaptive_mix)
        self.ucot_confidence_power = float(ucot_confidence_power)
        self.ucot_uniform_floor = float(ucot_uniform_floor)
        self.ucot_rival_margin = float(ucot_rival_margin)
        self.ucot_ablation = ucot_ablation

    def _ucot_normalize_with_uniform_fallback(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = raw.sum(dim=-1, keepdim=True)
        normalized = raw / total.clamp_min(self.eps)
        uniform = torch.full_like(raw, 1.0 / float(max(raw.shape[-1], 1)))
        fallback = total <= float(self.eps)
        return torch.where(fallback, uniform, normalized), fallback

    def _compute_ucot_threshold(
        self,
        flat_cost: torch.Tensor,
        way_num: int | None = None,
        shot_num: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (Nq, Way*Shot, Lq, Ls), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if way_num is None:
            way_num = 1
            shot_num = int(num_pairs)
        if shot_num is None:
            raise ValueError("shot_num must be provided when way_num is provided")
        expected_pairs = int(way_num) * int(shot_num)
        if num_pairs != expected_pairs:
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={expected_pairs}")

        threshold_source = flat_cost.detach() if self.ucot_threshold_detach else flat_cost
        cost_view = threshold_source.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
        )
        cost_scale = threshold_source.std(unbiased=False).clamp_min(self.eps)
        temperature = (float(self.ucot_temperature) * cost_scale).clamp_min(self.eps)
        class_token_cost = -temperature * (
            torch.logsumexp(-cost_view / temperature, dim=(2, 4))
            - math.log(float(max(int(shot_num) * support_len, 1)))
        )
        best_class_token_cost = class_token_cost.amin(dim=1)
        episode_threshold = torch.quantile(
            best_class_token_cost.flatten(),
            float(self.ucot_threshold_quantile),
        )
        base_threshold = self.transport_cost_threshold.to(
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if self.ucot_threshold_detach:
            base_threshold = base_threshold.detach()
        mix = flat_cost.new_tensor(float(self.ucot_threshold_mix))
        unclamped_threshold = (1.0 - mix) * base_threshold + mix * episode_threshold
        max_threshold = base_threshold * float(self.ucot_threshold_max_ratio)
        threshold = unclamped_threshold.clamp(
            min=base_threshold,
            max=max_threshold,
        ).clamp_min(self.eps)
        payload = {
            "ucot/base_threshold": base_threshold.detach(),
            "ucot/calibrated_threshold": threshold.detach(),
            "ucot/unclamped_threshold": unclamped_threshold.detach(),
            "ucot/threshold_ratio": (threshold.detach() / base_threshold.detach().clamp_min(self.eps)),
            "ucot/threshold_quantile": flat_cost.new_tensor(float(self.ucot_threshold_quantile)),
            "ucot/threshold_mix": flat_cost.new_tensor(float(self.ucot_threshold_mix)),
            "ucot/threshold_max_ratio": flat_cost.new_tensor(float(self.ucot_threshold_max_ratio)),
            "ucot/threshold_clipped": (
                (unclamped_threshold < base_threshold)
                | (unclamped_threshold > max_threshold)
            ).to(flat_cost.dtype).detach(),
            "ucot/threshold_source_mean": best_class_token_cost.mean().detach(),
            "ucot/threshold_source_std": best_class_token_cost.std(unbiased=False).detach(),
            "ucot/positive_edge_rate": (cost_view < threshold.detach()).to(flat_cost.dtype).mean().detach(),
        }
        return threshold, payload

    def _compute_ucot_marginals(
        self,
        flat_cost: torch.Tensor,
        threshold: torch.Tensor,
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
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={expected_pairs}")

        reference_cost = flat_cost.detach()
        cost_view = reference_cost.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
        )
        cost_scale = reference_cost.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
        temperature = (float(self.ucot_temperature) * cost_scale).clamp_min(self.eps)
        temperature_5d = temperature.reshape(num_query, 1, 1, 1, 1)
        threshold_value = torch.as_tensor(
            threshold,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if threshold_value.numel() != 1:
            raise ValueError("UCOT v1 requires a scalar threshold")

        utility = (threshold_value - cost_view) / temperature_5d
        positive = torch.sigmoid(utility)
        class_count = max(int(shot_num) * support_len, 1)
        class_token_cost = -temperature.reshape(num_query, 1, 1) * (
            torch.logsumexp(-cost_view / temperature_5d, dim=(2, 4))
            - math.log(float(class_count))
        )
        if int(way_num) > 1:
            rival_cost = torch.stack(
                [
                    torch.cat([class_token_cost[:, :idx], class_token_cost[:, idx + 1 :]], dim=1).amin(dim=1)
                    for idx in range(int(way_num))
                ],
                dim=1,
            )
            rival_advantage = rival_cost - class_token_cost
        else:
            rival_advantage = torch.zeros_like(class_token_cost)
        positive_rival_advantage = (
            rival_advantage - float(self.ucot_rival_margin)
        ).clamp_min(0.0)
        specificity = 1.0 - torch.exp(
            -positive_rival_advantage
            / temperature.reshape(num_query, 1, 1).clamp_min(self.eps)
        )
        query_positive = positive.amax(dim=(2, 4))
        query_specificity = specificity
        query_raw = query_specificity * query_positive
        query_selective, query_fallback = self._ucot_normalize_with_uniform_fallback(query_raw)

        k = max(1, query_len // 8)
        confidence = query_raw.topk(k, dim=-1).values.mean(dim=-1, keepdim=True).pow(
            float(self.ucot_confidence_power)
        )
        configured_mix = flat_cost.new_tensor(float(self.ucot_marginal_mix))
        effective_mix = (
            configured_mix * confidence
            if self.ucot_adaptive_mix
            else configured_mix.expand_as(confidence)
        ).clamp(0.0, 1.0)
        query_uniform = torch.full_like(query_selective, 1.0 / float(max(query_len, 1)))
        query_prob_class = (1.0 - effective_mix) * query_uniform + effective_mix * query_selective

        floor = flat_cost.new_tensor(float(self.ucot_uniform_floor))
        query_prob_class = (1.0 - floor) * query_prob_class + floor * query_uniform
        query_prob_class = query_prob_class / query_prob_class.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)
        query_prob = (
            query_prob_class[:, :, None, :]
            .expand(-1, -1, int(shot_num), -1)
            .reshape(num_query, num_pairs, query_len)
        )

        row_min = cost_view.amin(dim=-1, keepdim=True)
        row_affinity = torch.exp(-(cost_view - row_min) / temperature_5d).clamp(0.0, 1.0)
        pair_evidence = (specificity[:, :, None, :, None] * positive * row_affinity).clamp(0.0, 1.0)
        query_prob_view = query_prob.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
        )
        support_raw = (query_prob_view[..., None] * pair_evidence).sum(dim=-2)
        support_selective, support_fallback = self._ucot_normalize_with_uniform_fallback(support_raw)
        support_uniform = torch.full_like(support_selective, 1.0 / float(max(support_len, 1)))
        support_mix = effective_mix[:, :, None, :]
        support_prob = (1.0 - support_mix) * support_uniform + support_mix * support_selective
        support_prob = (1.0 - floor) * support_prob + floor * support_uniform
        support_prob = support_prob / support_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        support_prob = support_prob.reshape(num_query, num_pairs, support_len)

        query_entropy = -(query_prob_class * query_prob_class.clamp_min(self.eps).log()).sum(dim=-1)
        support_entropy = -(support_prob * support_prob.clamp_min(self.eps).log()).sum(dim=-1)
        query_l1_uniform = (query_prob_class - query_uniform).abs().sum(dim=-1)
        query_class_center = query_prob_class.mean(dim=1, keepdim=True)
        query_class_l1_spread = (
            query_prob_class - query_class_center
        ).abs().sum(dim=-1).mean()
        support_uniform_flat = support_uniform.reshape(num_query, num_pairs, support_len)
        support_l1_uniform = (support_prob - support_uniform_flat).abs().sum(dim=-1)
        payload = {
            "ucot_query_weight": query_prob.detach(),
            "ucot_support_weight": support_prob.detach(),
            "ucot_pair_evidence": pair_evidence.detach(),
            "ucot_query_specificity": specificity.detach(),
            "ucot/enabled": flat_cost.new_tensor(1.0),
            "ucot/ablation_id": flat_cost.new_tensor(self._ABLATION_IDS[self.ucot_ablation]),
            "ucot/temperature": flat_cost.new_tensor(float(self.ucot_temperature)),
            "ucot/marginal_mix": configured_mix.detach(),
            "ucot/adaptive_mix": flat_cost.new_tensor(float(self.ucot_adaptive_mix)),
            "ucot/confidence_power": flat_cost.new_tensor(float(self.ucot_confidence_power)),
            "ucot/uniform_floor": floor.detach(),
            "ucot/rival_margin": flat_cost.new_tensor(float(self.ucot_rival_margin)),
            "ucot/class_conditional_query": flat_cost.new_tensor(1.0),
            "ucot/query_positive_acceptance_mean": query_positive.mean().detach(),
            "ucot/edge_positive_acceptance_mean": positive.mean().detach(),
            "ucot/positive_rival_advantage_share": (
                positive_rival_advantage > self.eps
            ).to(flat_cost.dtype).mean().detach(),
            "ucot/query_specificity_mean": specificity.mean().detach(),
            "ucot/query_specificity_peak": query_specificity.amax(dim=1).mean().detach(),
            "ucot/common_query_share": (
                query_specificity <= self.eps
            ).to(flat_cost.dtype).mean().detach(),
            "ucot/discriminative_query_share": (
                query_specificity > self.eps
            ).to(flat_cost.dtype).mean().detach(),
            "ucot/query_raw_mean": query_raw.mean().detach(),
            "ucot/query_class_l1_spread": query_class_l1_spread.detach(),
            "ucot/evidence_confidence": confidence.mean().detach(),
            "ucot/effective_mix": effective_mix.mean().detach(),
            "ucot/query_entropy": query_entropy.mean().detach(),
            "ucot/support_entropy": support_entropy.mean().detach(),
            "ucot/query_peak": query_prob_class.amax(dim=-1).mean().detach(),
            "ucot/support_peak": support_prob.amax(dim=-1).mean().detach(),
            "ucot/query_l1_from_uniform": query_l1_uniform.mean().detach(),
            "ucot/support_l1_from_uniform": support_l1_uniform.mean().detach(),
            "ucot/uniform_fallback_query_share": query_fallback.to(flat_cost.dtype).mean().detach(),
            "ucot/uniform_fallback_support_share": support_fallback.to(flat_cost.dtype).mean().detach(),
        }
        return query_prob, support_prob, payload

    def _apply_ucot_local_flow(
        self,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        if not self.enable_ucot_calibration or self.ucot_ablation == "off":
            base_threshold = self.transport_cost_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
            return None, None, None, {
                "ucot/enabled": flat_cost.new_tensor(0.0),
                "ucot/ablation_id": flat_cost.new_tensor(self._ABLATION_IDS["off"]),
                "ucot/threshold_active": flat_cost.new_tensor(0.0),
                "ucot/marginal_active": flat_cost.new_tensor(0.0),
                "ucot/base_threshold": base_threshold.detach(),
                "ucot/calibrated_threshold": base_threshold.detach(),
                "ucot/unclamped_threshold": base_threshold.detach(),
                "ucot/threshold_ratio": flat_cost.new_tensor(1.0),
                "ucot/threshold_quantile": flat_cost.new_tensor(float(self.ucot_threshold_quantile)),
                "ucot/threshold_max_ratio": flat_cost.new_tensor(float(self.ucot_threshold_max_ratio)),
                "ucot/threshold_clipped": flat_cost.new_tensor(0.0),
            }

        if self.ucot_ablation == "marginal_only":
            threshold = self.transport_cost_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype).detach()
            threshold_payload = {
                "ucot/base_threshold": threshold.detach(),
                "ucot/calibrated_threshold": threshold.detach(),
                "ucot/unclamped_threshold": threshold.detach(),
                "ucot/threshold_ratio": flat_cost.new_tensor(1.0),
                "ucot/threshold_quantile": flat_cost.new_tensor(float(self.ucot_threshold_quantile)),
                "ucot/threshold_mix": flat_cost.new_tensor(0.0),
                "ucot/threshold_max_ratio": flat_cost.new_tensor(float(self.ucot_threshold_max_ratio)),
                "ucot/threshold_clipped": flat_cost.new_tensor(0.0),
                "ucot/threshold_active": flat_cost.new_tensor(0.0),
                "ucot/positive_edge_rate": (flat_cost.detach() < threshold).to(flat_cost.dtype).mean().detach(),
            }
            threshold_override = None
        else:
            threshold, threshold_payload = self._compute_ucot_threshold(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
            )
            threshold_payload["ucot/threshold_active"] = flat_cost.new_tensor(1.0)
            threshold_override = threshold

        query_weight = None
        support_weight = None
        marginal_payload: dict[str, torch.Tensor] = {}
        if self.ucot_ablation in {"full", "marginal_only"}:
            query_weight, support_weight, marginal_payload = self._compute_ucot_marginals(
                flat_cost,
                threshold,
                way_num,
                shot_num,
            )
        elif self.ucot_ablation == "threshold_only":
            query_len = flat_cost.shape[-2]
            support_len = flat_cost.shape[-1]
            query_uniform = flat_cost.new_full((flat_cost.shape[0], query_len), 1.0 / float(max(query_len, 1)))
            support_uniform = flat_cost.new_full(
                (flat_cost.shape[0], flat_cost.shape[1], support_len),
                1.0 / float(max(support_len, 1)),
            )
            _, _, marginal_payload = self._compute_ucot_marginals(flat_cost, threshold, way_num, shot_num)
            marginal_payload["ucot_query_weight"] = query_uniform.detach()
            marginal_payload["ucot_support_weight"] = support_uniform.detach()
            marginal_payload["ucot/query_entropy"] = (
                -(query_uniform * query_uniform.clamp_min(self.eps).log()).sum(dim=-1).mean().detach()
            )
            marginal_payload["ucot/support_entropy"] = (
                -(support_uniform * support_uniform.clamp_min(self.eps).log()).sum(dim=-1).mean().detach()
            )
            marginal_payload["ucot/query_peak"] = query_uniform.amax(dim=-1).mean().detach()
            marginal_payload["ucot/support_peak"] = support_uniform.amax(dim=-1).mean().detach()
            marginal_payload["ucot/query_l1_from_uniform"] = flat_cost.new_tensor(0.0)
            marginal_payload["ucot/support_l1_from_uniform"] = flat_cost.new_tensor(0.0)
            marginal_payload["ucot/class_conditional_query"] = flat_cost.new_tensor(0.0)
            marginal_payload["ucot/query_class_l1_spread"] = flat_cost.new_tensor(0.0)
            marginal_payload["ucot/effective_mix"] = flat_cost.new_tensor(0.0)

        payload = {
            **threshold_payload,
            **marginal_payload,
            "ucot/enabled": flat_cost.new_tensor(1.0),
            "ucot/ablation_id": flat_cost.new_tensor(self._ABLATION_IDS[self.ucot_ablation]),
            "ucot/marginal_active": flat_cost.new_tensor(
                float(self.ucot_ablation in {"full", "marginal_only"})
            ),
        }
        return query_weight, support_weight, threshold_override, payload

    def _add_ucot_transport_diagnostics(
        self,
        payload: dict[str, torch.Tensor],
        ucot_payload: dict[str, torch.Tensor],
        *,
        way_num: int,
        shot_num: int,
    ) -> None:
        plan = payload.get("transport_plan")
        shot_rho = payload.get("shot_rho")
        query_weight = ucot_payload.get("ucot_query_weight")
        support_weight = ucot_payload.get("ucot_support_weight")
        if not (torch.is_tensor(plan) and torch.is_tensor(shot_rho)):
            return
        transported_fraction = payload["shot_transported_mass"] / shot_rho.to(
            device=plan.device,
            dtype=plan.dtype,
        ).clamp_min(self.eps)
        ucot_payload["ucot/transported_mass_fraction"] = transported_fraction.mean().detach()
        if not (torch.is_tensor(query_weight) and torch.is_tensor(support_weight)):
            return
        query_prob = query_weight.to(device=plan.device, dtype=plan.dtype)
        if query_prob.dim() == 2:
            query_prob = query_prob[:, None, None, :].expand(
                -1,
                int(way_num),
                int(shot_num),
                -1,
            )
        elif query_prob.dim() == 3:
            query_prob = query_prob.reshape(
                plan.shape[0],
                int(way_num),
                int(shot_num),
                plan.shape[-2],
            )
        else:
            raise ValueError(
                "UCOT query weights must be shared or pair-conditional, "
                f"got {tuple(query_prob.shape)}"
            )
        support_prob = support_weight.to(device=plan.device, dtype=plan.dtype).reshape(
            plan.shape[0],
            int(way_num),
            int(shot_num),
            plan.shape[-1],
        )
        shot_rho = shot_rho.to(device=plan.device, dtype=plan.dtype)
        target_query = query_prob * shot_rho[..., None]
        target_support = support_prob * shot_rho[..., None]
        query_l1 = (plan.sum(dim=-1) - target_query).abs().sum(dim=-1)
        support_l1 = (plan.sum(dim=-2) - target_support).abs().sum(dim=-1)
        ucot_payload["ucot/query_l1_drift"] = query_l1.mean().detach()
        ucot_payload["ucot/support_l1_drift"] = support_l1.mean().detach()

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
    ) -> dict[str, torch.Tensor]:
        if query_weight is not None or support_weight is not None:
            raise ValueError("ours_final_ucot owns query/support marginals and cannot combine explicit weights")
        query_weight, support_weight, threshold_override, ucot_payload = self._apply_ucot_local_flow(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
        )
        payload = super()._forward_ecot_budget_bank(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            support_weight=support_weight,
            query_weight=query_weight,
            support_tokens=support_tokens,
            query_tokens=query_tokens,
            spatial_hw=spatial_hw,
            threshold_override=threshold_override,
        )
        self._add_ucot_transport_diagnostics(
            payload,
            ucot_payload,
            way_num=way_num,
            shot_num=shot_num,
        )
        payload.update(ucot_payload)
        return payload


__all__ = [
    "OursM2",
    "OursFinalM2",
    "OursFinalUCOT",
    "apply_ours_design_defaults",
    "OURS_ABLATIONS",
    "normalize_ours_ablation",
]
