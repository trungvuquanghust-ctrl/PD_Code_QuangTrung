"""PECT ablations copied from ``build_ours_final_pect_variants``.

Only the intended factor changes between variants; the training protocol is
defined once in :mod:`tim_2026.config`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    description: str
    overrides: dict[str, object]


PECT_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "pect_no_global",
        "PECT without the global residual head",
        {"enable_global_residual": False},
    ),
    *tuple(
        AblationSpec(
            f"pect_global_res_w{str(weight).replace('.', 'p')}",
            f"PECT global residual weight={weight}",
            {"global_residual_weight": weight},
        )
        for weight in (0.05, 0.1, 0.15, 0.2)
    ),
    AblationSpec(
        "pect_global_only",
        "Global prototype classifier without local UOT logits",
        {"global_residual_mode": "global_only", "global_residual_weight": 1.0},
    ),
    *tuple(
        AblationSpec(
            f"pect_rho_{str(rho).replace('.', 'p')}",
            f"PECT with UOT rho={rho}",
            {"rho": rho, "rho_bank": str(rho)},
        )
        for rho in (0.6, 0.7, 0.9)
    ),
    AblationSpec(
        "pect_latent_rho",
        "Episode-level posterior over rho={0.5,0.6,0.7,0.8,0.9}",
        {
            "enable_latent_rho": True,
            "rho_bank": "0.5,0.6,0.7,0.8,0.9",
            "budget_prior_init": "uniform",
            "budget_lambda_init": 4.0,
            "budget_identity_reg": 0.0,
        },
    ),
    AblationSpec(
        "pect_full_ot",
        "Balanced full OT control",
        {
            "ours_ablation": "full_ot",
            "rho": 1.0,
            "rho_bank": "1.0",
            "transport_mode": "balanced",
        },
    ),
    AblationSpec(
        "pect_partial_ot",
        "Fast Partial OT with cost-per-mass scoring",
        {"ot_backend": "partial", "cost_per_mass_score": True},
    ),
    AblationSpec(
        "pect_class_pooled",
        "Class-pooled support tokens before transport",
        {
            "pre_transport_shot_pool": True,
            "pre_transport_shot_pool_mode": "concat",
        },
    ),
    AblationSpec(
        "pect_cost_only",
        "Cost-only score without threshold-mass reward",
        {"ablate_threshold_mass": True},
    ),
)


def get_ablation(name: str) -> AblationSpec:
    for spec in PECT_ABLATIONS:
        if spec.name == name:
            return spec
    valid = ", ".join(spec.name for spec in PECT_ABLATIONS)
    raise KeyError(f"Unknown ablation {name!r}. Available: {valid}")


def apply_ablation(base: ModelConfig, name: str) -> ModelConfig:
    config = deepcopy(base)
    for key, value in get_ablation(name).overrides.items():
        setattr(config, key, value)
    config.validate()
    return config
