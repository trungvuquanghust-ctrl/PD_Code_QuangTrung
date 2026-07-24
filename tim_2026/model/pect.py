"""Paper-facing PECT model builder.

The numerical transport implementation is preserved in ``_reference``.  This
module is intentionally small: it maps one typed configuration to the exact
constructor arguments used by the original ``ours_final`` entry point.
"""

from __future__ import annotations

from tim_2026.config import ModelConfig
from tim_2026.model._reference.ours import OursFinalM2


class PECT(OursFinalM2):
    """Canonical PECT architecture (ResNet12 + local UOT + global residual)."""


def build_pect(config: ModelConfig) -> PECT:
    config.validate()

    return PECT(
        in_channels=3,
        hidden_dim=config.hidden_dim,
        token_dim=config.token_dim,
        use_raw_backbone_tokens=config.use_raw_backbone_tokens,
        backbone_name=config.backbone,
        image_size=config.image_size,
        resnet12_drop_rate=0.0,
        resnet12_dropblock_size=5,
        variant="J_ECOT_M2",
        ours_ablation=config.ours_ablation,
        ours_final_score_mode=config.score_mode,
        tau_q=config.tau_q,
        tau_c=config.tau_c,
        fixed_mass=config.fixed_mass,
        ecot_rho_bank=config.rho_bank,
        ecot_base_rho=config.rho,
        ecot_budget_prior_init=config.budget_prior_init,
        ecot_lambda_init=config.budget_lambda_init,
        ecot_identity_reg=config.budget_identity_reg,
        ecot_transport_mode=config.transport_mode,
        ecot_enable_tau_shot=config.enable_tau_shot,
        ecot_m2_ablate_threshold_mass=config.ablate_threshold_mass,
        ecot_m2_cost_per_mass_score=config.cost_per_mass_score,
        ecot_m2_cost_per_mass_detach_mass=True,
        pre_transport_shot_pool=config.pre_transport_shot_pool,
        pre_transport_shot_pool_mode=config.pre_transport_shot_pool_mode,
        enable_global_residual_score=config.enable_global_residual,
        global_residual_mode=config.global_residual_mode,
        global_residual_weight=config.global_residual_weight,
        ot_backend=config.ot_backend,
        enable_ours_final_failure_probe=False,
        hlm_return_diagnostics=False,
    )
