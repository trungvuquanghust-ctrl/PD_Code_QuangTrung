"""Standalone J-ECOT-M2/SB-ECOT model wrapper.

M2 is exposed as a first-class model instead of a J-ECOT ablation flag.  The
model keeps the HROTFSL implementation backend, but fixes the architectural
identity to the single-budget episode-calibrated transport module:
shot-decomposed local measures, one transport budget, default ECOT score
without the T * mass term (-transport cost), optional cost-per-mass scoring,
and episode-calibrated support-shot pooling.
"""

from __future__ import annotations

from tim_2026.model._reference.hrot_fsl import HROTFSL


class JECOTM2(HROTFSL):
    """Single-budget episode-calibrated OT model for few-shot scalograms."""

    def __init__(
        self,
        *args,
        transport_mode: str = "unbalanced",
        rho: float = 0.80,
        **kwargs,
    ) -> None:
        kwargs["variant"] = "J_ECOT_M2"
        kwargs.setdefault("ecot_rho_bank", f"{float(rho):.6g}")
        kwargs.setdefault("ecot_base_rho", float(rho))
        kwargs.setdefault("ecot_transport_mode", transport_mode)
        kwargs.setdefault("ecot_enable_ccdm_marginal", True)
        kwargs.setdefault("ecot_enable_egsm", False)
        kwargs.setdefault("ecot_m2_ablate_threshold_mass", True)
        kwargs.setdefault("ecot_m2_cost_per_mass_score", False)
        kwargs.setdefault("ecot_m2_cost_per_mass_detach_mass", False)
        super().__init__(*args, **kwargs)
