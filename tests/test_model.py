from tim_2026.config import ModelConfig
from tim_2026.model import build_pect


def test_canonical_model_signature() -> None:
    model = build_pect(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_608_390
    assert len(model.state_dict()) == 121
    assert model.enable_global_residual_score is True
    assert model.global_residual_weight == 0.1
    assert tuple(model.ecot_rho_bank) == (0.8,)
    assert model.ecot_m2_cost_per_mass_detach_mass is True
    assert model.enable_ours_final_failure_probe is False
