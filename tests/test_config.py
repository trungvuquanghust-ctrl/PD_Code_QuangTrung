from pathlib import Path

from tim_2026.config import ExperimentConfig, ModelConfig


def test_canonical_pect_config_is_locked() -> None:
    model = ModelConfig()
    assert model.backbone == "resnet12"
    assert model.image_size == 84
    assert model.token_dim == 128
    assert model.rho == 0.8
    assert model.tau_q == model.tau_c == 0.5
    assert model.transport_mode == "unbalanced"
    assert model.score_mode == "threshold_mass"
    assert model.enable_global_residual
    assert model.global_residual_mode == "residual"
    assert model.global_residual_weight == 0.1


def test_paper_protocol_defaults() -> None:
    config = ExperimentConfig(dataset_path=Path("data"))
    assert (config.train_episodes, config.val_episodes, config.test_episodes) == (130, 150, 150)
    assert config.learning_rate == 5e-4
    assert config.weight_decay == 5e-4
    assert config.warmup_epochs == 5
    assert config.runtime.final_test_seed == 200042
