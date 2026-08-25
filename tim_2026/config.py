"""Typed experiment configuration with Mamba backbone integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """PECT architecture parameters."""

    image_size: int = 224
    backbone: str = "vision_mamba"
    hidden_dim: int = 640
    token_dim: int = 128
    use_raw_backbone_tokens: bool = False

    rho: float = 0.8
    rho_bank: str = "0.8"
    tau_q: float = 0.5
    tau_c: float = 0.5
    fixed_mass: float = 0.8
    transport_mode: str = "unbalanced"
    ot_backend: str = "native"
    score_mode: str = "threshold_mass"
    ablate_threshold_mass: bool = False
    cost_per_mass_score: bool = False

    pre_transport_shot_pool: bool = False
    pre_transport_shot_pool_mode: str = "mean"
    enable_tau_shot: bool = True

    enable_global_residual: bool = True
    global_residual_mode: str = "residual"
    global_residual_weight: float = 0.1

    enable_latent_rho: bool = False
    budget_prior_init: str = "base"
    budget_lambda_init: float = -8.0
    budget_identity_reg: float = 1e-4

    ours_ablation: str = "full"

    def validate(self) -> None:
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        if self.global_residual_weight < 0.0:
            raise ValueError("global_residual_weight must be non-negative")


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    final_test_seed: int = 200042
    selection_base_seed: int = 42
    selection_seed_offset: int = 100000
    gpu_id: int = 0
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    deterministic: bool = True
    cudnn_benchmark: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    dataset_path: Path
    dataset_name: str = "knee_aug_split"
    output_dir: Path = Path("artifacts")
    run_name: str = "pect"
    mode: str = "train"
    weights: Path | None = None

    way_num: int = 4
    shot_num: int = 1
    query_num_train: int = 1
    query_num_val: int = 1
    query_num_test: int = 1
    training_samples: int | None = None

    num_epochs: int = 100
    batch_size: int = 1
    train_episodes: int = 130
    val_episodes: int = 150
    test_episodes: int = 150

    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_learning_rate: float = 1e-6
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    train_augment: bool = False

    save_confusion_matrix: bool = True
    save_tsne: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'")
        if self.mode == "test" and self.weights is None:
            raise ValueError("--weights is required in test mode")
        if self.shot_num not in {1, 5}:
            raise ValueError("shot_num must be 1 or 5")
        if self.way_num != 4:
            raise ValueError("TIM_2026 reproduces the original 4-way protocol")
        if self.training_samples is not None and self.training_samples % self.way_num != 0:
            raise ValueError("training_samples must be divisible by way_num")
        self.model.validate()

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        payload["weights"] = None if self.weights is None else str(self.weights)
        return payload
