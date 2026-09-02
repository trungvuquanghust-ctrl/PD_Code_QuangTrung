"""Command-line parsing shared by the canonical and ablation entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ablations import PECT_ABLATIONS, apply_ablation
from .config import ExperimentConfig, ModelConfig, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or test the paper-facing PECT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the external train/val/test dataset root",
    )
    parser.add_argument("--dataset-name", "--dataset_name", default="knee_aug_split")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-name", "--run_name", default=None)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--weights", type=Path)

    parser.add_argument("--shot", "--shot-num", "--shot_num", type=int, choices=(1, 5), default=1)
    parser.add_argument("--training-samples", "--training_samples", type=int)
    parser.add_argument("--epochs", "--num-epochs", "--num_epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--train-episodes", "--episode_num_train", type=int, default=130)
    parser.add_argument("--val-episodes", "--episode_num_val", type=int, default=150)
    parser.add_argument("--test-episodes", "--episode_num_test", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", type=int, default=5)
    parser.add_argument("--min-lr", "--min_lr", type=float, default=1e-6)

    # --- Cac flag nay chua ton tai trong ban goc, them vao vi ExperimentConfig
    # (config.py) da co san field train_augment / label_smoothing / grad_clip
    # nhung cli.py chua truyen duoc gia tri cho chung.
    parser.add_argument(
        "--train-augment",
        "--train_augment",
        action="store_true",
        help="Enable episodic data augmentation during training",
    )
    parser.add_argument(
        "--label-smoothing",
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing factor",
    )
    parser.add_argument(
        "--grad-clip",
        "--grad_clip",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0 disables clipping)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-subset-seed",
        "--training_subset_seed",
        type=int,
        default=None,
        help="Seed for picking the --training-samples subset, independent of "
             "--seed (model init/training). Defaults to --seed if unset.",
    )
    parser.add_argument("--final-test-seed", "--final_test_seed", type=int, default=200042)
    parser.add_argument("--gpu", "--gpu-id", "--gpu_id", type=int, default=0)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")

    parser.add_argument(
        "--variant",
        choices=tuple(spec.name for spec in PECT_ABLATIONS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-confusion-matrix", action="store_true")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    model = ModelConfig()
    if args.variant:
        model = apply_ablation(model, args.variant)

    samples_name = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    default_name = args.variant or "pect"
    run_name = args.run_name or f"{default_name}_{args.dataset_name}_{samples_name}_{args.shot}shot_seed{args.seed}"
    return ExperimentConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        run_name=run_name,
        mode=args.mode,
        weights=args.weights,
        shot_num=args.shot,
        training_samples=args.training_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate=args.min_lr,
        train_augment=args.train_augment,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        save_confusion_matrix=not args.no_confusion_matrix,
        save_tsne=not args.no_tsne,
        model=model,
        runtime=RuntimeConfig(
            seed=args.seed,
            training_subset_seed=args.training_subset_seed,
            final_test_seed=args.final_test_seed,
            gpu_id=args.gpu,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            persistent_workers=not args.no_persistent_workers,
        ),
    )
