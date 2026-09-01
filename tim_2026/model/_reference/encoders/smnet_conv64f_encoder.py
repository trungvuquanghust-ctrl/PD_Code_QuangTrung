"""Compact train/validation/test pipeline for PECT."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tim_2026.config import ExperimentConfig
from tim_2026.data import DatasetBundle, FewshotDataset, load_dataset_bundle
from tim_2026.logging import append_summary, write_csv, write_key_values
from tim_2026.model import build_pect
from tim_2026.runtime import configure_runtime
from tim_2026.visualization import save_confusion_matrix, save_tsne


def _seed_worker(worker_id: int, base_seed: int) -> None:
    import random

    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(
    dataset: FewshotDataset,
    config: ExperimentConfig,
    device: torch.device,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    runtime = config.runtime
    workers = max(0, runtime.num_workers)
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": runtime.pin_memory and device.type == "cuda",
    }
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    if workers > 0:
        kwargs["persistent_workers"] = runtime.persistent_workers
        if runtime.prefetch_factor > 0:
            kwargs["prefetch_factor"] = runtime.prefetch_factor
        kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=seed)
    return DataLoader(dataset, **kwargs)


def _episode_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    episodes: int,
    query_num: int,
    seed: int,
    config: ExperimentConfig,
) -> FewshotDataset:
    return FewshotDataset(
        images,
        labels,
        episodes,
        config.way_num,
        config.shot_num,
        query_num,
        seed=seed,
        augment=config.train_augment,
    )


def _prepare_batch(
    batch: tuple[torch.Tensor, ...],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query, query_labels, support, support_labels = batch[:4]
    batch_size, _, channels, height, width = query.shape
    support = support.view(
        batch_size,
        config.way_num,
        config.shot_num,
        channels,
        height,
        width,
    )
    non_blocking = config.runtime.pin_memory and device.type == "cuda"
    return (
        query.to(device, non_blocking=non_blocking),
        query_labels.view(-1).to(device, non_blocking=non_blocking),
        support.to(device, non_blocking=non_blocking),
        support_labels.view(batch_size, config.way_num, config.shot_num).to(
            device,
            non_blocking=non_blocking,
        ),
    )


def _forward_loss(
    model: torch.nn.Module,
    query: torch.Tensor,
    targets: torch.Tensor,
    support: torch.Tensor,
    support_targets: torch.Tensor,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        query,
        support,
        query_targets=targets,
        support_targets=support_targets,
        return_aux=False,
    )
    if isinstance(output, dict):
        logits = output["logits"]
        auxiliary = output.get("aux_loss")
    else:
        logits = output
        auxiliary = None
    loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    if auxiliary is not None:
        loss = loss + auxiliary
    return logits, loss


def _build_scheduler(optimizer: AdamW, config: ExperimentConfig):
    if config.warmup_epochs > 0 and config.num_epochs > config.warmup_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.num_epochs - config.warmup_epochs),
            eta_min=config.min_learning_rate,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs],
        )
    return CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.num_epochs),
        eta_min=config.min_learning_rate,
    )


# Substrings that identify Mamba SSM state parameters (mamba_ssm's real
# parameter names: "A_log", "D", "dt_proj.weight", "dt_proj.bias"). These,
# together with every 1-D parameter (biases, norm weights), should not get
# weight decay -- decaying the SSM state/discretization parameters tends to
# destabilize training.
_NO_DECAY_NAME_HINTS = ("a_log", "dt_proj", ".d", "_d_")


def _is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    if param.ndim <= 1:
        return True
    lowered = name.lower()
    return any(hint in lowered for hint in _NO_DECAY_NAME_HINTS)


def _build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_param(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=config.learning_rate)


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    losses: list[float] = []
    for batch in loader:
        query, targets, support, support_targets = _prepare_batch(batch, config, device)
        logits, loss = _forward_loss(
            model,
            query,
            targets,
            support,
            support_targets,
            config.label_smoothing,
        )
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.numel())
        losses.append(float(loss.item()))
    return correct / max(total, 1), float(np.mean(losses)) if losses else float("nan")


def _train(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[Path, list[dict[str, object]]]:
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    history: list[dict[str, object]] = []
    best_accuracy = -math.inf
    checkpoint_path = config.run_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.num_epochs + 1):
        train_seed = config.runtime.seed + epoch
        val_seed = (
            config.runtime.selection_seed_offset
            + config.runtime.selection_base_seed
        )  # co dinh qua moi epoch de loai nhieu do doi episode validation
        train_dataset = _episode_dataset(
            bundle.train_images,
            bundle.train_labels,
            episodes=config.train_episodes,
            query_num=config.query_num_train,
            seed=train_seed,
            config=config,
        )
        train_loader = _loader(
            train_dataset,
            config,
            device,
            batch_size=config.batch_size,
            shuffle=True,
            seed=train_seed,
        )

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.num_epochs}")
        for batch in progress:
            query, targets, support, support_targets = _prepare_batch(batch, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = _forward_loss(
                model,
                query,
                targets,
                support,
                support_targets,
                config.label_smoothing,
            )
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_accuracy = correct / max(total, 1)
        train_loss = epoch_loss / max(len(train_loader), 1)

        val_dataset = _episode_dataset(
            bundle.val_images,
            bundle.val_labels,
            episodes=config.val_episodes,
            query_num=config.query_num_val,
            seed=val_seed,
            config=config,
        )
        val_loader = _loader(
            val_dataset,
            config,
            device,
            batch_size=1,
            shuffle=False,
            seed=val_seed,
        )
        val_accuracy, val_loss = _evaluate_loader(model, val_loader, config, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        write_csv(config.run_dir / "history.csv", history)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")

    return checkpoint_path, history


def _load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(state)


def _extract_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            feature_map = model.encode(images[start : start + batch_size].to(device))
            if feature_map.dim() == 4:
                feature_map = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            elif feature_map.dim() > 2:
                feature_map = feature_map.flatten(1)
            feature_map = F.normalize(feature_map, p=2, dim=-1)
            features.append(feature_map.cpu().numpy())
    return np.vstack(features)


@torch.no_grad()
def _test(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, object]:
    test_dataset = _episode_dataset(
        bundle.test_images,
        bundle.test_labels,
        episodes=config.test_episodes,
        query_num=config.query_num_test,
        seed=config.runtime.final_test_seed,
        config=config,
    )
    test_loader = _loader(
        test_dataset,
        config,
        device,
        batch_size=1,
        shuffle=False,
        seed=config.runtime.final_test_seed,
    )

    model.eval()
    predictions: list[int] = []
    targets_all: list[int] = []
    episode_accuracies: list[float] = []
    episode_times_ms: list[float] = []
    losses: list[float] = []

    for batch in tqdm(test_loader, desc="Final test"):
        query, targets, support, support_targets = _prepare_batch(batch, config, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, loss = _forward_loss(
            model,
            query,
            targets,
            support,
            support_targets,
            config.label_smoothing,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        episode_times_ms.append((time.perf_counter() - start) * 1000.0)

        predicted = logits.argmax(dim=1)
        predictions.extend(predicted.cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
        episode_accuracies.append(float((predicted == targets).float().mean().item()))
        losses.append(float(loss.item()))

    predictions_np = np.asarray(predictions)
    targets_np = np.asarray(targets_all)
    episode_np = np.asarray(episode_accuracies, dtype=np.float64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np,
        predictions_np,
        average="macro",
        zero_division=0,
    )
    accuracy_mean = float(episode_np.mean())
    accuracy_std = float(episode_np.std())
    metrics: dict[str, object] = {
        "run_name": config.run_name,
        "dataset": config.dataset_name,
        "shot": config.shot_num,
        "training_samples": config.training_samples or "all",
        "training_seed": config.runtime.seed,
        "final_test_seed": config.runtime.final_test_seed,
        "test_episodes": config.test_episodes,
        "test_loss": float(np.mean(losses)),
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "accuracy_ci95": float(1.96 * accuracy_std / math.sqrt(max(len(episode_np), 1))),
        "query_accuracy": float((predictions_np == targets_np).mean()),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "inference_ms_mean": float(np.mean(episode_times_ms)),
        "inference_ms_std": float(np.std(episode_times_ms)),
    }

    if config.save_confusion_matrix:
        save_confusion_matrix(
            targets_np,
            predictions_np,
            bundle.class_names,
            config.run_dir / "confusion_matrix",
        )
    if config.save_tsne:
        features = _extract_features(model, bundle.test_images, device)
        save_tsne(
            features,
            bundle.test_labels.numpy(),
            bundle.class_names,
            config.run_dir / "tsne",
            seed=config.runtime.seed,
        )
    return metrics


def _flatten_config(config: ExperimentConfig) -> dict[str, object]:
    payload = {
        key: value
        for key, value in asdict(config).items()
        if key not in {"model", "runtime"}
    }
    payload.update({f"model.{key}": value for key, value in asdict(config.model).items()})
    payload.update({f"runtime.{key}": value for key, value in asdict(config.runtime).items()})
    return payload


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    config.validate()
    device = configure_runtime(config.runtime)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_key_values(config.run_dir / "config.txt", _flatten_config(config))

    print(f"Run      : {config.run_name}")
    print(f"Dataset  : {config.dataset_path}")
    print(f"Protocol : {config.way_num}-way {config.shot_num}-shot, device={device}")
    print(
        "PECT     : UOT rho="
        f"{config.model.rho}, tau=({config.model.tau_q},{config.model.tau_c}), "
        f"global={config.model.global_residual_mode}@{config.model.global_residual_weight}"
    )

    # Cho phep tach rieng seed chon subset anh khoi seed huan luyen model,
    # de do dung "do on dinh" khi chay nhieu seed (khong lan giua 2 nguon
    # bien thien: data-sampling vs training/optimization).
    subset_seed = (
        config.runtime.training_subset_seed
        if config.runtime.training_subset_seed is not None
        else config.runtime.seed
    )
    bundle = load_dataset_bundle(
        config.dataset_path,
        image_size=config.model.image_size,
        training_samples=config.training_samples,
        way_num=config.way_num,
        seed=subset_seed,
    )
    model = build_pect(config.model).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters:,}")

    if config.mode == "train":
        weights_path, _ = _train(model, bundle, config, device)
    else:
        assert config.weights is not None
        weights_path = config.weights
    _load_weights(model, weights_path, device)

    metrics = _test(model, bundle, config, device)
    metrics["parameters"] = parameters
    write_key_values(config.run_dir / "metrics.txt", metrics)
    write_csv(config.run_dir / "metrics.csv", [metrics])
    append_summary(config.output_dir / "summary.csv", metrics)
    print(f"Results   : {config.run_dir}")
    return metrics
