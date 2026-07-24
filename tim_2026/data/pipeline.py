"""Tensor preparation for the unchanged PD scalogram loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .loading import CLASS_DISPLAY_NAMES, load_dataset


@dataclass(slots=True)
class DatasetBundle:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    class_names: list[str]


def _to_tensor(images: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels).long()


def _balanced_training_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    total_samples: int | None,
    way_num: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_samples is None:
        return images, labels
    if total_samples % way_num != 0:
        raise ValueError("training_samples must be divisible by way_num")

    per_class = total_samples // way_num
    selected_images: list[torch.Tensor] = []
    selected_labels: list[torch.Tensor] = []
    for class_id in range(way_num):
        indices = (labels == class_id).nonzero(as_tuple=True)[0]
        if len(indices) < per_class:
            raise ValueError(f"Class {class_id}: need {per_class} samples, found {len(indices)}")
        # This deliberately mirrors the source pipeline: the same seed is reset
        # independently for each class before randperm.
        generator = torch.Generator().manual_seed(seed)
        chosen = indices[torch.randperm(len(indices), generator=generator)[:per_class]]
        selected_images.append(images[chosen])
        selected_labels.append(labels[chosen])
    return torch.cat(selected_images), torch.cat(selected_labels)


def load_dataset_bundle(
    dataset_path: Path,
    *,
    image_size: int,
    training_samples: int | None,
    way_num: int,
    seed: int,
) -> DatasetBundle:
    dataset = load_dataset(str(dataset_path), image_size=image_size)
    train_images, train_labels = _to_tensor(dataset.X_train, dataset.y_train)
    val_images, val_labels = _to_tensor(dataset.X_val, dataset.y_val)
    test_images, test_labels = _to_tensor(dataset.X_test, dataset.y_test)

    train_images, train_labels = _balanced_training_subset(
        train_images,
        train_labels,
        training_samples,
        way_num,
        seed,
    )
    class_names = [CLASS_DISPLAY_NAMES.get(name, name) for name in dataset.classes]
    if len(class_names) != way_num:
        raise ValueError(f"Expected {way_num} classes, found {len(class_names)}: {class_names}")

    return DatasetBundle(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        class_names=class_names,
    )
