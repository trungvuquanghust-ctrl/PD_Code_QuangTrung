"""The two publication figures retained in TIM_2026."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
        }
    )


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    _paper_style()
    matrix = confusion_matrix(targets, predictions, labels=range(len(class_names)))
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    percentages = matrix / row_sums * 100.0
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            annotations[row, column] = f"{matrix[row, column]}\n({percentages[row, column]:.1f}%)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.16, 7.16))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Greens",
        linewidths=0.5,
        linecolor="white",
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
    seed: int = 42,
) -> None:
    if len(features) < 6:
        raise ValueError("t-SNE requires at least 6 samples")
    _paper_style()
    scaled = StandardScaler().fit_transform(features)
    components = min(50, len(scaled), scaled.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(scaled)
    perplexity = min(30, max(5, len(reduced) // 50), len(reduced) - 1)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.16, 5.2))
    palette = sns.color_palette("colorblind", len(class_names))
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=28,
            alpha=0.8,
            color=palette[class_id],
            label=class_name,
            edgecolors="none",
        )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
