"""Visualisatiehulpfuncties voor training, evaluatie en inferentie."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history: dict):
    """Toon train/val curves voor accuracy en loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("accuracy", []), label="train")
    axes[0].plot(history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.get("loss", []), label="train")
    axes[1].plot(history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]):
    """Visualiseer confusion matrix als heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Werkelijk")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_prediction_samples(
    images: np.ndarray,
    predicted_labels: list[str],
    confidences: list[float],
    max_samples: int = 6,
):
    """Toon voorbeeldbeelden met label en confidence."""
    n = min(max_samples, len(images), len(predicted_labels), len(confidences))
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx in range(n):
        axes[idx].imshow(np.clip(images[idx].astype(np.float32) / 255.0, 0.0, 1.0))
        axes[idx].set_title(f"{predicted_labels[idx]} ({confidences[idx]:.2f})")
        axes[idx].axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
