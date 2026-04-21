"""Evaluatiescript voor fase 1 classificatie en fase 2 bbox-baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset_loader import load_phase_datasets
from src.data.preprocess import prepare_classification_dataset, prepare_detection_dataset
from src.utils.visualize import plot_confusion_matrix, plot_training_history


REPORTS_DIR = "reports/figures"
MODEL_DIR = "models"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Laad configuratie uit YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_history_if_available(history_path: str) -> dict | None:
    path = Path(history_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_classification(model: tf.keras.Model, test_ds: tf.data.Dataset, class_names: list[str]):
    """Evalueer classificatie met report + confusion matrix."""
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().astype(int).flatten().tolist())

    probs = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (probs >= 0.5).astype(int)

    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names if len(class_names) == 2 else ["No Drone", "Drone"],
        zero_division=0,
    )
    return matrix, report


def evaluate_detection(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """Evalueer fase 2 output op klassenauwkeurigheid en bbox MAE."""
    preds = model.predict(test_ds, verbose=0)

    y_true_batches = []
    for _, labels in test_ds:
        y_true_batches.append(labels.numpy())
    y_true = np.concatenate(y_true_batches, axis=0)

    true_cls = y_true[:, 0]
    pred_cls = (preds[:, 0] >= 0.5).astype(np.float32)
    cls_acc = float((pred_cls == true_cls).mean())

    bbox_true = y_true[:, 1:5]
    bbox_pred = preds[:, 1:5]
    bbox_mae = float(np.mean(np.abs(bbox_true - bbox_pred)))

    return {
        "classification_accuracy": cls_acc,
        "bbox_mae": bbox_mae,
    }


def evaluate(config_path: str, phase: str = "classification") -> dict:
    """Run evaluatie voor gekozen fase en sla resultaten/figuren op."""
    # Stap 1: Laad configuratie en model
    config = load_config(config_path)
    model_name = str(config.get("model_name", "mobilenetv2_drone"))
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{phase}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model niet gevonden: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Stap 2: Laad testdataset
    data = load_phase_datasets(config, phase=phase)
    test_ds = data["test"]

    if phase == "classification":
        # Stap 3: Preprocess en voorspel
        test_ds = prepare_classification_dataset(test_ds)
        matrix, report = evaluate_classification(model, test_ds, data["class_names"])

        # Stap 4: Sla rapportering op
        report_path = os.path.join(REPORTS_DIR, f"classification_report_{model_name}.txt")
        cm_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name}.png")
        with Path(report_path).open("w", encoding="utf-8") as file:
            file.write(report)

        fig = plot_confusion_matrix(matrix, class_names=["No Drone", "Drone"])
        fig.savefig(cm_path, dpi=150)

        # Stap 5: Visualiseer eventueel training history
        history_path = os.path.join(REPORTS_DIR, f"history_{model_name}_{phase}.json")
        history_dict = _load_history_if_available(history_path)
        history_fig_path = None
        if history_dict is not None:
            history_fig = plot_training_history(history_dict)
            history_fig_path = os.path.join(REPORTS_DIR, f"history_{model_name}_{phase}.png")
            history_fig.savefig(history_fig_path, dpi=150)

        return {
            "phase": phase,
            "model_path": model_path,
            "report_path": report_path,
            "confusion_matrix_path": cm_path,
            "history_figure_path": history_fig_path,
        }

    # Stap 3 (fase 2): Preprocess en evalueer bbox + klasse
    test_ds = prepare_detection_dataset(test_ds)
    detection_metrics = evaluate_detection(model, test_ds)
    metrics_path = os.path.join(REPORTS_DIR, f"detection_metrics_{model_name}.json")
    with Path(metrics_path).open("w", encoding="utf-8") as file:
        json.dump(detection_metrics, file, indent=2)

    return {
        "phase": phase,
        "model_path": model_path,
        "metrics_path": metrics_path,
        **detection_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI argumenten."""
    parser = argparse.ArgumentParser(description="Evalueer drone model")
    parser.add_argument("--config", default="configs/config.yaml", help="Pad naar config")
    parser.add_argument(
        "--phase",
        default="classification",
        choices=["classification", "detection"],
        help="Welke fase evalueren",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(config_path=args.config, phase=args.phase)
    print("Evaluatie voltooid:")
    for key, value in result.items():
        print(f"- {key}: {value}")
