"""Trainingsscript voor fase 1 classificatie en fase 2 bbox-baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
import yaml

from src.data.dataset_loader import load_phase_datasets
from src.data.preprocess import prepare_classification_dataset, prepare_detection_dataset


DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_SAVE_DIR = "models"
REPORTS_DIR = "reports/figures"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Laad configuratie uit YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_phase1_classifier(image_size: tuple[int, int], freeze_base: bool = True) -> tf.keras.Model:
    """Bouw fase 1 model met Sequential API en MobileNetV2 transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    model = tf.keras.Sequential(name="drone_classifier_phase1")
    model.add(tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)))
    model.add(tf.keras.layers.Resizing(image_size[0], image_size[1]))
    model.add(
        tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)
    )
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


def _detection_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combineer binaire classificatieloss met bbox regressieloss.

    Verwachte target-opbouw: [label, xmin, ymin, xmax, ymax].
    """
    cls_true = y_true[:, 0:1]
    box_true = y_true[:, 1:5]

    cls_pred = y_pred[:, 0:1]
    box_pred = y_pred[:, 1:5]

    bce = tf.keras.losses.binary_crossentropy(cls_true, cls_pred)
    mse = tf.reduce_mean(tf.square(box_true - box_pred), axis=1)
    return bce + 2.0 * mse


def build_phase2_detector(image_size: tuple[int, int], freeze_base: bool = True) -> tf.keras.Model:
    """Bouw fase 2 baseline met één Sequential outputvector: [p, xmin, ymin, xmax, ymax]."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    model = tf.keras.Sequential(name="drone_detector_phase2")
    model.add(tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)))
    model.add(tf.keras.layers.Resizing(image_size[0], image_size[1]))
    model.add(
        tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)
    )
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
    return model


def compile_phase1_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compileer fase 1 classificatiemodel."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def compile_phase2_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compileer fase 2 detectiemodel met gecombineerde loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_detection_loss,
        metrics=["mae"],
    )


def _save_history(history: tf.keras.callbacks.History, path: str) -> None:
    """Bewaar trainhistory als JSON voor latere visualisatie."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)


def train(config_path: str, phase: str = "classification") -> dict:
    """Train model voor de gekozen fase."""
    # Stap 1: Laad configuratie
    config = load_config(config_path)
    image_size = tuple(config.get("image_size", [224, 224]))
    epochs = int(config.get("epochs", 10))
    learning_rate = float(config.get("learning_rate", 1e-3))
    model_name = str(config.get("model_name", "mobilenetv2_drone"))

    # Stap 2: Definieer projectpaden
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Stap 3: Laad datasets
    datasets = load_phase_datasets(config, phase=phase)
    train_ds = datasets["train"]
    val_ds = datasets["val"]

    # Stap 4: Data preprocessing
    if phase == "classification":
        train_ds = prepare_classification_dataset(train_ds)
        val_ds = prepare_classification_dataset(val_ds)
    else:
        train_ds = prepare_detection_dataset(train_ds)
        val_ds = prepare_detection_dataset(val_ds)

    # Stap 5: Bouw model
    if phase == "classification":
        model = build_phase1_classifier(image_size=image_size, freeze_base=True)
        compile_phase1_model(model, learning_rate)
    elif phase == "detection":
        model = build_phase2_detector(image_size=image_size, freeze_base=True)
        compile_phase2_model(model, learning_rate)
    else:
        raise ValueError("phase moet 'classification' of 'detection' zijn")

    # Stap 6: Train model met validatie
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    # Stap 7: Sla model en history op
    model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{phase}.keras")
    history_path = os.path.join(REPORTS_DIR, f"history_{model_name}_{phase}.json")
    model.save(model_path)
    _save_history(history, history_path)

    return {
        "phase": phase,
        "model_path": model_path,
        "history_path": history_path,
        "epochs": epochs,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI argumenten voor lokaal of Colab gebruik."""
    parser = argparse.ArgumentParser(description="Train drone model voor fase 1 of fase 2")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Pad naar YAML config",
    )
    parser.add_argument(
        "--phase",
        default="classification",
        choices=["classification", "detection"],
        help="Te trainen fase",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(config_path=args.config, phase=args.phase)
    print("Training voltooid:")
    for key, value in result.items():
        print(f"- {key}: {value}")
