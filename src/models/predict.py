"""Predictiescript voor fase 1 classificatie en fase 2 bbox baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yaml


MODEL_DIR = "models"
REPORTS_DIR = "reports/figures"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Laad configuratie uit YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_and_prepare_image(image_path: str, image_size: tuple[int, int]) -> np.ndarray:
    """Laad afbeelding en zet om naar batchtensor."""
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array.astype(np.float32), axis=0)


def predict_classification(
    model: tf.keras.Model,
    image_path: str,
    image_size: tuple[int, int],
    threshold: float = 0.5,
) -> dict:
    """Voer fase 1 predictie uit voor 1 beeld."""
    batch = load_and_prepare_image(image_path, image_size)
    score = float(model.predict(batch, verbose=0).reshape(-1)[0])
    predicted_label = "Drone" if score >= threshold else "No Drone"
    confidence = score if score >= threshold else 1.0 - score
    return {
        "predicted_label": predicted_label,
        "score": score,
        "confidence": float(confidence),
        "threshold": threshold,
    }


def _bbox_from_normalized(
    bbox_normalized: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xmin = int(np.clip(bbox_normalized[0], 0.0, 1.0) * width)
    ymin = int(np.clip(bbox_normalized[1], 0.0, 1.0) * height)
    xmax = int(np.clip(bbox_normalized[2], 0.0, 1.0) * width)
    ymax = int(np.clip(bbox_normalized[3], 0.0, 1.0) * height)
    return xmin, ymin, xmax, ymax


def predict_detection(
    model: tf.keras.Model,
    image_path: str,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    save_visualization: bool = True,
) -> dict:
    """Voer fase 2 baseline predictie uit met klasse + bbox."""
    batch = load_and_prepare_image(image_path, image_size)
    pred = model.predict(batch, verbose=0).reshape(-1)

    class_score = float(pred[0])
    bbox_norm = pred[1:5]
    is_drone = class_score >= threshold

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Afbeelding niet gevonden of onleesbaar: {image_path}")

    h, w = image_bgr.shape[:2]
    xmin, ymin, xmax, ymax = _bbox_from_normalized(bbox_norm, w, h)

    vis_path = None
    if save_visualization:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        color = (0, 255, 0) if is_drone else (0, 0, 255)
        label_text = f"Drone: {class_score:.2f}" if is_drone else f"No Drone: {1-class_score:.2f}"
        cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image_bgr,
            label_text,
            (max(xmin, 10), max(ymin - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        stem = Path(image_path).stem
        vis_path = os.path.join(REPORTS_DIR, f"prediction_bbox_{stem}.jpg")
        cv2.imwrite(vis_path, image_bgr)

    return {
        "predicted_label": "Drone" if is_drone else "No Drone",
        "score": class_score,
        "bbox_normalized": [float(x) for x in bbox_norm.tolist()],
        "bbox_pixels": [xmin, ymin, xmax, ymax],
        "visualization_path": vis_path,
    }


def run_prediction(
    config_path: str,
    image_path: str,
    phase: str = "classification",
    threshold: float = 0.5,
) -> dict:
    """Voer predictie uit op basis van gekozen fase."""
    config = load_config(config_path)
    image_size = tuple(config.get("image_size", [224, 224]))
    model_name = str(config.get("model_name", "mobilenetv2_drone"))
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{phase}.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model niet gevonden: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    if phase == "classification":
        return predict_classification(model, image_path, image_size, threshold=threshold)
    if phase == "detection":
        return predict_detection(model, image_path, image_size, threshold=threshold)
    raise ValueError("phase moet 'classification' of 'detection' zijn")


def parse_args() -> argparse.Namespace:
    """Parse CLI argumenten voor predictie."""
    parser = argparse.ArgumentParser(description="Predict met drone model")
    parser.add_argument("--config", default="configs/config.yaml", help="Pad naar config")
    parser.add_argument("--image", required=True, help="Pad naar inputafbeelding")
    parser.add_argument(
        "--phase",
        default="classification",
        choices=["classification", "detection"],
        help="Fase voor predictie",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classificatiedrempel",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_prediction(
        config_path=args.config,
        image_path=args.image,
        phase=args.phase,
        threshold=args.threshold,
    )
    print("Predictie resultaat:")
    for key, value in output.items():
        print(f"- {key}: {value}")
