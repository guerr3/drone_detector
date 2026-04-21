"""Inference helpers for single-image drone presence prediction."""

import numpy as np
import tensorflow as tf


def load_and_prepare_image(image_path: str, image_size: tuple[int, int]) -> np.ndarray:
    """Load image from disk and prepare batch tensor for inference."""
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return array


def predict_image(model, image_path: str, image_size: tuple[int, int] = (224, 224)):
    """Run prediction and return class index and confidence."""
    batch = load_and_prepare_image(image_path, image_size=image_size)
    probs = model.predict(batch)
    class_idx = int(np.argmax(probs[0]))
    confidence = float(np.max(probs[0]))
    return class_idx, confidence
