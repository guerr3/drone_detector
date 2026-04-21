"""Preprocessing utilities voor fase 1 en fase 2 workflows."""

from __future__ import annotations

import tensorflow as tf


def build_data_augmentation() -> tf.keras.Sequential:
    """Bouw augmentatie voor kleine datasets om overfitting te beperken."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def normalize_images(images: tf.Tensor, labels: tf.Tensor):
    """Normaliseer naar [0, 1]."""
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def mobilenet_preprocess(images: tf.Tensor, labels: tf.Tensor):
    """Gebruik de officiële MobileNetV2 preprocessing naar [-1, 1]."""
    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(images, tf.float32)
    )
    return images, labels


def ensure_float_images(images: tf.Tensor, labels: tf.Tensor):
    """Zet enkel dtype om naar float32 zonder extra schaling."""
    return tf.cast(images, tf.float32), labels


def prepare_classification_dataset(
    dataset: tf.data.Dataset,
    use_mobilenet_preprocess: bool = False,
) -> tf.data.Dataset:
    """Pas preprocessing toe op een classificatie dataset."""
    if use_mobilenet_preprocess:
        dataset = dataset.map(mobilenet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(ensure_float_images, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def prepare_detection_dataset(
    dataset: tf.data.Dataset,
    use_mobilenet_preprocess: bool = False,
) -> tf.data.Dataset:
    """Preprocessing voor fase 2 bbox-trainingsdata."""
    if use_mobilenet_preprocess:
        dataset = dataset.map(mobilenet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(ensure_float_images, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)
