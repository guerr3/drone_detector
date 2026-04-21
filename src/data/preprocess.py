"""Preprocessing utilities for image data."""

import tensorflow as tf


def build_data_augmentation() -> tf.keras.Sequential:
    """Create a simple data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def normalize_images(images, labels):
    """Normalize image pixel values to the [0, 1] range."""
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels
