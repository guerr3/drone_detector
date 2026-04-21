"""Training entry points for drone image classification models."""

from pathlib import Path

import tensorflow as tf
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classifier(image_size: tuple[int, int], num_classes: int = 2) -> tf.keras.Model:
    """Build a transfer-learning classifier with MobileNetV2."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile model with sensible classification defaults."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
