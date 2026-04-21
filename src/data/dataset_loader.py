"""Dataset loading helpers for image classification experiments."""

from pathlib import Path

import tensorflow as tf


def load_image_dataset(
    directory: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
):
    """Load an image dataset from a directory structure.

    Expected folder format:
        directory/
            class_a/
            class_b/
    """
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
