"""Dataset loading helpers for fase 1 (classificatie) en fase 2 (bbox)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def _to_hw_tuple(image_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Zet een lijst/tuple om naar (hoogte, breedte)."""
    if len(image_size) != 2:
        raise ValueError("image_size moet exact 2 waarden bevatten")
    return int(image_size[0]), int(image_size[1])


def load_classification_dataset(
    directory: str,
    image_size: tuple[int, int] | list[int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
) -> tuple[tf.data.Dataset, list[str]]:
    """Laad een classificatie-dataset vanuit mapstructuur per klasse."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory bestaat niet: {data_dir}")

    hw = _to_hw_tuple(image_size)
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=hw,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=shuffle,
    )
    class_names = list(dataset.class_names)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, class_names


def load_detection_dataset(
    annotations_csv: str,
    image_root: str,
    image_size: tuple[int, int] | list[int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Laad dataset met labels [class, xmin, ymin, xmax, ymax] genormaliseerd [0, 1].

    Vereiste CSV-kolommen:
    - image_path (relatief t.o.v. image_root, of absoluut)
    - label (0/1)
    - xmin, ymin, xmax, ymax (genormaliseerd tussen 0 en 1)
    """
    csv_path = Path(annotations_csv)
    root = Path(image_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotatiebestand bestaat niet: {csv_path}")
    if not root.exists():
        raise FileNotFoundError(f"Image root bestaat niet: {root}")

    frame = pd.read_csv(csv_path)
    required_cols = {"image_path", "label", "xmin", "ymin", "xmax", "ymax"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise ValueError(f"Ontbrekende kolommen in annotaties: {sorted(missing)}")

    image_paths: list[str] = []
    for rel_or_abs in frame["image_path"].astype(str).tolist():
        p = Path(rel_or_abs)
        image_paths.append(str(p if p.is_absolute() else root / p))

    targets = frame[["label", "xmin", "ymin", "xmax", "ymax"]].to_numpy(
        dtype=np.float32
    )

    hw = _to_hw_tuple(image_size)
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    target_ds = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((path_ds, target_ds))

    def _load_image(path: tf.Tensor, target: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, hw)
        image = tf.cast(image, tf.float32)
        return image, target

    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(len(frame), batch_size * 4))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_phase_datasets(config: dict, phase: str = "classification") -> dict:
    """Laad train/val/test datasets op basis van fase en config."""
    image_size = config.get("image_size", [224, 224])
    batch_size = int(config.get("batch_size", 32))

    if phase == "classification":
        train_ds, class_names = load_classification_dataset(
            config["train_dir"], image_size, batch_size, shuffle=True
        )
        val_ds, _ = load_classification_dataset(
            config["val_dir"], image_size, batch_size, shuffle=False
        )
        test_ds, _ = load_classification_dataset(
            config["test_dir"], image_size, batch_size, shuffle=False
        )
        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "class_names": class_names,
        }

    if phase == "detection":
        train_ds = load_detection_dataset(
            config["train_annotations"],
            config["train_images_dir"],
            image_size,
            batch_size,
            shuffle=True,
        )
        val_ds = load_detection_dataset(
            config["val_annotations"],
            config["val_images_dir"],
            image_size,
            batch_size,
            shuffle=False,
        )
        test_ds = load_detection_dataset(
            config["test_annotations"],
            config["test_images_dir"],
            image_size,
            batch_size,
            shuffle=False,
        )
        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "class_names": ["No Drone", "Drone"],
        }

    raise ValueError("Ongeldige phase. Gebruik 'classification' of 'detection'.")
