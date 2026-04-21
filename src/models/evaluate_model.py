"""Evaluation helpers for trained classification models."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred):
    """Return confusion matrix and text classification report."""
    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return matrix, report


def predict_labels(model, dataset):
    """Generate class predictions for a TensorFlow dataset."""
    probabilities = model.predict(dataset)
    return np.argmax(probabilities, axis=1)
