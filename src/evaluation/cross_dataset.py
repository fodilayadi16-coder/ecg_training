"""
Cross-Dataset Evaluation – General Pipeline
============================================
Pure logic: load records, extract segments, normalise, predict, compute metrics.
No hardcoded paths, no plotting, no print titles.
Experiments scripts in src/experiments/ call these functions.
"""

import numpy as np
from tensorflow.keras.models import load_model

# ── Data helpers ─────────────────────────────

def normalize(X):
    """Per-sample zero-mean / unit-variance (same as training)."""
    mean = np.mean(X, axis=1, keepdims=True)
    std  = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


# ── Extraction wrappers ─────────────────────

def extract_beat_data(records, extract_fn, label_map):
    """
    Extract beat segments from a list of WFDB record paths.

    Parameters
    ----------
    records    : list[str]  – full record paths (no extension)
    extract_fn : callable   – (records) -> (X, y, rr) from beat_extractor
    label_map  : dict       – symbol -> int mapping

    Returns
    -------
    X : ndarray (N, window, 1)
    y : ndarray (N,)
    """
    X, y, _ = extract_fn(records)
    return X, y


def extract_rhythm_data(records, extract_fn):
    """
    Extract rhythm windows from a list of WFDB record paths.

    Parameters
    ----------
    records    : list[str]
    extract_fn : callable  – (records) -> (X, y) from rhythm_extractor

    Returns
    -------
    X : ndarray (N, window, 1)
    y : ndarray (N,)
    """
    X, y = extract_fn(records)
    return X, y


# ── Prediction ───────────────────────────────

def predict_multiclass(model, X):
    """Softmax model → class indices."""
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob


def predict_binary(model, X, threshold=0.5):
    """Sigmoid model → 0/1."""
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


# ── Full pipelines ───────────────────────────

def evaluate_beat_cross(model_path, records, extract_fn, metrics_fn):
    """
    End-to-end cross-dataset evaluation for the beat model.

    Parameters
    ----------
    model_path : str         – path to .h5 model
    records    : list[str]   – eval record paths
    extract_fn : callable    – beat extraction function
    metrics_fn : callable    – (y_true, y_pred) -> dict

    Returns
    -------
    metrics : dict
    y_true  : ndarray
    y_pred  : ndarray
    """
    X, y_true = extract_beat_data(records, extract_fn, label_map=None)
    X = normalize(X)

    model = load_model(model_path, compile=False)
    y_pred, _ = predict_multiclass(model, X)

    metrics = metrics_fn(y_true, y_pred)
    return metrics, y_true, y_pred


def evaluate_rhythm_cross(model_path, records, extract_fn, metrics_fn,
                          threshold=0.5):
    """
    End-to-end cross-dataset evaluation for the rhythm model.

    Parameters
    ----------
    model_path : str
    records    : list[str]
    extract_fn : callable    – rhythm extraction function
    metrics_fn : callable    – (y_true, y_pred) -> dict
    threshold  : float       – sigmoid decision boundary

    Returns
    -------
    metrics : dict
    y_true  : ndarray
    y_pred  : ndarray
    """
    X, y_true = extract_rhythm_data(records, extract_fn)
    X = normalize(X)

    model = load_model(model_path, compile=False)
    y_pred, _ = predict_binary(model, X, threshold)

    metrics = metrics_fn(y_true, y_pred)
    return metrics, y_true, y_pred
