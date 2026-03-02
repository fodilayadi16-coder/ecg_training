"""
Beat Model – Evaluation Metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BEAT_CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = len(BEAT_CLASSES)


# ── Metrics ──────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return a dict with accuracy, macro precision, recall, f1."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_sensitivity(y_true, y_pred):
    """Per-class sensitivity (recall / TPR)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    sens = {}
    for i, cls in enumerate(BEAT_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sens[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sens


def per_class_specificity(y_true, y_pred):
    """Per-class specificity (TNR)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    spec = {}
    for i, cls in enumerate(BEAT_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spec


def print_report(y_true, y_pred):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred,
                                target_names=BEAT_CLASSES,
                                labels=range(NUM_CLASSES),
                                zero_division=0))


# ── Confusion Matrix Plot ────────────────────

def plot_confusion_matrix(y_true, y_pred, normalize=None,
                          title="Beat – Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix heatmap.
    normalize : None | 'true' | 'pred' | 'all'
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES), normalize=normalize)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=BEAT_CLASSES, yticklabels=BEAT_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()
