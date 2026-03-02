"""
Cross-Dataset Beat Experiment
=============================
Evaluate the beat model on unseen datasets (QTDB, LUDB)
using the general pipeline from evaluation/cross_dataset.py.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.cross_dataset import evaluate_beat_cross
from evaluation.beat_metrics import (
    compute_metrics,
    per_class_sensitivity,
    per_class_specificity,
    print_report,
    plot_confusion_matrix,
)
from preprocessing.beat_extractor import extract_beats


# ── Record finder for eval folders ───────────

def find_records(base_path):
    """Return list of record paths (no extension) under a single folder."""
    records = []
    for f in os.listdir(base_path):
        if f.endswith(".hea"):
            records.append(os.path.join(base_path, f.replace(".hea", "")))
    return sorted(records)


# ── Datasets to evaluate ─────────────────────

EVAL_DATASETS = {
    "QTDB": "data/eval/qtdb",
}

MODEL_PATH  = "models/cnn/cnn_beat_best.h5"
RESULTS_DIR = "results_cnn/cross_dataset_cnn"


# ── Run experiments ──────────────────────────

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for name, path in EVAL_DATASETS.items():
        print(f"\n{'='*50}")
        print(f"  Dataset: {name}")
        print(f"{'='*50}")

        records = find_records(path)
        print(f"Records found: {len(records)}")

        if len(records) == 0:
            print("No records, skipping.")
            continue

        # --- evaluate ---
        metrics, y_true, y_pred = evaluate_beat_cross(
            model_path=MODEL_PATH,
            records=records,
            extract_fn=extract_beats,
            metrics_fn=compute_metrics,
        )

        print(f"Beats extracted: {len(y_true)}")
        unique, counts = np.unique(y_true, return_counts=True)
        print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))

        # --- print results ---
        print_report(y_true, y_pred)

        sens = per_class_sensitivity(y_true, y_pred)
        spec = per_class_specificity(y_true, y_pred)
        print("Sensitivity:", sens)
        print("Specificity:", spec)

        # --- plots ---
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"Beat CM – {name}",
            save_path=os.path.join(RESULTS_DIR, f"cm_{name.lower()}.png"),
        )
        plot_confusion_matrix(
            y_true, y_pred,
            normalize="true",
            title=f"Beat CM (norm) – {name}",
            save_path=os.path.join(RESULTS_DIR, f"cm_norm_{name.lower()}.png"),
        )

        # --- save json ---
        metrics["sensitivity"] = sens
        metrics["specificity"] = spec
        all_results[name] = metrics

    # save combined results
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    run()
