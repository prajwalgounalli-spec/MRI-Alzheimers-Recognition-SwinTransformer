"""Evaluation utilities for Alzheimer MRI classification."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, class_names=None, output_dir="results"):
    """Evaluate existing predictions and save genuine measured results."""
    labels = sorted(set(np.asarray(y_true).tolist()) |
                    set(np.asarray(y_pred).tolist()))

    report = classification_report(
        y_true, y_pred, labels=labels,
        target_names=class_names if class_names else None,
        output_dict=True, zero_division=0
    )

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
        "labels": labels,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print("Evaluation utility ready.")
    print("Call evaluate_predictions(y_true, y_pred) with real test labels")
    print("and predictions to generate evaluation_metrics.json.")
