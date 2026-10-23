"""
Evaluate the detector on a Hugging Face dataset.

Example:
    python evaluation/evaluate.py --split test --max-samples 500

If the dataset uses non-standard column names:
    python evaluation/evaluate.py --text-column text --label-column label

This script assumes the dataset's AI class is label 1 (matching the model card).
Inspect the dataset's label feature before interpreting the metrics.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import SlopDetector


DATASET_ID = "gouwsxander/wikipedia-human-ai"


def find_column(columns, requested, candidates, kind):
    if requested:
        if requested not in columns:
            raise ValueError(f"{kind} column '{requested}' not found. Available: {columns}")
        return requested
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not infer the {kind} column from {columns}. "
        f"Pass it explicitly with --{kind}-column."
    )


def label_to_int(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "ai", "ai-generated", "ai generated", "synthetic"}:
            return 1
        if normalized in {"0", "human", "human-written", "human written", "real"}:
            return 0
    raise ValueError(
        f"Unrecognized label value {value!r}. Map your dataset labels to 0=human, 1=AI."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    text_col = find_column(
        dataset.column_names,
        args.text_column,
        ["text", "content", "paragraph", "document"],
        "text",
    )
    label_col = find_column(
        dataset.column_names,
        args.label_column,
        ["label", "labels", "class", "generated"],
        "label",
    )

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    detector = SlopDetector()
    y_true, y_pred, y_score = [], [], []

    for index, row in enumerate(dataset):
        true_label = label_to_int(row[label_col])
        result = detector.predict(str(row[text_col]))
        y_true.append(true_label)
        y_pred.append(1 if result["ai_probability"] >= 0.5 else 0)
        y_score.append(result["ai_probability"])

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(dataset)} examples")

    print("\nMetrics (0=human, 1=AI):")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_true, y_score):.4f}")
    except ValueError:
        print("ROC-AUC  : unavailable (only one class found)")
    print("\nConfusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Human", "AI"], zero_division=0
    ))


if __name__ == "__main__":
    main()
