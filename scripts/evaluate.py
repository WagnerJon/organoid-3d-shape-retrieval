"""Evaluate 3D instance labels with one-to-one IoU matching (no training)."""
import argparse
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


def evaluate(reference, prediction, threshold=0.5):
    if reference.shape != prediction.shape:
        raise ValueError("Prediction must be on the reference/native image grid")
    ref_ids, ref = np.unique(reference, return_inverse=True)
    pred_ids, pred = np.unique(prediction, return_inverse=True)
    counts = np.bincount(ref * len(pred_ids) + pred, minlength=len(ref_ids) * len(pred_ids)).reshape(len(ref_ids), len(pred_ids))
    intersection = counts[ref_ids != 0][:, pred_ids != 0]
    ref_area = counts.sum(axis=1)[ref_ids != 0]
    pred_area = counts.sum(axis=0)[pred_ids != 0]
    union = ref_area[:, None] + pred_area[None, :] - intersection
    iou = intersection / np.maximum(union, 1)
    n_ref, n_pred = iou.shape
    # Maximize valid matches first, then total IoU to break ties.
    row, col = linear_sum_assignment(-(iou >= threshold).astype(float) - iou / (2 * max(min(iou.shape), 1)))
    matched = iou[row, col] >= threshold
    tp = int(matched.sum())
    fp, fn = n_pred - tp, n_ref - tp
    return {"iou_threshold": threshold, "reference_cells": n_ref, "predicted_cells": n_pred,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "mean_iou_of_valid_matches": float(iou[row[matched], col[matched]].mean()) if tp else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(tifffile.imread(args.reference), tifffile.imread(args.prediction))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
