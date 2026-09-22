"""Open the held-out organoids once, after training, and evaluate cross-organoid shape retrieval."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from train_shape_retrieval import ROOT, SHAPE_FEATURES, ShapeModel


def embeddings(model, array, ids, device, batch=12):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            x = torch.from_numpy(np.array(array[ids[start:start+batch]], copy=True)).unsqueeze(1).float().to(device)
            h, _, _ = model(x); result.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
    return np.concatenate(result)


def retrieval_metrics(embedding, truth, samples, k=5):
    predicted = pairwise_distances(embedding, metric="cosine")
    actual = pairwise_distances(truth)
    allowed = samples[:, None] != samples[None, :]
    recalls, ndcgs = [], []
    pair_pred, pair_actual = [], []
    for i in range(len(samples)):
        candidates = np.flatnonzero(allowed[i])
        if len(candidates) < k: continue
        true_order = candidates[np.argsort(actual[i, candidates])]
        pred_order = candidates[np.argsort(predicted[i, candidates])]
        relevant = set(true_order[:k]); recalls.append(len(relevant.intersection(pred_order[:k])) / k)
        gains = np.exp(-actual[i, pred_order[:k]])
        ideal = np.exp(-actual[i, true_order[:k]])
        discount = 1 / np.log2(np.arange(2, k + 2))
        ndcgs.append(float((gains @ discount) / max(ideal @ discount, 1e-12)))
        pair_pred.extend(predicted[i, candidates].tolist()); pair_actual.extend(actual[i, candidates].tolist())
    rho = spearmanr(pair_pred, pair_actual).statistic
    return {f"recall_at_{k}": float(np.mean(recalls)), f"ndcg_at_{k}": float(np.mean(ndcgs)),
            "distance_spearman": float(rho), "queries": len(recalls)}


def main():
    cfg = json.loads((ROOT / "config/shape_retrieval.json").read_text()); out = ROOT / cfg["output_dir"]
    status = json.loads((out / "status.json").read_text())
    if status["status"] != "trained_pending_test_evaluation": raise RuntimeError("Training has not completed")
    manifest = pd.read_csv(out / "manifest.csv"); test_ids = manifest.index[manifest.split == "test"].to_numpy()
    train_ids = manifest.index[manifest.split == "train"].to_numpy()
    device = torch.device(cfg["device"])
    student = ShapeModel(cfg).to(device); teacher = ShapeModel(cfg).to(device)
    student.load_state_dict(torch.load(out / "student_best.pt", map_location="cpu", weights_only=False)["model"])
    teacher.load_state_dict(torch.load(out / "teacher_best.pt", map_location="cpu", weights_only=False)["model"])
    raw = np.load(ROOT / "results/simclr3d_raw/raw64.npy", mmap_mode="r"); masks = np.load(out / "masks_fixed64.npy", mmap_mode="r")
    raw_indices = manifest["index"].to_numpy(int)
    student_h = embeddings(student, raw, raw_indices[test_ids], device, cfg["batch_size"])
    teacher_h = embeddings(teacher, masks, test_ids, device, cfg["batch_size"])
    features = manifest[SHAPE_FEATURES].to_numpy(float)
    mean = features[train_ids].mean(0); std = features[train_ids].std(0).clip(1e-8)
    truth = (features[test_ids] - mean) / std
    samples = manifest.loc[test_ids, "sample"].to_numpy()
    result = {"test_stacks": int(len(np.unique(samples))), "test_cells": int(len(test_ids)),
              "retrieval_scope": "candidate cells from different held-out organoids",
              "student_raw": retrieval_metrics(student_h, truth, samples),
              "teacher_mask_upper_bound": retrieval_metrics(teacher_h, truth, samples)}
    np.save(out / "test_student_embeddings.npy", student_h); np.save(out / "test_teacher_embeddings.npy", teacher_h)
    pd.DataFrame({"cell_id": manifest.loc[test_ids, "cell_id"], "sample": samples,
                  **{f"embedding_{i:03d}": student_h[:, i] for i in range(student_h.shape[1])}}).to_csv(out / "test_embeddings.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    status.update({"status": "complete", "test_evaluation_complete": True})
    (out / "status.json").write_text(json.dumps(status, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
