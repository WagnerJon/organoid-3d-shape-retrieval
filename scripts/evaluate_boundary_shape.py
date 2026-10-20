"""Evaluate grouped-holdout raw shape retrieval against matched controls."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances

from train_boundary_shape import ROOT, ShapeEncoder


SHAPE_FEATURES = ["sphericity", "aspect_ratio", "elongation", "flatness", "mesh_solidity"]


def arrangement(labels):
    centers = []
    for label in np.unique(labels):
        if label > 0:
            centers.append(np.argwhere(labels == label).mean(0))
    centers = np.asarray(centers)
    if len(centers) < 2:
        return np.array([], dtype=np.float32)
    delta = centers[:, None] - centers[None, :]
    distances = np.sort(np.linalg.norm(delta, axis=-1)[np.triu_indices(len(centers), 1)])
    return (distances / max(float(np.sqrt(np.mean(distances ** 2))), 1e-6)).astype(np.float32)


def retrieval(frame, vectors, targets, control_vectors=None, min_candidates=3):
    distance = pairwise_distances(vectors, metric="cosine")
    control_distance = pairwise_distances(control_vectors, metric="cosine") if control_vectors is not None else None
    rows = []
    for i, query in frame.iterrows():
        candidates = frame.index[(frame.level == query.level) &
            (frame.member_count == query.member_count) &
            (frame["sample"] != query["sample"]) &
            (np.abs(np.log(frame.volume_um3 / query.volume_um3)) <= np.log(1.25))].to_numpy()
        if len(candidates) < min_candidates:
            continue
        truth = np.array([np.linalg.norm(targets[i] - targets[j]) for j in candidates])
        if np.ptp(truth) < 1e-8:
            continue
        selected = candidates[np.argmin(distance[i, candidates])]
        volume = candidates[np.argmin(np.abs(np.log(frame.loc[candidates, "volume_um3"] / query.volume_um3)))]
        optimum = float(truth.min())
        choices = [("raw_shape_encoder", selected), ("volume_baseline", volume)]
        if control_distance is not None:
            choices.append(("untrained_raw_encoder", candidates[np.argmin(control_distance[i, candidates])]))
        for method, found in choices:
            value = float(np.linalg.norm(targets[i] - targets[found]))
            rows.append({"level": query.level, "query": int(i), "sample": query["sample"],
                         "similarity_group": int(query.similarity_group), "member_count": int(query.member_count),
                         "candidates": int(len(candidates)), "method": method,
                         "top1_hit": float(value <= optimum + 1e-8),
                         "rank_percentile": float(np.mean(truth >= value)),
                         "target_distance": value, "retrieved": int(found)})
    return pd.DataFrame(rows)


def grouped_signflip(results, level, seed=23, trials=10000):
    sub = results[results.level == level]
    if sub.empty:
        return {"groups": 0, "paired_difference": None, "p_two_sided": None}
    wide = sub.pivot(index=["query", "similarity_group"], columns="method", values="top1_hit")
    delta = wide["raw_shape_encoder"] - wide["volume_baseline"]
    group_means = delta.groupby(level="similarity_group").mean().to_numpy()
    observed = abs(float(group_means.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(trials, len(group_means)))
    random_values = np.abs((signs * group_means).mean(1))
    return {"groups": int(len(group_means)), "paired_difference": float(delta.mean()),
            "p_two_sided": float((np.count_nonzero(random_values >= observed) + 1) / (trials + 1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/boundary_shape.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.device:
        cfg["device"] = args.device
    out = ROOT / cfg["output_dir"]
    report_out = args.report_dir or out
    report_out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(out / "manifest.csv").set_index("index", drop=False)
    groups = pd.read_csv(out / "split_groups.csv").set_index("sample")
    frame["similarity_group"] = frame["sample"].map(groups.similarity_group)
    test = frame[frame.split == "test"].copy()
    raw = np.load(out / "raw96.npy", mmap_mode="r")
    labels = np.load(out / "labels96.npy", mmap_mode="r")
    checkpoint = torch.load(args.checkpoint or out / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device(cfg["device"])
    model = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(cfg["seed"] + 1)
    control = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device).eval()
    embeddings, control_embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(test), 4):
            ids = test.index[start:start + 4].to_numpy()
            x = torch.from_numpy(np.array(raw[ids], copy=True)).unsqueeze(1).float().to(device)
            h, _ = model(x, decode=False)
            embeddings.append(h.cpu().numpy())
            h0, _ = control(x, decode=False)
            control_embeddings.append(h0.cpu().numpy())
    embedding = np.concatenate(embeddings)
    control_embedding = np.concatenate(control_embeddings)
    np.save(report_out / "test_embeddings.npy", embedding)
    test = test.reset_index(drop=True)
    morphology = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    train_cell_ids = frame.loc[(frame.split == "train") & (frame.level == "single"), "source_cell_id"].dropna()
    training_features = morphology.loc[train_cell_ids, SHAPE_FEATURES].to_numpy(float)
    mean = np.nanmean(training_features, axis=0)
    std = np.nanstd(training_features, axis=0).clip(1e-6)
    targets = []
    for _, row in test.iterrows():
        if row.level == "single":
            target = ((morphology.loc[row.source_cell_id, SHAPE_FEATURES].to_numpy(float) - mean) / std).astype(np.float32)
        else:
            target = arrangement(labels[int(row["index"])])
        targets.append(target)
    records = []
    for level in ["single", "neighborhood", "organoid"]:
        subset = test[test.level == level]
        if subset.empty:
            continue
        indexes = subset.index.to_numpy()
        local_frame = subset.reset_index(drop=True)
        local_targets = [targets[i] for i in indexes]
        local_embed = embedding[indexes]
        local_control = control_embedding[indexes]
        result = retrieval(local_frame, local_embed, local_targets, local_control)
        if not result.empty:
            records.append(result)
    if not records:
        raise RuntimeError("No eligible count/volume-matched queries in grouped test set")
    result = pd.concat(records, ignore_index=True)
    result.to_csv(report_out / "retrieval_queries.csv", index=False)
    summary = result.groupby(["level", "method"]).agg(queries=("query", "size"),
        top1_accuracy=("top1_hit", "mean"), mean_rank_percentile=("rank_percentile", "mean"),
        median_candidates=("candidates", "median"), mean_target_distance=("target_distance", "mean")).reset_index()
    summary.to_csv(report_out / "retrieval_summary.csv", index=False)
    inference = {level: grouped_signflip(result, level) for level in ["single", "neighborhood", "organoid"]}
    (report_out / "retrieval_inference.json").write_text(json.dumps(inference, indent=2))
    split = json.loads((out / "split_audit.json").read_text())
    report = ["# Boundary-aware raw 3D shape experiment", "",
              f"Checkpoint epoch: {checkpoint['epoch']}. Test stacks: {split['split_stacks']['test']}. ",
              f"Maximum pixel-aligned cross-split correlation: {split['max_cross_split_pearson']:.3f}; biological independence remains unverified.", "",
              "Matched candidates have the same cell count, total foreground volume within 80–125%, and come from another test stack. An identical untrained raw encoder is included as a representation control.",
              "Single-cell targets are five standardized morphology descriptors from predicted masks; multicell targets are sorted, RMS-normalized pairwise cell-centroid distances.",
              "Thus the multicell score measures arrangement, not cell surface or contact topology. The paired uncertainty calculation clusters by image-similarity group.", "",
              "| Level | Method | Queries | Correct top match | Mean rank percentile |", "|---|---|---:|---:|---:|"]
    for row in summary.itertuples():
        report.append(f"| {row.level} | {row.method} | {row.queries} | {row.top1_accuracy:.3f} | {row.mean_rank_percentile:.3f} |")
    report += ["", "Paired group-level sign-flip tests:", "", "```json", json.dumps(inference, indent=2), "```", "",
               "This is an internal dataset benchmark. The 0.90 image-similarity grouping is a proxy for acquisition/specimen grouping and does not prove independent organoid generalization."]
    (report_out / "REPORT.md").write_text("\n".join(report) + "\n")
    (report_out / "status.json").write_text(json.dumps({"status": "complete", "checkpoint_epoch": checkpoint["epoch"],
                                                  "test_stacks": split["split_stacks"]["test"]}, indent=2))
    print(summary.to_string(index=False), flush=True)
    print("EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
