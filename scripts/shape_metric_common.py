"""Mask-derived targets and matched retrieval for raw-only shape embeddings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["sphericity", "aspect_ratio", "elongation", "flatness", "mesh_solidity"]


def make_targets(dataset, output):
    frame = pd.read_csv(dataset / "manifest.csv").sort_values("index")
    if not np.array_equal(frame["index"].to_numpy(), np.arange(len(frame))):
        raise ValueError("Manifest indices must be contiguous and ordered")
    cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    labels = np.load(dataset / "labels96.npy", mmap_mode="r")
    geometry = np.load(dataset / "geometry.npy", mmap_mode="r")
    train_ids = frame[(frame.split == "train") & (frame.level == "single")].source_cell_id.unique()
    train_features = cells.loc[train_ids, FEATURES].to_numpy(float)
    mean = train_features.mean(0)
    scale = train_features.std(0).clip(1e-6)
    if not np.isfinite(train_features).all():
        raise ValueError("Nonfinite training morphology features")
    combined, shapes, arrangements = [], [], []
    for row in frame.itertuples():
        found = np.unique(labels[row.index])
        found = found[found > 0]
        if len(found) != row.member_count:
            raise ValueError(f"Count mismatch at field {row.index}")
        ids = [f"{row.sample}:L{int(label)}" for label in found]
        shape = (cells.loc[ids, FEATURES].to_numpy(float) - mean) / scale
        if not np.isfinite(shape).all():
            raise ValueError(f"Nonfinite morphology target at field {row.index}")
        # Sort each descriptor independently to make the cell-set target permutation invariant.
        shape = np.sort(shape, axis=0).T.ravel().astype(np.float32)
        shape = shape / np.sqrt(len(shape))
        count = int(row.member_count)
        n_pairs = count * (count - 1) // 2
        arrangement = np.array(geometry[row.index, :n_pairs], dtype=np.float32)
        if n_pairs:
            arrangement = arrangement / np.sqrt(n_pairs)
            target = np.concatenate([shape / np.sqrt(2), arrangement / np.sqrt(2)])
        else:
            target = shape
        shapes.append(shape)
        arrangements.append(arrangement)
        combined.append(target)
    max_length = max(map(len, combined))
    array = np.full((len(frame), max_length), np.nan, np.float32)
    for i, vector in enumerate(combined):
        array[i, :len(vector)] = vector
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "metric_targets.npy", array)
    pd.DataFrame({"index": frame["index"], "target_length": list(map(len, combined)),
                  "shape_length": list(map(len, shapes)),
                  "arrangement_length": list(map(len, arrangements))}).to_csv(output / "target_lengths.csv", index=False)
    (output / "target_metadata.json").write_text(json.dumps({
        "features": FEATURES, "train_mean": mean.tolist(), "train_std": scale.tolist(),
        "source": "predicted instance masks only; reference annotations excluded",
        "multicell_target": "equal-weight, permutation-invariant sorted per-cell morphology and normalized centroid distances"
    }, indent=2))
    return frame, shapes, arrangements, combined


def matched_retrieval(frame, embeddings, targets, methods, min_candidates=3):
    """Evaluate each representation against shape, arrangement and combined targets."""
    distances = {name: pairwise_distances(vector, metric="cosine") for name, vector in embeddings.items()}
    result = []
    frame = frame.reset_index(drop=True)
    for level in ["single", "neighborhood", "organoid"]:
        local = frame[frame.level == level]
        for i in local.index:
            query = frame.loc[i]
            candidates = local.index[(local.member_count == query.member_count) &
                (local["similarity_group"] != query.similarity_group) &
                (np.abs(np.log(local.volume_um3 / query.volume_um3)) <= np.log(1.25))].to_numpy()
            if len(candidates) < min_candidates:
                continue
            for target_name in (["shape"] if level == "single" else ["shape", "arrangement", "combined"]):
                target = targets[target_name]
                truth = np.array([np.linalg.norm(target[i] - target[j]) for j in candidates])
                if np.ptp(truth) < 1e-8:
                    continue
                optimum = truth.min()
                for name in methods:
                    if name == "volume_only":
                        choice = candidates[np.argmin(np.abs(np.log(frame.loc[candidates, "volume_um3"] / query.volume_um3)))]
                    else:
                        choice = candidates[np.argmin(distances[name][i, candidates])]
                    found_distance = float(np.linalg.norm(target[i] - target[choice]))
                    result.append({"level": level, "target": target_name, "query": int(query["index"]),
                        "sample": query["sample"], "similarity_group": int(query.similarity_group),
                        "member_count": int(query.member_count), "candidates": int(len(candidates)),
                        "method": name, "top1_hit": float(found_distance <= optimum + 1e-8),
                        "rank_percentile": float(np.mean(truth >= found_distance)),
                        "target_distance": found_distance, "retrieved": int(frame.loc[choice, "index"])})
    return pd.DataFrame(result)
