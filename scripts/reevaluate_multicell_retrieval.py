"""Count/volume-matched 3D arrangement retrieval with visual audit cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/shape_retrieval_multicell"
OUT = SOURCE / "matched_retrieval"
VOLUME_TOLERANCE = 0.25


def sampled_labels(row, source_paths, centroids, size=64):
    path = source_paths[row["sample"]]
    labels = tifffile.imread(path)
    info = json.loads((path.parent / "complete.json").read_text())
    spacing = np.asarray(info["effective_spacing_zyx_um"], float)
    sample_centers = centroids[row["sample"]]
    if row["level"] == "organoid":
        all_centers = np.stack(list(sample_centers.values()))
        center_um = .5 * (all_centers.min(axis=0) + all_centers.max(axis=0))
    else:
        center_um = sample_centers[int(row["anchor_label"])]
    center = center_um / spacing - .5
    fov = float(row["fov_um"])
    axes = [center[k] + ((np.arange(size) + .5) / size - .5) * fov / spacing[k] for k in range(3)]
    coords = np.meshgrid(*axes, indexing="ij")
    return map_coordinates(labels, coords, order=0, mode="constant", cval=0, prefilter=False)


def arrangement_vector(labels):
    counts = np.bincount(labels.ravel().astype(np.int64))
    ids = np.flatnonzero(counts[1:]) + 1
    centers = []
    for label in ids:
        points = np.argwhere(labels == label)
        centers.append(points.mean(0))
    centers = np.asarray(centers)
    distances = pairwise_distances(centers)
    upper = np.sort(distances[np.triu_indices(len(centers), 1)])
    rms = np.sqrt(np.mean(upper ** 2))
    return upper / rms, len(ids), int(sum(counts[1:]))


def candidate_rows(frame, q, tolerance=VOLUME_TOLERANCE):
    count = int(frame.iloc[q].member_count)
    volume = float(frame.iloc[q].shape_0)
    return np.flatnonzero((frame["sample"].to_numpy() != frame.iloc[q]["sample"]) &
                          (frame.member_count.to_numpy() == count) &
                          (np.abs(np.log(frame.shape_0.to_numpy() / volume)) <= np.log1p(tolerance)))


def evaluate_level(frame, embeddings, vectors, min_candidates=3):
    model_dist = pairwise_distances(embeddings, metric="cosine")
    n = len(frame)
    rows = []
    for q in range(n):
        candidates = candidate_rows(frame, q)
        if len(candidates) < min_candidates:
            continue
        oracle = np.asarray([np.linalg.norm(vectors[q] - vectors[j]) for j in candidates])
        learned = model_dist[q, candidates]
        volume = np.abs(np.log(frame.iloc[candidates].shape_0.to_numpy() / float(frame.iloc[q].shape_0)))
        truth_order = np.argsort(oracle)
        if np.ptp(oracle) < 1e-12:
            continue
        for method, distance in [("raw_student", learned), ("volume_baseline", volume)]:
            best = int(np.argmin(distance))
            rank = int(np.flatnonzero(truth_order == best)[0])
            rows.append(dict(level=frame.iloc[q].level, query=frame.iloc[q]["index"],
                             sample=frame.iloc[q]["sample"], member_count=int(frame.iloc[q].member_count),
                             candidates=len(candidates), method=method,
                             top1_hit=int(best == truth_order[0]), rank_percentile=1 - rank / (len(candidates)-1),
                             top1_arrangement_distance=float(oracle[best]),
                             candidate_spearman=float(spearmanr(distance, oracle).statistic) if len(candidates) >= 3 and np.ptp(distance) > 1e-12 else np.nan,
                             retrieved=frame.iloc[candidates[best]]["index"],
                             oracle=frame.iloc[candidates[truth_order[0]]]["index"]))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SOURCE / "manifest.csv", index_col="index")
    test = manifest.loc[manifest.split == "test"].copy()
    test_ids = test.index.to_numpy()
    embeddings = np.load(SOURCE / "test_student_embeddings.npy")
    original = pd.read_csv(ROOT / "results/shape_retrieval/manifest.csv")
    source_paths = {s: Path(g.iloc[0].source_label_file) for s, g in original.groupby("sample")}
    original_cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    center_lookup = {sample: {int(row.label): original_cells.loc[row.cell_id,
                     ["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
                     for _, row in group.iterrows()} for sample, group in original.groupby("sample")}
    stored = np.load(SOURCE / "masks64.npy", mmap_mode="r")
    vectors = {}
    checks = []
    for idx, row in test.iterrows():
        if row.level not in {"neighborhood", "organoid"}:
            continue
        volume = sampled_labels(row, source_paths, center_lookup)
        union = (volume > 0) | (np.asarray(stored[idx]) > 0)
        iou = float(((volume > 0) & (np.asarray(stored[idx]) > 0)).sum() / max(union.sum(), 1))
        if iou < .999:
            raise RuntimeError(f"Reconstructed target differs from trained mask at {idx}: IoU={iou:.4f}")
        vector, observed_count, foreground = arrangement_vector(volume)
        vectors[idx] = vector
        checks.append(dict(index=idx, sample=row["sample"], level=row.level, recorded_count=int(row.member_count),
                           sampled_count=observed_count, foreground_voxels=foreground,
                           count_agrees=observed_count == int(row.member_count), mask_iou=iou))
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / "target_integrity.csv", index=False)
    if not checks.count_agrees.all():
        raise RuntimeError(f"{(~checks.count_agrees).sum()} sampled label counts disagree with the manifest")
    results = []
    for level in ["neighborhood", "organoid"]:
        use = np.flatnonzero(test.level.to_numpy() == level)
        frame = test.iloc[use].reset_index()
        local_vectors = [vectors[i] for i in frame["index"]]
        result = evaluate_level(frame, embeddings[use], local_vectors)
        results.append(result)
    details = pd.concat(results, ignore_index=True)
    details.to_csv(OUT / "query_results.csv", index=False)
    summary = details.groupby(["level", "method"]).agg(queries=("query", "size"),
        top1_accuracy=("top1_hit", "mean"), mean_rank_percentile=("rank_percentile", "mean"),
        mean_candidate_spearman=("candidate_spearman", "mean"),
        median_candidates=("candidates", "median"), mean_top1_arrangement_distance=("top1_arrangement_distance", "mean")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    inference = {}
    rng = np.random.default_rng(20260923)
    for level, group in details.groupby("level"):
        paired = group.pivot(index=["query", "sample"], columns="method", values=["top1_hit", "rank_percentile"])
        sample_ids = paired.index.get_level_values("sample").to_numpy()
        unique_samples = np.unique(sample_ids)
        inference[level] = {}
        for metric in ["top1_hit", "rank_percentile"]:
            delta = (paired[metric]["raw_student"] - paired[metric]["volume_baseline"]).to_numpy()
            sample_delta = {s: delta[sample_ids == s] for s in unique_samples}
            boot = []
            for _ in range(10000):
                draw = rng.choice(unique_samples, size=len(unique_samples), replace=True)
                boot.append(float(np.mean(np.concatenate([sample_delta[s] for s in draw]))))
            # A paired organoid-level sign-flip test keeps correlated queries together.
            permuted = []
            for _ in range(10000):
                signs = rng.choice([-1, 1], size=len(unique_samples))
                permuted.append(float(np.mean(np.concatenate([sample_delta[s] * signs[j] for j, s in enumerate(unique_samples)]))))
            observed = float(np.mean(delta))
            inference[level][metric] = {"student_minus_volume_baseline": observed,
                "cluster_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
                "paired_sample_signflip_p_two_sided": float((1 + sum(abs(v) >= abs(observed) for v in permuted)) / 10001),
                "independent_query_organoids": len(unique_samples)}
    (OUT / "paired_inference.json").write_text(json.dumps(inference, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(inference, indent=2))


if __name__ == "__main__":
    main()
