"""Prepare group-disjoint, higher-resolution raw/instance/SDF fields."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import find_boundaries

from prepare_multicell_shape_retrieval import sample_labels
from prepare_raw_simclr3d import raw_crop

ROOT = Path(__file__).resolve().parents[1]


def stack_splits(source, threshold, fractions, seed):
    """Keep any connected component of near-duplicate stacks entirely in one split."""
    one = source.drop_duplicates("sample").sort_values("sample")
    samples = one["sample"].tolist()
    vectors = []
    for path in one["raw_image"]:
        image = tifffile.imread(path)[::4, ::8, ::8].astype(np.float32).ravel()
        vectors.append((image - image.mean()) / max(float(image.std()), 1e-6))
    vectors = np.stack(vectors)
    corr = vectors @ vectors.T / vectors.shape[1]
    np.fill_diagonal(corr, 1)
    n_groups, groups = connected_components(csr_matrix(corr >= threshold), directed=False)
    sizes = np.bincount(groups)
    rng = np.random.default_rng(seed)
    target = np.array(fractions) * len(samples)
    best = None
    # Largest near-duplicate component belongs to training. Search group allocations
    # for reasonably balanced validation/test sets without splitting components.
    largest = int(np.argmax(sizes))
    for _ in range(10000):
        assignment = rng.choice(3, n_groups, p=[0.56, 0.22, 0.22])
        assignment[largest] = 0
        counts = np.bincount(assignment[groups], minlength=3)
        if min(counts[1:]) < 12 or counts[0] < 50:
            continue
        score = np.sum(((counts - target) / target) ** 2)
        if best is None or score < best[0]:
            best = (score, assignment.copy(), counts.copy())
    if best is None:
        raise RuntimeError("Could not form three adequately sized similarity-disjoint splits")
    names = np.array(["train", "validation", "test"])
    split = names[best[1][groups]]
    cross = np.empty(len(samples), dtype=np.float32)
    for i in range(len(samples)):
        cross[i] = corr[i, split != split[i]].max()
    table = pd.DataFrame({"sample": samples, "similarity_group": groups,
                          "split": split, "max_cross_split_pearson": cross})
    audit = {"threshold": float(threshold), "groups": int(n_groups),
             "largest_group_stacks": int(sizes.max()),
             "split_stacks": {name: int(n) for name, n in table.split.value_counts().items()},
             "max_cross_split_pearson": float(cross.max()),
             "median_cross_split_pearson": float(np.median(cross)),
             "comparison": "Pixel-aligned TIFF Pearson after [4,8,8] downsampling; grouping is a proxy, not biological identity"}
    if audit["max_cross_split_pearson"] >= threshold + 1e-5:
        raise AssertionError("Near-duplicate pair crossed splits")
    return table, audit


def sdf_target(labels, pitch_um, clip_um):
    """Signed distance to the nearest *instance* boundary in physical micrometers."""
    boundary = find_boundaries(labels, connectivity=1, mode="inner")
    if not boundary.any():
        raise ValueError("No instance boundary in field")
    distance = ndimage.distance_transform_edt(~boundary).astype(np.float32) * pitch_um
    signed = np.where(labels > 0, distance, -distance)
    return np.clip(signed / clip_um, -1, 1).astype(np.float16)


def geometry(labels, pitch_um):
    """Size-normalized arrangement plus per-cell size/axis variation and contacts."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    centers, volumes, aspects = [], [], []
    for label in ids:
        points = np.argwhere(labels == label).astype(np.float32)
        centers.append(points.mean(0))
        volumes.append(len(points))
        if len(points) < 4:
            aspects.append(1.0)
        else:
            centered = points - points.mean(0)
            eigen = np.linalg.eigvalsh(centered.T @ centered / len(points)).clip(1e-6)
            aspects.append(float(np.sqrt(eigen[-1] / eigen[0])))
    if not centers:
        raise ValueError("Field contains no substantial cell")
    centers = np.stack(centers)
    if len(centers) > 1:
        delta = centers[:, None] - centers[None, :]
        distances = np.sort(np.linalg.norm(delta, axis=-1)[np.triu_indices(len(centers), 1)])
        distances = distances / max(float(np.sqrt(np.mean(distances ** 2))), 1e-6)
    else:
        distances = np.array([], dtype=np.float32)
    rel_volumes = np.sort(np.asarray(volumes) / np.mean(volumes))
    axis_ratios = np.sort(np.asarray(aspects))
    return np.concatenate([distances, rel_volumes, axis_ratios]).astype(np.float32), len(centers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/boundary_shape.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ROOT / "results/shape_retrieval/manifest.csv")
    previous = pd.read_csv(ROOT / "results/shape_retrieval_multicell/manifest.csv")
    groups, audit = stack_splits(source, cfg["similarity_threshold"], cfg["split_fractions"], cfg["seed"])
    groups.to_csv(out / "split_groups.csv", index=False)
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))
    print("SPLIT", json.dumps(audit), flush=True)
    split_for = groups.set_index("sample")["split"].to_dict()
    metadata = source.drop_duplicates("sample").set_index("sample")
    cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    native = np.asarray(json.loads((ROOT / "config/organoids.json").read_text())["spacing_zyx_um"], float)
    size = int(cfg["input_size"])
    count = len(previous)
    raw_mem = np.lib.format.open_memmap(out / "raw96.npy", mode="w+", dtype=np.float16, shape=(count, size, size, size))
    labels_mem = np.lib.format.open_memmap(out / "labels96.npy", mode="w+", dtype=np.uint8, shape=(count, size, size, size))
    sdf_mem = np.lib.format.open_memmap(out / "sdf96.npy", mode="w+", dtype=np.float16, shape=(count, size, size, size))
    rows, geoms = [], {}
    for sample, sample_rows in previous.groupby("sample", sort=True):
        item = metadata.loc[sample]
        label_path = Path(item.source_label_file)
        labels = tifffile.imread(label_path)
        info = json.loads((label_path.parent / "complete.json").read_text())
        effective = np.asarray(info["effective_spacing_zyx_um"], float)
        image = tifffile.imread(item.raw_image).astype(np.float32)
        low, high = np.percentile(image, [1, 99.8])
        image = np.clip((image - low) / max(high - low, 1), 0, 1)
        sample_cells = source[source["sample"] == sample]
        all_centers = np.stack([cells.loc[cid, ["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
                                for cid in sample_cells.cell_id])
        for idx, row in sample_rows.iterrows():
            fov = float(row.fov_um)
            if row.level == "organoid":
                center_um = .5 * (all_centers.min(0) + all_centers.max(0))
            else:
                cid = f"{sample}:L{int(row.anchor_label)}"
                center_um = cells.loc[cid, ["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
            label_center = center_um / effective - .5
            sampled = sample_labels(labels, label_center, effective, fov, size=size)
            allowed = set(map(int, str(row.member_labels).split(",")))
            sampled[~np.isin(sampled, list(allowed))] = 0
            if row.level == "single":
                sampled[sampled != int(row.anchor_label)] = 0
            if sampled.max() >= 256:
                raise ValueError("Instance label exceeds uint8 capacity")
            sampled = sampled.astype(np.uint8)
            center_native = (center_um + .5 * effective) / native - .5
            raw, _ = raw_crop(image, center_native, native, fov=fov, size=size)
            raw_mem[idx] = raw.astype(np.float16)
            labels_mem[idx] = sampled
            sdf_mem[idx] = sdf_target(sampled, fov / size, cfg["sdf_clip_um"])
            geom, actual_count = geometry(sampled, fov / size)
            geoms[idx] = geom
            rows.append({"index": idx, "sample": sample, "split": split_for[sample], "level": row.level,
                         "fov_um": fov, "member_count": actual_count, "foreground_voxels": int((sampled > 0).sum()),
                         "volume_um3": float((sampled > 0).sum() * (fov / size) ** 3),
                         "source_cell_id": row.source_cell_id if isinstance(row.source_cell_id, str) else "",
                         "geometry_length": len(geom)})
        print(f"PREPARED {sample}: {len(sample_rows)} fields", flush=True)
    raw_mem.flush(); labels_mem.flush(); sdf_mem.flush()
    max_length = max(map(len, geoms.values()))
    geometry_mem = np.lib.format.open_memmap(out / "geometry.npy", mode="w+", dtype=np.float32,
                                             shape=(count, max_length))
    geometry_mem[:] = np.nan
    for idx, geom in geoms.items():
        geometry_mem[idx, :len(geom)] = geom
    geometry_mem.flush()
    manifest = pd.DataFrame(rows).sort_values("index")
    manifest.to_csv(out / "manifest.csv", index=False)
    summary = manifest.groupby(["split", "level"]).agg(fields=("index", "size"), stacks=("sample", "nunique"),
                                                        min_cells=("member_count", "min"),
                                                        max_cells=("member_count", "max")).reset_index()
    summary.to_csv(out / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (out / "prepare_complete.json").write_text(json.dumps({"fields": count, "shape": [size] * 3,
        "split_audit": audit, "raw_only_inference": True,
        "targets": "predicted-mask-derived instance boundary signed distance; supplied annotations not used"}, indent=2))


if __name__ == "__main__":
    main()
