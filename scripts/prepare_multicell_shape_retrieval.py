"""Create aligned single-cell, local-neighborhood and complete-organoid raw/mask pairs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates

from prepare_raw_simclr3d import raw_crop

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/shape_retrieval_multicell.json").read_text())
OUT = ROOT / CFG["output_dir"]


def sample_labels(labels, center, spacing, fov, size=64):
    spacing = np.asarray(spacing, float); center = np.asarray(center, float)
    axes = [center[k] + ((np.arange(size)+.5)/size-.5)*fov/spacing[k] for k in range(3)]
    coords = np.meshgrid(*axes, indexing="ij")
    return map_coordinates(labels, coords, order=0, mode="constant", cval=0, prefilter=False)


def descriptors(mask, fov):
    points = np.argwhere(mask)
    pitch = fov / mask.shape[0]
    volume = len(points) * pitch**3
    if len(points) < 4:
        return [volume, 0, 0, 0, 0, 0, 0, 0]
    centered = (points-points.mean(0))*pitch
    eig = np.sort(np.linalg.eigvalsh(centered.T@centered/len(points)))[::-1].clip(1e-9)
    axes = np.sqrt(eig)
    extent = (points.max(0)-points.min(0)+1)*pitch
    bbox_volume = float(np.prod(extent))
    radius = float(np.sqrt((centered*centered).sum(1).mean()))
    return [volume, bbox_volume, radius, axes[0]/axes[1], axes[1]/axes[2],
            extent.max()/max(extent.min(), 1e-6), volume/max(bbox_volume, 1e-9), float(mask.mean())]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ROOT / "results/shape_retrieval/manifest.csv")
    base_masks = np.load(ROOT / "results/shape_retrieval/masks_fixed64.npy", mmap_mode="r")
    base_raw = np.load(ROOT / "results/simclr3d_raw/raw64.npy", mmap_mode="r")
    cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    native_spacing = np.asarray(json.loads((ROOT / "config/organoids.json").read_text())["spacing_zyx_um"], float)
    records, raw_volumes, mask_volumes = [], [], []

    # Preserve every validated single-cell example exactly.
    for i, row in base.iterrows():
        mask = np.asarray(base_masks[i], dtype=np.uint8)
        raw_volumes.append(np.asarray(base_raw[int(row["index"])], dtype=np.float16)); mask_volumes.append(mask)
        records.append(dict(sample=row["sample"], split=row["split"], level="single", fov_um=40.0,
                            member_labels=str(int(row.label)), member_count=1, anchor_label=int(row.label),
                            source_cell_id=row.cell_id, **{f"shape_{j}": v for j, v in enumerate(descriptors(mask, 40.0))}))

    for sample, group in base.groupby("sample", sort=True):
        label_path = Path(group.iloc[0].source_label_file); labels = tifffile.imread(label_path)
        info = json.loads((label_path.parent / "complete.json").read_text())
        effective = np.asarray(info["effective_spacing_zyx_um"], float)
        image = tifffile.imread(info["image"]).astype(np.float32)
        lo, hi = np.percentile(image, [1, 99.8]); normalized = np.clip((image-lo)/max(hi-lo, 1), 0, 1)
        centroids_um = np.stack([cells.loc[cid, ["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
                                 for cid in group.cell_id])
        labels_here = group.label.to_numpy(int)
        generated = set()
        # One local field per anchor, deduplicated by the cells actually visible in the target mask.
        for anchor, center_um in zip(labels_here, centroids_um):
            center_label = center_um/effective-.5
            sampled = sample_labels(labels, center_label, effective, 48.0)
            members = tuple(sorted(int(x) for x in np.unique(sampled) if x > 0))
            key = ("neighborhood", members)
            if len(members) < 2 or key in generated: continue
            generated.add(key); target = np.isin(sampled, members).astype(np.uint8)
            center_native = (center_um+.5*effective)/native_spacing-.5
            raw, _ = raw_crop(normalized, center_native, native_spacing, fov=48.0, size=64)
            raw_volumes.append(raw.astype(np.float16)); mask_volumes.append(target)
            records.append(dict(sample=sample, split=group.iloc[0]["split"], level="neighborhood", fov_um=48.0,
                                member_labels=",".join(map(str, members)), member_count=len(members), anchor_label=int(anchor),
                                source_cell_id="", **{f"shape_{j}": v for j, v in enumerate(descriptors(target, 48.0))}))
        # One complete-organoid field. A 64 um cube covers every observed centroid in this cohort.
        center_um = .5*(centroids_um.min(0)+centroids_um.max(0)); center_label = center_um/effective-.5
        sampled = sample_labels(labels, center_label, effective, 64.0)
        members = tuple(sorted(int(x) for x in np.unique(sampled) if x > 0)); target = (sampled > 0).astype(np.uint8)
        center_native = (center_um+.5*effective)/native_spacing-.5
        raw, _ = raw_crop(normalized, center_native, native_spacing, fov=64.0, size=64)
        raw_volumes.append(raw.astype(np.float16)); mask_volumes.append(target)
        records.append(dict(sample=sample, split=group.iloc[0]["split"], level="organoid", fov_um=64.0,
                            member_labels=",".join(map(str, members)), member_count=len(members), anchor_label=0,
                            source_cell_id="", **{f"shape_{j}": v for j, v in enumerate(descriptors(target, 64.0))}))
        print(f"Prepared {sample}: {len(generated)} neighborhoods + organoid", flush=True)

    raw_array = np.lib.format.open_memmap(OUT/"raw64.npy", mode="w+", dtype=np.float16,
                                          shape=(len(records),64,64,64))
    mask_array = np.lib.format.open_memmap(OUT/"masks64.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(records),64,64,64))
    for i,(raw,mask) in enumerate(zip(raw_volumes,mask_volumes)): raw_array[i]=raw;mask_array[i]=mask
    raw_array.flush();mask_array.flush()
    manifest = pd.DataFrame(records); manifest.index.name="index";manifest.to_csv(OUT/"manifest.csv")
    summary = manifest.groupby(["split","level"]).agg(examples=("sample","size"),stacks=("sample","nunique"),
                  median_cells=("member_count","median"),min_cells=("member_count","min"),max_cells=("member_count","max")).reset_index()
    summary.to_csv(OUT/"dataset_summary.csv",index=False)
    (OUT/"dataset.json").write_text(json.dumps({"examples":len(manifest),"levels":manifest.level.value_counts().to_dict(),
        "principle":"single targets plus all segmented cells visible in local fields and complete-organoid fields",
        "ambiguity_control":"no identical raw crop is assigned conflicting cell subsets","raw_dtype":"float16",
        "split":"inherits the previous organoid-level train/validation/test assignment"},indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
