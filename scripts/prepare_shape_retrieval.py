"""Build raw/mask pairs on one fixed physical grid and an organoid-level 70/15/15 split."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/shape_retrieval.json").read_text())
OUT = ROOT / CFG["output_dir"]
FEATURES = ["volume_um3", "surface_area_um2", "sphericity", "aspect_ratio",
            "elongation", "flatness", "mesh_solidity", "surface_to_volume_per_um"]


def fixed_mask_crop(labels, label, spacing, fov, size):
    binary = labels == label
    if not binary.any():
        raise ValueError(f"Missing label {label}")
    center = np.asarray(center_of_mass(binary), dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    axes = [center[k] + ((np.arange(size) + .5) / size - .5) * fov / spacing[k] for k in range(3)]
    coords = np.meshgrid(*axes, indexing="ij")
    return (map_coordinates(binary.astype(np.uint8), coords, order=0, mode="constant", cval=0,
                            prefilter=False) > 0).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(ROOT / "results/simclr3d_raw/manifest.csv")
    cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    keep = old.shape_qc_pass.astype(bool).to_numpy(copy=True)
    target_voxel_volume = (CFG["physical_fov_um"] / CFG["input_size"]) ** 3
    keep &= old.cell_id.map(cells["volume_um3"]).to_numpy() >= 4 * target_voxel_volume
    manifest = old.loc[keep].copy().reset_index(drop=True)
    samples = np.array(sorted(manifest["sample"].unique()))
    rng = np.random.default_rng(CFG["seed"]); rng.shuffle(samples)
    n_train = int(round(len(samples) * CFG["train_stack_fraction"]))
    n_val = int(round(len(samples) * CFG["validation_stack_fraction"]))
    split_map = {s: "train" for s in samples[:n_train]}
    split_map.update({s: "validation" for s in samples[n_train:n_train+n_val]})
    split_map.update({s: "test" for s in samples[n_train+n_val:]})
    manifest["split"] = manifest["sample"].map(split_map)
    manifest["paired_index"] = np.arange(len(manifest))
    for feature in FEATURES:
        manifest[feature] = manifest.cell_id.map(cells[feature])

    masks = np.lib.format.open_memmap(OUT / "masks_fixed64.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(manifest), 64, 64, 64))
    hashes = {}
    for sample, group in manifest.groupby("sample", sort=True):
        label_path = Path(group.iloc[0].source_label_file)
        labels = tifffile.imread(label_path)
        complete = json.loads((label_path.parent / "complete.json").read_text())
        spacing = complete["effective_spacing_zyx_um"]
        hashes[sample] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        for idx, row in group.iterrows():
            masks[idx] = fixed_mask_crop(labels, int(row.label), spacing,
                                          CFG["physical_fov_um"], CFG["input_size"])
        print(f"Prepared {sample}: {len(group)} cells", flush=True)
    masks.flush()
    manifest["fixed_mask_foreground_voxels"] = np.asarray(masks).reshape(len(masks), -1).sum(1)
    manifest.to_csv(OUT / "manifest.csv", index=False)
    split_samples = {key: sorted([s for s, value in split_map.items() if value == key])
                     for key in ["train", "validation", "test"]}
    metadata = {
        "objects": len(manifest), "stacks": len(samples),
        "split_cell_counts": manifest.split.value_counts().to_dict(),
        "split_stack_counts": {k: len(v) for k, v in split_samples.items()},
        "split_samples": split_samples,
        "test_access_policy": "training code rejects test rows; only evaluation loads them",
        "geometry": "raw and target masks sampled on aligned 40 um, 64^3 physical fields of view",
        "voxel_pitch_um": CFG["physical_fov_um"] / CFG["input_size"],
        "spacing_status": "uses project-configured spacing; original TIFF spacing remains unconfirmed",
        "mask_source": "CPSAM-v2 predicted masks used as privileged training targets",
        "resolution_filter": f"shape-QC cells with volume >= four target voxels ({4*target_voxel_volume:.6f} um3)",
        "raw_source": "results/simclr3d_raw/raw64.npy addressed by the manifest index; segmentation supplied centers during dataset creation"
    }
    (OUT / "dataset.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ["objects", "stacks", "split_cell_counts", "split_stack_counts"]}, indent=2))


if __name__ == "__main__": main()
