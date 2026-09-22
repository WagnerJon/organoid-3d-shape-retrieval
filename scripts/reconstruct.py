"""Batch Cellpose orthoview inference, u-Segment3D aggregation and mesh export."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(ROOT / ".cache/cellpose"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
from skimage.transform import resize
import trimesh


def pairs(config):
    root = Path(os.environ.get("ORGANOID_DATA_DIR", config["data_dir"]))
    if not root.is_absolute():
        root = ROOT / root
    images = sorted((root / "images").glob("X*.tif"))
    if not images:
        raise ValueError(f"No X*.tif images found in {root}")
    result = []
    for image in images:
        mask = root / "masks" / ("Y" + image.name[1:])
        if not mask.exists():
            raise ValueError(f"Missing mask: {mask}")
        result.append((image, mask))
    return result


def read_pair(image, mask):
    raw, labels = tifffile.imread(image), tifffile.imread(mask)
    if raw.ndim != 3 or raw.shape != labels.shape:
        raise ValueError(f"Expected matching ZYX volumes: {image}, {raw.shape}, {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError("Masks must contain nonnegative integer labels")
    return raw, labels


def export(labels, spacing, folder):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in measure.regionprops(labels):
        binary = np.pad(region.image, 1)
        vertices, faces, _, _ = measure.marching_cubes(binary, level=0.5, spacing=spacing)
        vertices += (np.array(region.bbox[:3]) - 1) * spacing
        # Mesh consumers use XYZ; swapping axes reverses handedness.
        mesh = trimesh.Trimesh(vertices=vertices[:, ::-1], faces=faces[:, ::-1], process=False)
        mesh.fix_normals(multibody=True)
        mesh.export(folder / f"cell_{region.label:04d}.ply")
        rows.append({"label": region.label, "volume_um3": float(region.area * np.prod(spacing)),
                     "surface_area_um2": float(mesh.area),
                     "touches_border": any(a == 0 for a in region.bbox[:3]) or
                     any(b == n for b, n in zip(region.bbox[3:], labels.shape))})
    with (folder / "cells.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "volume_um3", "surface_area_um2", "touches_border"])
        writer.writeheader()
        writer.writerows(rows)


def preview(raw, reference, prediction, path):
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    low, high = np.percentile(raw, [2, 99.8])
    normalized = np.clip((raw.astype(float) - low) / max(high - low, 1), 0, 1)
    for row, (name, labels) in enumerate([("Reference", reference), ("Prediction", prediction)]):
        for axis in range(3):
            index = raw.shape[axis] // 2
            axes[row, axis].imshow(mark_boundaries(np.take(normalized, index, axis), np.take(labels, index, axis)))
            axes[row, axis].set_title(f"{name}: {'XY XZ YZ'.split()[axis]} mid-slice")
            axes[row, axis].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(image, mask, config, folder):
    from segment3D import parameters, usegment3d
    import torch
    if config.get("require_mps") and not torch.backends.mps.is_available():
        raise RuntimeError("This run requires Apple MPS GPU access; run outside the sandbox")
    start = time.time()
    raw = tifffile.imread(image)
    if raw.ndim != 3:
        raise ValueError("Expected a ZYX image volume")
    spacing = np.array(config["spacing_zyx_um"])
    pre = parameters.get_preprocess_params()
    pre.update(factor=config["scale"], voxel_res=(spacing / spacing.min()).tolist(),
               do_bg_correction=config["background_correction"], bg_ds=8)
    processed = np.squeeze(usegment3d.preprocess_imgs(raw, pre))
    processed = ndimage.median_filter(processed, size=3)
    cp = parameters.get_Cellpose_autotune_params()
    cp.update(cellpose_modelname=config["model"], gpu=config["gpu"],
              best_diam=config["diameter"], debug_viz=False)
    probs, flows = [], []
    for view in ["xy", "xz", "yz"]:
        cache = folder / f"{view}.npz"
        if cache.exists():
            with np.load(cache) as saved:
                prob, flow = saved["prob"], saved["flow"]
        else:
            print(f"{image.stem}: Cellpose {view}", flush=True)
            _, prob, flow, _ = usegment3d.Cellpose2D_model_auto(processed[..., None], view, cp)
            np.savez_compressed(cache, prob=prob, flow=flow)
        probs.append(prob)
        flows.append(flow)
    agg = parameters.get_2D_to_3D_aggregation_params()
    agg["gradient_descent"]["gradient_decay"] = config["gradient_decay"]
    agg["postprocess_binary"]["binary_fill_holes"] = True
    labels, _ = usegment3d.aggregate_2D_to_3D_segmentation_direct_method(probs, flows, agg)
    labels = labels.astype(np.uint32)
    effective_spacing = spacing * np.array(raw.shape) / np.array(labels.shape)
    tifffile.imwrite(folder / "labels_isotropic.tif", labels, metadata={"axes": "ZYX", "spacing_zyx_um": effective_spacing.tolist()})
    native = resize(labels, raw.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
    tifffile.imwrite(folder / "labels_native.tif", native, metadata={"axes": "ZYX", "spacing_zyx_um": spacing.tolist()})
    export(labels, effective_spacing, folder / "meshes")
    # Load annotations only after image-only prediction and mesh export have finished.
    reference = tifffile.imread(mask)
    if reference.shape != native.shape:
        raise ValueError("Reference and prediction dimensions differ")
    preview(raw, reference, native, folder / "comparison.png")
    from evaluate import evaluate
    instance_metrics = evaluate(reference, native)
    (folder / "instance_metrics.json").write_text(json.dumps(instance_metrics, indent=2))
    # Reference masks are evaluation only: never used as Cellpose inputs.
    foreground_a, foreground_b = reference > 0, native > 0
    dice = 2 * np.count_nonzero(foreground_a & foreground_b) / max(int(foreground_a.sum() + foreground_b.sum()), 1)
    stats = {"image": str(image), "mask": str(mask), "seconds": time.time() - start,
             "reference_cells": int(len(np.unique(reference)) - (0 in reference)),
             "predicted_cells": int(len(np.unique(native)) - (0 in native)),
             "foreground_dice": dice, "note": "Foreground Dice does not measure instance separation.",
             "effective_spacing_zyx_um": effective_spacing.tolist()}
    stats["instance_metrics"] = instance_metrics
    (folder / "complete.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "reference", "predict"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/organoids.json")
    parser.add_argument("--sample", help="Example: X001")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spacing = np.array(config["spacing_zyx_um"], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) or not 0 < config["scale"] <= 1:
        raise ValueError("Positive finite ZYX spacing and scale in (0, 1] required")
    samples = pairs(config)
    if args.sample:
        samples = [(im, ma) for im, ma in samples if im.stem == args.sample]
        if not samples:
            raise ValueError(f"Unknown sample {args.sample}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        samples = samples[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for image, mask in samples:
        if args.command == "inventory":
            raw, labels = read_pair(image, mask)
            inventory.append({"sample": image.stem, "shape_zyx": list(raw.shape), "dtype": str(raw.dtype),
                              "cells": int(np.count_nonzero(np.unique(labels))), "image": str(image), "mask": str(mask)})
            continue
        if args.command == "reference":
            _, labels = read_pair(image, mask)
            export(labels, spacing, args.output / "reference" / image.stem)
            continue
        # Separate settings and source modification times prevent reuse of stale results.
        signature = dict(config=config, image_stat=[image.stat().st_size, image.stat().st_mtime_ns],
                         mask_stat=[mask.stat().st_size, mask.stat().st_mtime_ns],
                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
        folder = args.output / f"run_{key}" / image.stem
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "settings.json").write_text(json.dumps(signature, indent=2))
        if (folder / "complete.json").exists():
            print(f"Skipping completed {image.stem}", flush=True)
            continue
        run(image, mask, config, folder)
    if args.command == "inventory":
        (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
        print(f"Validated {len(inventory)} image/mask pairs, {sum(x['cells'] for x in inventory)} labeled cells")


if __name__ == "__main__":
    main()
