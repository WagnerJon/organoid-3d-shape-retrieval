"""Prepare cell-isolated raw intensities using predicted masks only as gates."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/simclr3d_raw_isolated"


def isolate(raw, labels, label, size=64, padding=1.15):
    mask = labels == label
    bounds = ndimage.find_objects(mask.astype(np.uint8))[0]
    cell_mask = mask[bounds]
    cell_raw = raw[bounds] * cell_mask
    side = int(np.ceil(np.linalg.norm(cell_mask.shape) * padding))
    pads = [((side-n)//2, side-n-(side-n)//2) for n in cell_mask.shape]
    raw_cube = np.pad(cell_raw, pads)
    mask_cube = np.pad(cell_mask, pads)
    raw64 = resize(raw_cube, (size,)*3, order=1, preserve_range=True,
                   anti_aliasing=side > size).astype(np.float32)
    mask64 = resize(mask_cube.astype(np.float32), (size,)*3, order=0,
                    preserve_range=True, anti_aliasing=False) >= .5
    raw64 *= mask64
    if not mask64.any() or not np.isfinite(raw64).all():
        raise ValueError(f"Invalid isolated cell {label}")
    return raw64, mask64, side


def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / "config/simclr3d_raw_isolated.json").read_text())
    manifest = pd.read_csv(ROOT / "results/simclr3d/manifest.csv")
    output = np.lib.format.open_memmap(
        OUT / "isolated_raw64.npy", mode="w+", dtype=np.float32,
        shape=(len(manifest), cfg["input_size"], cfg["input_size"], cfg["input_size"]))
    rows = []
    for sample, group in manifest.groupby("sample", sort=False):
        folder = Path(group.iloc[0].source_label_file).parent
        info = json.loads((folder / "complete.json").read_text())
        image_path = Path(info["image"])
        native = tifffile.imread(image_path).astype(np.float32)
        labels = tifffile.imread(folder / "labels_isotropic.tif")
        lo, hi = np.percentile(native, [1, 99.8])
        normalized = np.clip((native-lo)/(hi-lo), 0, 1)
        # Match the exact working grid used by the segmentation and binary model.
        isotropic_raw = resize(normalized, labels.shape, order=1,
                               preserve_range=True, anti_aliasing=True).astype(np.float32)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        for idx, row in group.iterrows():
            raw64, mask64, side = isolate(
                isotropic_raw, labels, int(row.label), cfg["input_size"],
                cfg["rotation_safe_padding"])
            output[idx] = raw64
            values = raw64[mask64]
            q = np.percentile(values, [10, 50, 90, 99])
            rows.append(dict(**row.to_dict(), raw_image=str(image_path),
                raw_sha256=digest, isolation="predicted target mask gate",
                isolated_foreground_voxels=int(mask64.sum()),
                isolated_cube_side_voxels=side, intensity_mean=float(values.mean()),
                intensity_std=float(values.std()), intensity_p10=float(q[0]),
                intensity_p50=float(q[1]), intensity_p90=float(q[2]),
                intensity_p99=float(q[3])))
        print(f"Prepared {sample}: {group.index.max()+1}/{len(manifest)}", flush=True)
    output.flush()
    pd.DataFrame(rows).sort_values("index").to_csv(OUT / "manifest.csv", index=False)
    (OUT / "dataset.json").write_text(json.dumps(dict(
        objects=len(manifest), source="raw TIFF intensities gated by each predicted cell mask",
        mask_supplied_as_input_channel=False, geometry="same adaptive isotropic cube as binary model",
        intensity_normalization="per-stack percentiles 1 and 99.8, clipped to [0,1]",
        splits="identical to binary and context-rich raw models",
        interpretation="segmentation-guided cell representation; not segmentation-free",
        spacing_confirmed_from_tiff_metadata=False), indent=2))


if __name__ == "__main__":
    main()
