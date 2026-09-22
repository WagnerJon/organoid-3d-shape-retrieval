"""Render a calibrated 3D preview of an instance-label TIFF (XYZ in micrometers)."""
import argparse
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import tifffile
from skimage.measure import marching_cubes, regionprops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--title", default="Cell reconstruction")
    args = parser.parse_args()
    labels = tifffile.imread(args.labels)
    spacing = np.array(args.spacing)
    fig = plt.figure(figsize=(9, 8), facecolor="#101827")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101827")
    bounds = []
    for index, region in enumerate(regionprops(labels)):
        verts, faces, _, _ = marching_cubes(np.pad(region.image, 2), 0.5, spacing=spacing, step_size=2)
        verts += (np.array(region.bbox[:3]) - 2) * spacing
        verts = verts[:, ::-1]
        bounds.append(verts)
        poly = Poly3DCollection(verts[faces], alpha=0.85, linewidth=0)
        poly.set_facecolor(plt.get_cmap("tab10")(index % 10))
        ax.add_collection3d(poly)
    if not bounds:
        raise ValueError("No labeled cells")
    bounds = np.concatenate(bounds)
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=25, azim=35)
    ax.set_xlabel("X (µm)", color="white"); ax.set_ylabel("Y (µm)", color="white"); ax.set_zlabel("Z (µm)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(args.title, color="white", pad=24)
    fig.text(0.5, 0.04, "Each color is one cell • Dataset-based voxel calibration", ha="center", color="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
