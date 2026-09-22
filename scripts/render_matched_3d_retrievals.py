"""Render actual raw projections and colored 3D cell meshes for matched retrieval cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import trimesh

from reevaluate_multicell_retrieval import ROOT, SOURCE, OUT, sampled_labels


def source_data():
    old = pd.read_csv(ROOT / "results/shape_retrieval/manifest.csv")
    paths = {s: Path(g.iloc[0].source_label_file) for s, g in old.groupby("sample")}
    cells = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id")
    centers = {sample: {int(r.label): cells.loc[r.cell_id,
               ["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
               for _, r in g.iterrows()} for sample, g in old.groupby("sample")}
    return paths, centers


def meshes_from_labels(labels, fov):
    pitch = fov / labels.shape[0]
    palette = plt.get_cmap("tab20")
    meshes = []
    for j, label in enumerate(x for x in np.unique(labels) if x > 0):
        binary = (labels == label).astype(np.uint8)
        if binary.sum() < 8:
            continue
        padded = np.pad(binary, 1)
        try:
            verts, faces, _, _ = marching_cubes(padded, .5, step_size=2)
        except RuntimeError:
            verts, faces, _, _ = marching_cubes(padded, .5, step_size=1)
        xyz = (verts[:, ::-1] - 1) * pitch
        mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
        color = np.asarray(palette(j % 20)[:3])
        mesh.visual.face_colors = np.array([*(color * 255).astype(np.uint8), 225], dtype=np.uint8)
        meshes.append((mesh, color))
    return meshes


def render_case(query, details, manifest, raw, paths, centers, title, filename):
    q = details[(details["query"] == query) & (details.method == "raw_student")].iloc[0]
    v = details[(details["query"] == query) & (details.method == "volume_baseline")].iloc[0]
    ids = [int(query), int(q.retrieved), int(v.retrieved)]
    roles = ["Query", "Raw model's nearest", "Volume baseline's nearest"]
    fig = plt.figure(figsize=(12.2, 10.5))
    for col, (index, role) in enumerate(zip(ids, roles)):
        row = manifest.loc[index]
        labels = sampled_labels(row, paths, centers)
        fov = float(row.fov_um)
        mesh_data = meshes_from_labels(labels, fov)
        scene = trimesh.Scene()
        for j, (mesh, _) in enumerate(mesh_data): scene.add_geometry(mesh, node_name=f"cell_{j+1:02d}")
        glb = OUT / f"{filename}_{['query','model','volume'][col]}.glb"
        scene.export(glb)
        img = np.asarray(raw[index], dtype=np.float32)
        ax = fig.add_subplot(3, 3, col + 1)
        xy = img.max(axis=0)
        ax.imshow(xy, cmap="gray", vmin=0, vmax=max(float(np.quantile(xy, .995)), .01))
        ax.set_title(f"{role}: {row['sample']}\n{int(row.member_count)} cells, {row.shape_0:.0f} µm³", fontsize=10)
        ax.axis("off")
        for view in range(2):
            ax3 = fig.add_subplot(3, 3, (view + 1) * 3 + col + 1, projection="3d")
            for mesh, color in mesh_data:
                faces = np.asarray(mesh.faces)
                if len(faces) > 1500: faces = faces[::int(np.ceil(len(faces) / 1500))]
                triangles = np.asarray(mesh.vertices)[faces]
                surface = Poly3DCollection(triangles, facecolor=color, edgecolor="none", alpha=.82)
                ax3.add_collection3d(surface)
            ax3.set(xlim=(0, fov), ylim=(0, fov), zlim=(0, fov))
            ax3.set_box_aspect((1, 1, 1)); ax3.view_init(elev=23 if view == 0 else 70, azim=38 if view == 0 else -52)
            ax3.set_axis_off()
            ax3.set_title("3D cells: oblique" if view == 0 else "3D cells: upper view", fontsize=9)
    fig.suptitle(title + f"\nMatched candidates: {int(q.candidates)}; normalized layout distance: model {q.top1_arrangement_distance:.3f}, volume {v.top1_arrangement_distance:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / f"{filename}.png", dpi=180)
    plt.close(fig)
    return {"query": int(query), "level": manifest.loc[query,"level"], "query_sample": manifest.loc[query,"sample"],
            "model_index": int(q.retrieved), "volume_index": int(v.retrieved), "model_hit": bool(q.top1_hit),
            "volume_hit": bool(v.top1_hit), "figure": f"{filename}.png", "mesh_prefix": filename}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    details = pd.read_csv(OUT / "query_results.csv")
    manifest = pd.read_csv(SOURCE / "manifest.csv", index_col="index")
    raw = np.load(SOURCE / "raw64.npy", mmap_mode="r")
    paths, centers = source_data()
    # Prespecified illustrative classes: large and moderate gains, a failure, and a whole organoid.
    cases = [(1264, "Neighborhood: large model gain", "case_neighborhood_large_gain"),
             (1097, "Neighborhood: five-cell model gain", "case_neighborhood_five_cells"),
             (1098, "Neighborhood: model failure", "case_neighborhood_failure"),
             (1060, "Whole stack field: model gain", "case_whole_organoid")]
    index = [render_case(q, details, manifest, raw, paths, centers, title, filename) for q, title, filename in cases]
    (OUT / "visual_cases.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))


if __name__ == "__main__": main()
