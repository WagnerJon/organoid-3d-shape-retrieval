"""Frozen target-dataset shape retrieval comparison and published ZMPY3D adapter."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys, json, hashlib, pickle
import numpy as np
import pandas as pd
import igl, trimesh
from scipy.special import expit
from scipy.spatial.distance import cdist
from spectral64_core import ROOT, sha, save_json
from spatial_benchmark_core import surface_frame
from prepare_pointnet_shape import read_mesh
from pointnet_shape_core import sample_surface

JOINT = Path('/Volumes/CHUNKY/Scientific_data/GFP-24hrs-2ndy-cleared-MV_all5_analysis')
ATLAS = JOINT / 'atlas_dataset'
OUT = ROOT / 'results/five_organoid_shape_retrieval'
ZMPY = ROOT / '.cache/zmpy3d-reference'
ORDER = 20
RESOLUTION = 64
METHODS = ['morph_all', 'spatial_robust', 'spherical_harmonics', 'spherical_harmonics_adjusted',
           'point_mae_pretrained_pose_average_raw', 'point_mae_adapted_pose_average_raw',
           'allen_pretrained_raw', 'allen_head_decoder_raw', 'shapeembed3d_invariant_raw',
           'shapeembed3d_cnn_raw', 'cajal_qgw', 'zernike20']
PRIMARY = ['spherical_harmonics', 'zernike20', 'point_mae_adapted_pose_average_raw', 'spatial_robust']
NAMES = {'morph_all': 'Nine traditional features', 'spatial_robust': 'Spatial PCA + ambiguity fallback',
         'spherical_harmonics': 'Raw multi-shell harmonics', 'spherical_harmonics_adjusted': 'Adjusted shell harmonics',
         'point_mae_pretrained_pose_average_raw': 'Point-MAE: pretrained, pose averaged',
         'point_mae_adapted_pose_average_raw': 'Point-MAE: adapted, pose averaged',
         'allen_pretrained_raw': 'Allen VN-DGCNN: pretrained', 'allen_head_decoder_raw': 'Allen VN-DGCNN: adapted',
         'shapeembed3d_invariant_raw': 'ShapeEmbedLite3D: invariant', 'shapeembed3d_cnn_raw': 'ShapeEmbedLite3D: CNN',
         'cajal_qgw': 'CAJAL: quantized GW', 'zernike20': '3D Zernike: order20 (ZMPY3D)'}


def source():
    return json.loads((ATLAS / 'dataset.json').read_text())


def mesh(row):
    path = ATLAS / row['sample'] / f"{row['label']}.mesh"
    if sha(path) != row['source_mesh_sha256']: raise ValueError('Source geometry changed')
    v, f = read_mesh(path)
    return v.astype(float), f.astype(np.int64)


def normalize(v, f):
    center, _, values, _ = surface_frame(v, f)
    radius = float(np.sqrt(values.clip(0).sum()))
    return (v - center) / radius


_ZM = None
def zernike_volume(voxels, order=ORDER):
    """Author NumPy routines, preserving all121 structural entries including exact zeros."""
    global _ZM
    if _ZM is None:
        sys.path.insert(0, str(ZMPY))
        import ZMPY3D as z
        with (ZMPY / 'ZMPY3D/cache_data/LogG_CLMCache_MaxOrder20.pkl').open('rb') as handle:
            cache = pickle.load(handle)
        _ZM = z, cache
    z, c = _ZM
    # Fixed geometric center, sphere radius in voxel units, coordinates exactly[-1,1].
    edges = np.linspace(-1, 1, voxels.shape[0] + 1)
    xyz = dict(X_sample=edges, Y_sample=edges, Z_sample=edges)
    _, _, moments = z.calculate_bbox_moment(voxels.astype(float), order, xyz)
    scaled, _ = z.calculate_bbox_moment_2_zm(order, c['GCache_complex'], c['GCache_pqr_linear'],
                                           c['GCache_complex_index'], c['CLMCache3D'], moments)
    invariant = z.get_3dzd_121_descriptor(scaled)
    descriptor = np.array([invariant[n, l] for n in range(order+1) for l in range(n+1) if (n-l) % 2 == 0])
    descriptor = np.nan_to_num(descriptor, nan=0.)
    if not np.isfinite(descriptor).all() or np.linalg.norm(descriptor) < 1e-12:
        raise ValueError('Invalid Zernike descriptor')
    return descriptor / np.linalg.norm(descriptor)


_GRID = None
def zernike(v, f, resolution=RESOLUTION):
    global _GRID
    # Exact volume centroid for closed surfaces; surface centroid for open QC-failed objects.
    m = trimesh.Trimesh(v, f, process=False)
    center = m.center_mass if m.is_watertight and m.is_winding_consistent and abs(m.volume) > 1e-12 else surface_frame(v, f)[0]
    v = v - center
    radius = np.linalg.norm(v, axis=1).max() / .9
    v /= radius
    if _GRID is None or _GRID[0] != resolution:
        axis = (np.arange(resolution) + .5) * (2 / resolution) - 1
        points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        _GRID = resolution, points, np.linalg.norm(points, axis=1) <= 1
    _, points, inside = _GRID
    sdf = igl.signed_distance(points, v, f, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    # Half-voxel soft boundary reduces grid aliasing. No source mask/mesh is changed.
    field = expit(-sdf / (1 / resolution)); field[~inside] = 0
    return zernike_volume(field.reshape((resolution,) * 3))


def matrices(data, zernike_codes):
    from atlas_shape_space import distances
    result = {}
    specifications = {m['id']: m for m in data['methods']}
    for key in METHODS:
        if key == 'zernike20': result[key] = cdist(zernike_codes, zernike_codes)
        elif key == 'cajal_qgw':
            cached = np.load(JOINT / 'representations/cajal_codes.npz')
            assert cached['ids'].tolist() == [c['id'] for c in data['cells']]
            result[key] = cached['distances']
        else:
            x = [c['representations'][key]['vector'] for c in data['cells']]
            result[key] = distances(x, specifications[key]['metric'])
        d = result[key]
        if not np.isfinite(d).all() or not np.allclose(d, d.T, atol=1e-6): raise ValueError(key)
    return result


def perturb(v, f, seed):
    v = normalize(v, f); rng = np.random.default_rng(seed)
    phase = rng.normal(size=(3, 3)) * 5; angle = rng.uniform(0, 2*np.pi, 3)
    delta = np.sin(v @ phase.T + angle).mean(1); delta -= delta.mean(); delta /= max(delta.std(), 1e-12)
    normals = trimesh.Trimesh(v, f, process=False).vertex_normals
    return v + .02 * delta[:, None] * normals


def bootstrap(values, groups, weighted=False):
    f = pd.DataFrame(dict(value=values, group=groups)); a = f.groupby('group').value.agg(['mean', 'sum', 'count'])
    ix = np.random.default_rng(42).integers(0, len(a), (4000, len(a)))
    if weighted:
        boot = a['sum'].to_numpy()[ix].sum(1) / a['count'].to_numpy()[ix].sum(1); mean = float(f.value.mean())
    else:
        boot = a['mean'].to_numpy()[ix].mean(1); mean = float(a['mean'].mean())
    return dict(mean=mean, ci95=np.percentile(boot, [2.5, 97.5]).tolist(), objects=len(f), organoids=len(a))
