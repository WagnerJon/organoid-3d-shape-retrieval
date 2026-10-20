"""Build a portable, local-only dataset for the native Organoid Explorer.

Joins representations by cell_id (never by assumed row order). PCA and k-means
are descriptive fits on main-organoid cells; distances retain the full feature vectors.
"""
from pathlib import Path
import argparse
import json
import struct
import numpy as np
import pandas as pd
import tifffile
import trimesh
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from organoid_membership import update as update_membership

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['volume_um3', 'surface_area_um2', 'sphericity', 'elongation', 'flatness',
            'mesh_solidity', 'major_axis_um', 'middle_axis_um', 'minor_axis_um']
METHODS = [('simclr3d', 'Mask SimCLR'), ('simclr3d_raw_isolated', 'Isolated raw SimCLR'),
           ('simclr3d_raw', 'Context raw SimCLR'), ('icone3d_raw_isolated', 'Isolated raw IConE'),
           ('icone3d_raw_context', 'Context raw IConE')]


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def projections(ids, vectors):
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite representation')
    pca = PCA(n_components=2, svd_solver='full').fit(vectors)
    points = pca.transform(vectors)
    clusters = KMeans(n_clusters=min(6, len(ids)), random_state=42, n_init=10).fit_predict(vectors)
    return {str(key): dict(vector=np.round(v, 7).tolist(), point=np.round(p, 5).tolist(), cluster=int(c))
            for key, v, p, c in zip(ids, vectors, points, clusters)}, pca.explained_variance_ratio_.tolist()


def export(root, output, limit=None):
    result = root / 'results'
    morph = pd.read_csv(result / 'cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id', drop=False)
    manifest = pd.read_csv(result / 'simclr3d/manifest.csv').set_index('cell_id', drop=False)
    if not morph.index.is_unique or not manifest.index.is_unique:
        raise ValueError('Duplicate cell IDs')
    if set(morph.index) != set(manifest.index):
        raise ValueError('Morphology/manifest ID mismatch')
    morph = morph.loc[manifest.index]
    output.mkdir(parents=True, exist_ok=True)
    cells = {key: dict(id=key, sample=r['sample'], label=int(r.label), qc=bool(r.shape_qc_pass),
                      split=str(manifest.loc[key, 'split']), flags=str(r.qc_flags),
                      metrics={f: float(r[f]) for f in FEATURES + ['centroid_z_um', 'centroid_y_um', 'centroid_x_um']}, representations={})
             for key, r in morph.iterrows()}
    stacks = {}
    for sample, rows in morph.groupby('sample'):
        stacks[sample] = dict(id=sample, sample=sample, label=0, qc=bool(rows.shape_qc_pass.any()),
                             split='stack', flags='Stack aggregate of QC-passing cells',
                             metrics={f: float(rows.loc[rows.shape_qc_pass, f].mean())
                                      if rows.shape_qc_pass.any() else float(rows[f].mean()) for f in FEATURES},
                             representations={}, cellCount=len(rows))
        stacks[sample]['metrics']['volume_um3'] = float(rows.volume_um3.sum())
    methods = []
    values = morph[FEATURES].to_numpy()
    mean, std = values.mean(0), values.std(0)
    standard = (values - mean) / np.maximum(std, 1e-12)
    inputs = [('traditional', 'Traditional morphometry', 'euclidean', morph.index, standard)]
    for folder, name in METHODS:
        path = result / folder / 'embeddings.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path).set_index('cell_id')
        if not frame.index.is_unique:
            raise ValueError(f'Duplicate cell IDs: {path}')
        ids = morph.index.intersection(frame.index, sort=False)
        cols = [c for c in frame if c.startswith('embedding_')]
        if not cols:
            raise ValueError(f'No embedding columns: {path}')
        inputs.append((folder, name, 'cosine', ids, normalize(frame.loc[ids, cols].to_numpy())))
    for key, name, metric, ids, vectors in inputs:
        projected, variance = projections(ids, vectors)
        for cid, value in projected.items():
            cells[cid]['representations'][key] = value
        stack_ids, stack_vectors = [], []
        for sample in stacks:
            use = [i for i, cid in enumerate(ids) if cells[cid]['sample'] == sample and cells[cid]['qc']]
            if use:
                stack_ids.append(sample)
                stack_vectors.append(vectors[use].mean(0))
        stack_vectors = np.array(stack_vectors)
        if metric == 'cosine':
            stack_vectors = normalize(stack_vectors)
        projected_stacks, stack_variance = projections(stack_ids, stack_vectors)
        for sample, value in projected_stacks.items():
            stacks[sample]['representations'][key] = value
        methods.append(dict(id=key, name=name, metric=metric, variance=variance, stackVariance=stack_variance,
                            dimensions=vectors.shape[1]))
    samples = []
    for n, (sample, rows) in enumerate(manifest.groupby('sample')):
        if limit is not None and n >= limit:
            break
        source = Path(rows.iloc[0].source_label_file)
        # Relocate an existing result tree when the original absolute prefix changed.
        if not source.exists() and 'results' in source.parts:
            source = root.joinpath(*source.parts[source.parts.index('results'):])
        info = json.loads((source.parent / 'complete.json').read_text())
        labels = tifffile.imread(source)
        spacing = np.asarray(info['effective_spacing_zyx_um'])
        shape = np.asarray(labels.shape)
        folder = output / sample
        folder.mkdir(exist_ok=True)
        mesh_count = 0
        for row in rows.itertuples():
            original = source.parent / 'meshes' / f'cell_{row.label:04d}.ply'
            # Use original measured surfaces; no silent smoothing or decimation.
            mesh = trimesh.load(original, process=False)
            vertices = np.asarray(mesh.vertices, dtype='<f4')
            normals = np.asarray(mesh.vertex_normals, dtype='<f4')
            faces = np.asarray(mesh.faces, dtype='<u4')
            with (folder / f'{row.label}.mesh').open('wb') as f:
                f.write(struct.pack('<II', len(vertices), len(faces)))
                f.write(vertices.tobytes()); f.write(normals.tobytes()); f.write(faces.tobytes())
            mesh_count += 1
        target = np.maximum(1, np.rint(shape * min(1, 144 / shape.max())).astype(int))
        preview_labels = zoom(labels, target / shape, order=0, prefilter=False).astype('<u2')
        preview_labels.tofile(folder / 'labels.u16')
        raw_path = Path(info['image'])
        raw_available = raw_path.exists()
        if raw_available:
            raw = tifffile.imread(raw_path).astype(np.float32)
            raw = zoom(raw, target / np.asarray(raw.shape), order=1, prefilter=False)
            low, high = np.percentile(raw, [1, 99.7])
            raw = np.clip((raw - low) / max(high - low, 1e-6) * 255, 0, 255).astype('u1')
            raw.tofile(folder / 'raw.u8')
        samples.append(dict(id=sample, shape=target.tolist(), spacing=(spacing * shape / target).tolist(),
                            extent=(spacing * (shape - 1)).tolist(), rawAvailable=raw_available,
                            sourceImage=str(raw_path), sourceLabels=str(source), meshCount=mesh_count))
        if n % 10 == 0:
            print(f'Prepared {n+1}/108 stacks: {sample}', flush=True)
    valid = {s['id'] for s in samples}
    document = dict(version=1, title='Organoid Atlas', methods=methods, samples=samples,
                    cells=[v for v in cells.values() if v['sample'] in valid],
                    organoids=[v for k, v in stacks.items() if k in valid],
                    provenance='Original predicted surfaces. Raw previews: 1–99.7 percentile display contrast, resampled to ≤144 voxels per axis. Physical calibration is inferred, not TIFF-verified. Organoids denote image stacks, not verified independent biological specimens. PCA and 6-group k-means are descriptive fits on main-organoid cells; groups are not biological classes. Stack representations average QC-passing cell vectors; stack table volume sums all cells.',
                    traditionalFeatures=FEATURES, normalization=dict(mean=mean.tolist(), std=std.tolist()))
    (output / 'dataset.json').write_text(json.dumps(document, allow_nan=False, separators=(',', ':')))
    document = update_membership(output)
    print(f'Exported {len(document["cells"])} cells, {len(samples)} stacks, {len(methods)} representations → {output}', flush=True)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/explorer_dataset')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    export(args.root, args.output, args.limit)
