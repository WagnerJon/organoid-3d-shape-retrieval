"""Export a standalone Cellpose/uSegment3D run for the native Atlas viewer."""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import tifffile
import trimesh
from scipy.ndimage import map_coordinates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def export(run, output, sample_id='GFP24h_MV5'):
    info = json.loads((run / 'run.json').read_text())
    processing = json.loads((run / 'processing.json').read_text())
    with (run / 'cell_measurements.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    folder = output / sample_id
    if (output / 'dataset.json').exists():
        raise FileExistsError('Dataset already exists; choose another output directory')
    folder.mkdir(parents=True, exist_ok=True)
    # Solidity is unavailable for open surfaces; retain it where valid, but do
    # not impute it into the PCA or similarity distances.
    features = ['volume_um3', 'surface_area_um2', 'sphericity', 'elongation',
                'flatness', 'major_axis_um', 'middle_axis_um', 'minor_axis_um']
    if not rows:
        raise ValueError('No cells detected. Try different segmentation settings or inspect the raw stack.')
    if all(r.get('mesh_solidity') and np.isfinite(float(r['mesh_solidity'])) for r in rows):
        features.append('mesh_solidity')
    values = np.array([[float(r[k]) for k in features] for r in rows])
    qc = np.array([r['requires_review'].lower() != 'true' for r in rows])
    fit = values[qc] if qc.sum() >= 2 else values
    mean, std = fit.mean(0), np.maximum(fit.std(0), 1e-12)
    vectors = (values - mean) / std
    fit_vectors = (fit - mean) / std
    variance = [0., 0.]
    points = np.zeros((len(rows), 2))
    if len(fit) > 1 and np.max(np.ptp(fit_vectors, axis=0)) > 1e-12:
        pca = PCA(n_components=min(2, *fit_vectors.shape), svd_solver='full').fit(fit_vectors)
        projected = pca.transform(vectors)
        points[:, :projected.shape[1]] = projected
        variance[:len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_.tolist()
    groups = KMeans(n_clusters=min(6, len(np.unique(fit_vectors, axis=0))), random_state=42, n_init=10).fit(fit_vectors).predict(vectors)
    cells = []
    for row, vector, point, group, passed in zip(rows, vectors, points, groups, qc):
        label = int(row['label'])
        metrics = {}
        for key, value in row.items():
            if key in ('label', 'requires_review', 'quality_flags', 'touches_border', 'mesh_watertight') or value == '':
                continue
            number = float(value)
            if np.isfinite(number):
                metrics[key] = number
        cells.append(dict(id=f'{sample_id}:L{label}', sample=sample_id, label=label,
                          qc=bool(passed), split='unassigned', flags=row['quality_flags'],
                          inMainOrganoid=True,
                          metrics=metrics, representations={'traditional': dict(
                              vector=vector.tolist(), point=point.tolist(), cluster=int(group))}))
        mesh = trimesh.load(run / 'meshes' / f'cell_{label:04d}.ply', process=False)
        vertices = np.asarray(mesh.vertices, dtype='<f4')
        normals = np.asarray(mesh.vertex_normals, dtype='<f4')
        faces = np.asarray(mesh.faces, dtype='<u4')
        with (folder / f'{label}.mesh').open('wb') as handle:
            handle.write(struct.pack('<II', len(vertices), len(faces)))
            handle.write(vertices.tobytes())
            handle.write(normals.tobytes())
            handle.write(faces.tobytes())
    print(f'Converted {len(cells)} original surfaces', flush=True)
    labels = np.load(run / 'labels_processed.npy', mmap_mode='r')
    shape = np.array(labels.shape)
    spacing = np.array(processing['effectiveSpacingZYXum'])
    raw_offset = np.array(processing.get('rawGridOffsetZYXum', [0,0,0]))
    native_spacing = np.array(info['spacingZYXum'])
    if labels.max() > 65535:
        raise ValueError('Atlas previews currently support at most 65535 label IDs per stack.')
    target = np.maximum(2, np.rint(shape * min(1, 144 / shape.max())).astype(int))
    coordinates = [np.linspace(0, size - 1, int(count)) for size, count in zip(shape, target)]
    yy, xx = np.meshgrid(coordinates[1], coordinates[2], indexing='ij')
    label_preview = np.empty(tuple(target), dtype='<u2')
    raw_preview = np.empty(tuple(target), dtype=np.float32)
    # Read only the raw planes needed for the preview; never materialize the
    # full native stack or its float32 copy on this 16 GB machine.
    with tifffile.TiffFile(info['source']) as tif:
        if len(tif.pages) != info['nativeShape'][0]:
            raise ValueError('Expected one TIFF page per Z plane')
        for iz, z in enumerate(coordinates[0]):
            label_preview[iz] = map_coordinates(labels[int(round(z))], [yy, xx], order=0, prefilter=False)
            raw_z = np.clip((z * spacing[0] + raw_offset[0]) / native_spacing[0], 0, len(tif.pages)-1)
            lo = min(int(np.floor(raw_z)), len(tif.pages) - 1)
            hi = min(lo + 1, len(tif.pages) - 1)
            weight = raw_z - lo
            plane = tif.pages[lo].asarray().astype(np.float32)
            if hi != lo:
                plane *= 1 - weight
                plane += weight * tif.pages[hi].asarray()
            raw_preview[iz] = map_coordinates(plane, [(yy * spacing[1] + raw_offset[1]) / native_spacing[1],
                                                     (xx * spacing[2] + raw_offset[2]) / native_spacing[2]],
                                              order=1, mode='nearest', prefilter=False)
    low, high = np.percentile(raw_preview, [1, 99.7])
    raw_u8 = np.clip((raw_preview - low) / max(high - low, 1e-6) * 255, 0, 255).astype('u1')
    label_preview.tofile(folder / 'labels.u16')
    raw_u8.tofile(folder / 'raw.u8')
    metrics = {key: float(fit[:, n].mean()) for n, key in enumerate(features)}
    metrics['volume_um3'] = float(values[:, 0].sum())
    organoid = dict(id=sample_id, sample=sample_id, label=0, qc=True, split='stack',
                    flags='Single image stack; all candidate objects retained; no membership classification performed',
                    inMainOrganoid=True, metrics=metrics, cellCount=len(cells),
                    representations={'traditional': dict(vector=fit_vectors.mean(0).tolist(), point=[0., 0.], cluster=0)})
    document = dict(version=1, title=sample_id.replace('_', ' '), cells=cells, organoids=[organoid],
        methods=[dict(id='traditional', name='Traditional morphometry', metric='euclidean',
                      variance=variance, stackVariance=[0., 0.], dimensions=len(features), family='traditional')],
        samples=[dict(id=sample_id, shape=target.tolist(),
                      spacing=(spacing * (shape - 1) / np.maximum(target - 1, 1)).tolist(),
                      extent=(spacing * (shape - 1)).tolist(), rawAvailable=True,
                      labelGrid=dict(shape=shape.tolist(), spacingZYXum=spacing.tolist(), rawGridOffsetZYXum=raw_offset.tolist(), coordinates='Grid-local voxel centers in micrometers', linkingMode=info.get('linkingMode','standard'), segmentationMode=info.get('segmentationMode','usegment3d')),
                      sourceImage=info['source'], sourceLabels=str(run / 'labels_native.ome.tif'), meshCount=len(cells))],
        traditionalFeatures=features, normalization=dict(mean=mean.tolist(), std=std.tolist()),
        provenance=("FOCUS-3D instance segmentation restored to the original raw grid. " if info.get('segmentationMode')=='focus3d' else f"Cellpose {info['model']} ({'native 3D dynamics' if info.get('segmentationMode')=='cellpose3d' else 'u-Segment3D linking'}), processing scale {info['scale']}. ")+
        f"Original surfaces in calibrated micrometers; spacing Z/Y/X {info['spacingZYXum']}. No mesh smoothing. "
        f"All {len(cells)} candidate objects retained; {int((~qc).sum())} flagged for review. "
        'Membership inclusion is provisional until dataset preparation classifies detached cells. '
        'Raw preview is sampled to at most 144 voxels per axis. Organoid denotes an image stack; '
        'no biological labels or independent accuracy validation are supplied.')
    if info.get('segmentationMode')=='focus3d':
        document['samples'][0]['segmentation']=dict(method='focus3d',settings=info['focusIdentity']['focusSettings'],
            checkpointSHA256=info['focusIdentity']['checkpointSHA256'],configSHA256=info['focusIdentity']['configSHA256'])
    (output / 'dataset.json').write_text(json.dumps(document, allow_nan=False, separators=(',', ':')))
    print(f'Exported {len(cells)} cells ({int(qc.sum())} QC passing), preview {target.tolist()} → {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sample-id', default='GFP24h_MV5')
    args = parser.parse_args()
    export(args.run, args.output, args.sample_id)
