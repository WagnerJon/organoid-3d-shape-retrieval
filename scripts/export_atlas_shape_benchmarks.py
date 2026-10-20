"""Publish cached Allen, Point-MAE and Lite3D readouts without retraining.

Use original benchmark codes, or their verified full-data frozen-inference
extension when complete. Benchmark nuisance fits remain unchanged; descriptive
display PCA/groups follow Atlas membership.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pointnet_shape_core import ROOT, sha256
from evaluate_spatial_benchmark import residual_apply
from organoid_membership import rebuild

OUT = ROOT / 'results/atlas_shape_benchmarks'
ATLAS = ROOT / 'results/explorer_dataset'
FAMILIES = ('allen', 'point_mae', 'shapeembed3d')


def benchmark_readouts():
    """Return all-cell readouts and verify against the original validation exports."""
    methods, sources, reference_ids = [], {}, None

    def load(path):
        sources[str(path.relative_to(ROOT))] = sha256(path)
        return path

    for family, folder in zip(FAMILIES, ('allen_reference', 'point_mae', 'shapeembed3d')):
        base = ROOT / 'results' / folder
        frame = pd.read_csv(load(base / 'manifest.csv'))
        ids = frame.cell_id.tolist()
        if len(set(ids)) != len(ids) or set(frame.split) - {'train', 'val', 'validation'}:
            raise ValueError('Duplicate identities or unexpected evaluation split')
        if reference_ids is not None and ids != reference_ids:
            raise ValueError('Benchmark source identity/order mismatch')
        reference_ids = ids
        validation = np.load(load(base / ('evaluation/readouts.npz' if family == 'allen' else 'readouts.npz')))
        positions = {cid: i for i, cid in enumerate(ids)}
        validation_ids = validation['ids'].tolist()
        rows = [positions[c] for c in validation_ids]
        if family == 'allen':
            codes = np.load(load(base / 'evaluation/codes.npz'))
            if codes['ids'].tolist() != ids:
                raise ValueError('Allen code identities differ')
            specs = [(arm, codes[arm + '_norms'], base / 'evaluation' / (arm + '_readout_fit.npz'),
                      'Allen VN-DGCNN · ' + title,
                      '512 vector-channel norms from the released VN-DGCNN/SDF encoder. ' + details)
                     for arm, title, details in [
                         ('pretrained', 'released', 'Released NPM1 weights; no local encoder adaptation.'),
                         ('head_decoder', 'head adapted', 'Only the final encoder head and SDF decoder were adapted; the backbone stayed frozen.')]]
        elif family == 'point_mae':
            specs = []
            for arm in ('pretrained', 'adapted'):
                codes = np.load(load(base / (arm + '_features.npz')))
                for kind in ('pose_average', 'native'):
                    detail = ('External surface-PCA pose handling averages four proper sign frames; ambiguous axes can affect results. This is not an intrinsically rotation-invariant encoder.'
                              if kind == 'pose_average' else 'Native orientation reference: sensitive to object rotation.')
                    specs.append((arm + '_' + kind, codes[kind], base / (arm + '_' + kind + '_fit.npz'),
                                  'Point-MAE · ' + ('adapted' if arm == 'adapted' else 'released') + (' · pose averaged' if kind == 'pose_average' else ' · native orientation'),
                                  '768D mean/max token descriptor. ' + detail + ' Masked patch completion is not reconstruction from this global descriptor.'))
        else:
            specs = []
            for kind in ('invariant', 'cnn'):
                codes = np.load(load(base / (kind + '_codes.npz')))
                if codes['ids'].tolist() != ids:
                    raise ValueError('Lite3D code identities differ')
                detail = ('Permutation-invariant EDM graph encoder; O(3) invariance includes reflections, so chirality is lost.' if kind == 'invariant'
                          else 'Literal CNN reference on distance matrices; sensitive to point indexing.')
                specs.append((kind, codes['mean'], base / (kind + '_fit.npz'),
                              'ShapeEmbedLite3D · ' + ('invariant' if kind == 'invariant' else 'CNN reference'),
                              'Experimental unofficial 128D VAE latent mean, adapted from ShapeEmbedLite. ' + detail + ' Sparse global point reconstructions remain generic; no geometry improvement is established.'))
        for arm, raw, fit_path, title, description in specs:
            fit = dict(np.load(load(fit_path)))
            adjusted = residual_apply(frame, (raw - fit['center']) / fit['scale'], fit)
            for suffix, values in [('raw', raw), ('adjusted', adjusted)]:
                key = ('allen_' if family == 'allen' else family + '_') + arm + '_' + suffix
                check_key = arm + '_' + suffix if family == 'allen' else key
                np.testing.assert_allclose(values[rows], validation[check_key], atol=1e-10, rtol=1e-10)
                if values.shape[0] != len(ids) or not np.isfinite(values).all():
                    raise ValueError('Incomplete or non-finite embedding')
                info = description + ' Cell-level train/validation coverage only; test cells have no code.'
                if suffix == 'adjusted':
                    info += ' Standardized and residualized for log volume, elongation and sphericity using the original fixed 96 training-cell fit; no refitting.'
                methods.append((dict(id=key, name=title + (' · adjusted' if suffix == 'adjusted' else ' · raw'),
                                     dimensions=values.shape[1], metric='euclidean', projection='pca',
                                     family=family, scope='cells', coverage='benchmark', description=info, variance=[0., 0.], stackVariance=[0., 0.]), values))
    return frame, methods, sources


def readouts(include_full=True):
    frame, methods, sources = benchmark_readouts()
    full = ROOT/'results/atlas_full_shape'
    if not include_full or not (full/'completed.json').exists():
        return frame, methods, sources
    complete = json.loads((full/'completed.json').read_text())
    for name, digest in complete['artifacts'].items():
        if sha256(full/name) != digest:
            raise ValueError('Stale full-data inference artifact: '+name)
        sources[str((full/name).relative_to(ROOT))] = digest
    sources[str((full/'completed.json').relative_to(ROOT))] = sha256(full/'completed.json')
    full_frame = pd.read_csv(full/'manifest.csv')
    vectors = np.load(full/'vectors.npz')
    if vectors['ids'].tolist() != full_frame.cell_id.tolist() or set(frame.cell_id)-set(full_frame.cell_id):
        raise ValueError('Full-data identities differ')
    positions = {cid: i for i, cid in enumerate(full_frame.cell_id)}
    rows = [positions[cid] for cid in frame.cell_id]
    expanded = []
    for meta, original in methods:
        values = vectors[meta['id']]
        np.testing.assert_array_equal(values[rows], original)
        if values.shape != (len(full_frame), meta['dimensions']) or not np.isfinite(values).all():
            raise ValueError('Invalid full-data embedding')
        meta = dict(meta)
        meta['coverage'] = 'all_cells'
        meta['description'] = meta['description'].replace(
            'Cell-level train/validation coverage only; test cells have no code.',
            'Frozen inference on all cells for exploration, including excluded/QC-flagged cells for review. Original 609 benchmark codes preserved; no retraining. Analysis membership and QC filters still apply.')
        expanded.append((meta, values))
    return full_frame, expanded, sources


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ATLAS / 'dataset.json'
    before = sha256(path)
    data = json.loads(path.read_text())
    lookup = {c['id']: c for c in data['cells']}
    frame, methods, sources = readouts()
    for row in frame.to_dict('records'):
        cell = lookup[row['cell_id']]
        if sha256(ATLAS / cell['sample'] / f"{cell['label']}.mesh") != row['source_mesh_sha256']:
            raise ValueError('Stale mesh: ' + row['cell_id'])
        for feature in ('volume_um3', 'elongation', 'sphericity'):
            if not np.isclose(cell['metrics'][feature], row[feature], rtol=1e-8):
                raise ValueError('Stale nuisance measurements: ' + row['cell_id'])
    keys = {m['id'] for m, _ in methods}
    for cell in data['cells']:
        for key in keys:
            cell['representations'].pop(key, None)
    for method, values in methods:
        for cid, vector in zip(frame.cell_id, values):
            lookup[cid]['representations'][method['id']] = dict(vector=vector.tolist(), point=[0., 0.], cluster=0)
    data['methods'] = [m for m in data['methods'] if m['id'] not in keys] + [m for m, _ in methods]
    benchmark = pd.read_csv(ROOT/'results/allen_reference/manifest.csv')
    full_coverage = len(frame) == len(data['cells'])
    data['shapeBenchmarks'] = dict(version=2 if full_coverage else 1, cells=len(frame), ids=frame.cell_id.tolist(),
                                   splits=frame.split.value_counts().to_dict(), sources=sources,
                                   fullCoverage=full_coverage, benchmarkCells=len(benchmark),
                                   benchmarkSplits=dict(zip(benchmark.cell_id, benchmark.split)),
                                   scope='Full-data frozen inference for exploration; original benchmark609 preserved. Fixed original adjustment fits; descriptive PCA and groups on included cells. No organoid aggregation or new biological validation.' if full_coverage else 'Cached train/validation codes only. Fixed benchmark adjustment fits; descriptive PCA and groups on included cells. No organoid aggregation or new biological validation.')
    data['organoids'] = data.get('membershipStackTemplates', data['organoids'])
    rebuild(data)
    if sha256(path) != before:
        raise ValueError('Atlas changed while exporting; cancelled')
    backup = OUT / f'dataset.before-{before[:12]}.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_suffix('.benchmarks.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')))
    os.replace(tmp, path)
    (OUT / 'export.json').write_text(json.dumps(dict(before=before, after=sha256(path), backup=str(backup),
                                                   methods=sorted(keys), sources=sources), indent=2))
    print(f'Published {len(methods)} shape methods for {len(frame)} cells; existing methods retained.')


if __name__ == '__main__':
    main()
