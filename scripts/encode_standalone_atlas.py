"""Frozen spatial-shape inference and feature controls for a standalone Atlas export."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from atlas_shape_space import signature, pack
from organoid_membership import project
from pointnet_shape_core import ROOT, sha256, sample_surface, normalize_cloud
from prepare_pointnet_shape import read_mesh
from spatial_benchmark_core import surface_frame, orbit_grids, summary_features, morphology, CLASSICAL, CONTROLS


def residual_fit(frame, values):
    x = morphology(frame, CONTROLS)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    poly = PolynomialFeatures(2, include_bias=False)
    model = Ridge(alpha=1).fit(poly.fit_transform((x-mean)/std), values)
    return dict(mean=mean, std=std, powers=poly.powers_, coef=model.coef_, intercept=model.intercept_)


def residual(frame, values, fit):
    x = (morphology(frame, CONTROLS)-fit['mean'])/fit['std']
    terms = np.prod(x[:, None, :]**fit['powers'][None], axis=-1)
    return values-(terms@fit['coef'].T+fit['intercept'])


def main(folder):
    path = folder/'dataset.json'
    before = sha256(path)
    data = json.loads(path.read_text())
    out = folder/'shape_representations'
    cache = out/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    weights = ROOT/'results/spatial_benchmark/aligned_pca.npz'
    frozen = json.loads((weights.parent/'freeze.json').read_text())
    assert sha256(weights) == frozen['artifacts']['aligned_pca.npz']
    sources = [weights, ROOT/'results/atlas_spatial/signature_encoder.npz',
               ROOT/'results/atlas_spatial/signature_nuisance.npz', ROOT/'results/atlas_spatial/spatial_nuisance.npz',
               ROOT/'results/pointnet_shape/data/manifest.csv', ROOT/'results/spatial_benchmark/dev_manifest.csv']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    pca = np.load(weights)
    sig_fit = np.load(sources[1])
    manifest = pd.read_csv(sources[4])
    dev = pd.read_csv(sources[5])
    train_ids = set(dev.loc[dev.split == 'train', 'cell_id'])
    train = manifest.set_index('cell_id').loc[sorted(train_ids)]
    assert len(train) == 96
    config = json.loads((ROOT/'config/pointnet_shape.json').read_text())
    loaded = []
    started = time.monotonic()
    for index, cell in enumerate(data['cells']):
        mesh_path = folder/cell['sample']/f"{cell['label']}.mesh"
        mesh_hash = sha256(mesh_path)
        destination = cache/f"{cell['sample']}_L{cell['label']}.npz"
        identity = hashlib.sha256(json.dumps([mesh_hash, hashes, config['seed'], config['points'], 'original-cloud-normalization-v1'], sort_keys=True).encode()).hexdigest()
        if destination.exists():
            with np.load(destination) as saved:
                valid = str(saved['identity']) == identity
        else:
            valid = False
        if not valid:
            vertices, faces = read_mesh(mesh_path)
            seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{cell['id']}".encode()).digest()[:8], 'little')
            points = sample_surface(vertices, faces, config['points'], np.random.default_rng(seed))
            _, _, center, radius = normalize_cloud(points, points)
            v = (vertices.astype(float)-center)/radius
            f = faces.astype(np.int64)
            frame = surface_frame(v, f)
            orbit = (orbit_grids(v, f, frame).reshape(4, -1)-pca['mean'])@pca['components'].T
            sig = (signature(v, f)-sig_fit['mean'])@sig_fit['components'].T*sig_fit['scale']
            if not np.isfinite(orbit).all() or not np.isfinite(sig).all():
                raise ValueError(f"Nonfinite descriptor: {cell['id']}")
            np.savez_compressed(destination, identity=identity, orbit=orbit, signature=sig, gap=frame[3],
                                normalization_center=center, normalization_radius=radius, mesh_sha256=mesh_hash)
        with np.load(destination) as saved:
            loaded.append({key: saved[key].copy() for key in ['orbit', 'signature', 'gap']})
        if (index+1) % 10 == 0 or index == 0 or index+1 == len(data['cells']):
            print(f"Encoded {index+1}/{len(data['cells'])} surfaces · {time.monotonic()-started:.1f}s", flush=True)
    z = np.stack([a['orbit'] for a in loaded])
    sig = np.stack([a['signature'] for a in loaded])
    gaps = np.array([float(a['gap']) for a in loaded])
    frame = pd.DataFrame([dict(cell_id=c['id'], **c['metrics']) for c in data['cells']])
    missing = []
    # A finite comparison vector is needed for every object. Impute only in
    # descriptor inputs; never rewrite physical measurements or quality flags.
    for column in CLASSICAL:
        if column not in frame:
            frame[column] = np.nan
        bad = ~np.isfinite(frame[column].to_numpy(float))
        fill = float(train[column].median())
        for cid in frame.loc[bad, 'cell_id']:
            missing.append(dict(cell_id=cid, feature=column, comparison_value=fill,
                                rule='Original 96-cell training median; measured value remains missing'))
        frame.loc[bad, column] = fill
    specs = []
    def add(key, name, vectors, description, family='shape', metric='euclidean', projection='pca'):
        vectors = np.asarray(vectors, float)
        assert len(vectors) == len(data['cells']) and np.isfinite(vectors).all()
        spec = dict(id=key, name=name, metric=metric, dimensions=vectors.shape[1],
                    variance=[0., 0.], stackVariance=[0., 0.], family=family,
                    description=description, projection=projection, coverage=f"All {len(data['cells'])} objects")
        if metric == 'shape_hybrid95':
            spec['scope'] = 'cells'
        for cell, vector in zip(data['cells'], vectors):
            cell['representations'][key] = dict(vector=vector.tolist(), point=[0., 0.], cluster=0)
        specs.append(spec)
    add('spatial_robust', 'Spatial PCA · ambiguity fallback', pack(gaps, z, sig),
        'Frozen original-dataset 95D spatial SDF PCA in four proper sign frames. Axis gap < 0.05 uses the frozen invariant surface signature. MDS shows hybrid distances. Geometry inference only; no new training or biological validation.', metric='shape_hybrid95', projection='mds')
    add('surface_signature', 'Surface signature · 95D', sig,
        'Frozen surface radial/pair-distance signature PCA and scale from the original 96 training cells. Discards chirality and some spatial arrangement.')
    add('surface_signature_adjusted', 'Surface signature · adjusted', residual(frame, sig, np.load(sources[2])),
        'Frozen original degree-2 Ridge adjustment for log volume, elongation and sphericity; no fitting on this stack.')
    add('spatial_adjusted', 'Spatial PCA · adjusted summary', residual(frame, summary_features(z), np.load(sources[3])),
        'Frozen original adjusted mean/std of four spatial frames. This summary does not have the ambiguity fallback.')
    groups = [('morph_controls', 'Volume + elongation + sphericity', CONTROLS),
              ('morph_size', 'Size features', ['volume_um3', 'surface_area_um2', 'major_axis_um', 'middle_axis_um', 'minor_axis_um']),
              ('morph_shape', 'Shape features', ['sphericity', 'elongation', 'flatness', 'mesh_solidity']),
              ('morph_all', 'All 9 traditional features', CLASSICAL)]
    fit_records = {}
    for key, name, cols in groups:
        reference = morphology(train, cols)
        mean, std = reference.mean(0), reference.std(0).clip(1e-6)
        values = (morphology(frame, cols)-mean)/std
        fit_records[key] = dict(columns=cols, mean=mean.tolist(), std=std.tolist())
        add(key, name, values, 'Original 96-cell training standardization. Positive size features log-transformed. Features: '+', '.join(cols)+'. Missing solidity uses original training median in comparison vectors only.', family='traditional')
        if key == 'morph_all':
            fit = residual_fit(train, (reference-mean)/std)
            np.savez(out/'morph_all_nuisance.npz', **fit)
            add('morph_adjusted', 'All 9 traditional · adjusted', residual(frame, values, fit),
                'Original-training-only degree-2 Ridge adjustment for log volume, elongation and sphericity. Missing solidity uses original training median.', family='traditional')
    keys = {s['id'] for s in specs}
    for spec in specs:
        spec['variance'] = project(data['cells'], spec['id'], spec['metric'])
        if spec.get('scope') != 'cells':
            for stack in data['organoids']:
                cells = [c for c in data['cells'] if c['sample'] == stack['sample'] and c['qc'] and c.get('inMainOrganoid', True)]
                if cells:
                    stack['representations'][spec['id']] = dict(vector=np.mean([c['representations'][spec['id']]['vector'] for c in cells], axis=0).tolist(), point=[0., 0.], cluster=0)
    # Atlas initially selects the last method when opening an unfamiliar dataset.
    # Put the primary shape method last, rather than a traditional control that
    # is intentionally absent from its learned/shape-space picker.
    ordered = sorted(specs, key=lambda s: (s['family'] != 'traditional', s['id'] == 'spatial_robust'))
    data['methods'] = [m for m in data['methods'] if m['id'] not in keys]+ordered
    data['shapeInference'] = dict(protocol='Frozen original-dataset spatial and feature encoders; new-stack inference only',
        sources=hashes, training_ids=sorted(train_ids), morphology_scaling=fit_records, imputed_comparison_inputs=missing,
        normalization='Same original spatial benchmark: deterministic 2048 area-weighted surface samples; sample centroid / maximum radius; original physical mesh then transformed.',
        display='PCA/MDS and six exploratory groups fit on this included stack; not phenotype labels. One stack cannot establish cross-stack retrieval performance.',
        ambiguous_objects=int((gaps < .05).sum()), objects=len(data['cells']))
    data['provenance'] += ' Added frozen original-dataset spatial PCA with ambiguity fallback, signature and adjusted spaces, and size/shape/control/all-nine feature groups. No retraining on this stack. Comparison-input-only training-median imputation is recorded in shapeInference. Original measurements, meshes, QC and inclusion flags unchanged. Descriptor projections/groups are descriptive fits on all included objects.'
    if sha256(path) != before:
        raise RuntimeError('Dataset changed during inference; refusing publication')
    for source in sources:
        assert sha256(source) == hashes[str(source.relative_to(ROOT))]
    backup = out/f'dataset.before-{before[:12]}.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    temp = path.with_suffix('.representations.tmp')
    temp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')))
    os.replace(temp, path)
    (out/'complete.json').write_text(json.dumps(dict(before=before, after=sha256(path), objects=len(frame),
        methods=sorted(keys), imputed_inputs=len(missing), ambiguous_objects=int((gaps < .05).sum()),
        elapsed_seconds=time.monotonic()-started, source_hashes=hashes), indent=2))
    print(f'Published {len(specs)} spaces for {len(frame)} objects; {len(missing)} imputed comparison inputs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    main(args.dataset)
