"""Validate joint Atlas assets, method coverage and independent frozen-inference probes."""
import argparse
import csv
import json
import subprocess
import struct
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pointnet_shape_core import ROOT, sha256
from prepare_pointnet_shape import read_mesh
from encode_standalone_atlas import residual
from infer_joint_organoid_methods import original_methods


def cajal_checks(folder):
    from scipy.spatial.distance import cdist
    from cajal.qgw import quantized_icdm, quantized_gw
    out = folder/'representations'
    source = np.load(out/'surfaces.npz')
    codes = np.load(out/'cajal_codes.npz')
    u = np.full(128, 1/128)
    probes = [(0, len(source['ids'])-1), (len(source['ids'])//3, len(source['ids'])//2)]
    for i, j in probes:
        a = quantized_icdm(cdist(source['points'][i], source['points'][i]), u, 32)
        b = quantized_icdm(cdist(source['points'][j], source['points'][j]), u, 32)
        value = quantized_gw(a, b)[1]
        assert np.isclose(value, codes['distances'][i, j], atol=1e-10, rtol=1e-8)
    print('Independent CAJAL pair checks passed', flush=True)


def main(folder):
    import torch
    from atlas_shape_space import distances
    from simclr3d_core import ResNet3DSimCLR
    from shapeembed3d_core import ShapeEmbed3D, fps_indices, edm
    out = folder/'representations'
    atlas = folder/'atlas_dataset'
    data = json.loads((atlas/'dataset.json').read_text())
    native_validator = Path('/tmp/validate_standalone_atlas')
    if native_validator.exists():
        subprocess.run([str(native_validator), str(atlas/'dataset.json')], check=True)
    assert len(data['samples']) == len(data['organoids']) == 5
    frame = pd.read_csv(out/'manifest.csv')
    assert frame.cell_id.tolist() == [c['id'] for c in data['cells']]
    methods = {m['id']: m for m in data['methods']}
    expected = {m['id'] for m in original_methods()}-{'cajal_gw_validation'}
    assert len(expected) == 36 and set(methods) == expected
    for key in ['traditional', 'morph_all', 'morph_size', 'morph_shape', 'surface_signature']:
        points = np.array([s['representations'][key]['point'] for s in data['organoids']])
        assert points.shape == (5, 2) and np.isfinite(points).all()
        assert np.ptp(points, axis=0).max() > 0, f'Collapsed organoid projection: {key}'
    vectors = np.load(out/'vectors.npz')
    assert vectors['ids'].tolist() == frame.cell_id.tolist()
    for key in expected:
        np.testing.assert_array_equal(vectors[key], np.array([c['representations'][key]['vector'] for c in data['cells']]))
    summary = []
    for number in range(1, 6):
        sample = f'GFP24h_MV{number}'
        run = folder/'organoids'/f'MV{number}'
        cfg = json.loads((run/'run.json').read_text())
        # Atlas's segmentation tab reads this established sidecar to recover
        # native calibration rather than inheriting example-dataset defaults.
        (run/'settings.json').write_text(json.dumps(dict(
            config=dict(model=cfg['model'], spacing_zyx_um=cfg['spacingZYXum'],
                        scale=cfg['scale'], diameter=cfg['diameterProcessedPx'],
                        background_correction=cfg['backgroundCorrection'], gpu=True),
            calibration_source=cfg['calibrationSource'], source_run='run.json'), indent=2))
        raw = folder/'raw'/Path(cfg['source']).name
        original = folder.parent/raw.name
        assert sha256(raw) == sha256(original) == cfg['sourceSHA256']
        rows = list(csv.DictReader((run/'cell_measurements.csv').open()))
        selected = [c for c in data['cells'] if c['sample'] == sample]
        assert len(selected) == len(rows)
        lookup = {int(r['label']): r for r in rows}
        for c in selected:
            source = lookup[c['label']]
            assert c['qc'] == (source['requires_review'].lower() != 'true')
            for name in c['metrics']:
                assert np.isclose(c['metrics'][name], float(source[name]), rtol=1e-12)
        stack = next(s for s in data['organoids'] if s['sample'] == sample)
        assert stack['cellCount'] == len(selected)
        assert np.isclose(stack['metrics']['volume_um3'], sum(c['metrics']['volume_um3'] for c in selected))
        sample_info = next(s for s in data['samples'] if s['id'] == sample)
        count = int(np.prod(sample_info['shape']))
        assert (atlas/sample/'labels.u16').stat().st_size == count*2
        assert (atlas/sample/'raw.u8').stat().st_size == count
        summary.append(dict(organoid=f'MV{number}', objects=len(selected), qc_passing=sum(c['qc'] for c in selected), raw_sha256=cfg['sourceSHA256']))
    for c in data['cells']:
        assert set(c['representations']) == expected
        for key, rep in c['representations'].items():
            assert len(rep['vector']) == methods[key]['dimensions'] and len(rep['point']) == 2
            assert np.isfinite(rep['vector']+rep['point']).all()
        path = atlas/c['sample']/f"{c['label']}.mesh"
        v, f = read_mesh(path)
        assert np.isfinite(v).all() and len(f)>0
        assert sha256(path) == frame.loc[frame.cell_id == c['id'], 'source_mesh_sha256'].iloc[0]
        cloud = atlas/c['pointReconstructions']['pointnet_shape128']
        assert cloud.stat().st_size == 4+2048*3*4
        payload = cloud.read_bytes()
        assert struct.unpack('<I', payload[:4])[0] == 2048
        assert np.isfinite(np.frombuffer(payload, dtype='<f4', offset=4)).all()
    for rel, digest in json.loads((out/'frozen_sources.json').read_text()).items():
        assert sha256(ROOT/rel) == digest
    hybrid = distances([c['representations']['spatial_robust']['vector'] for c in data['cells']], 'shape_hybrid95')
    assert np.isfinite(hybrid).all() and (hybrid>=0).all() and np.allclose(hybrid, hybrid.T)
    qgw = np.load(out/'cajal_codes.npz')['distances']
    assert qgw.shape == (len(frame), len(frame)) and np.allclose(qgw, qgw.T) and (qgw>=0).all()
    # Verify every adjusted learned vector against its independent frozen fit.
    for family, arms in [('allen', ['pretrained', 'head_decoder']),
                         ('point_mae', ['pretrained_native', 'pretrained_pose_average', 'adapted_native', 'adapted_pose_average']),
                         ('shapeembed3d', ['invariant', 'cnn'])]:
        for arm in arms:
            fit_path = ROOT/'results'/('allen_reference/evaluation' if family == 'allen' else family)/(arm+('_readout_fit.npz' if family == 'allen' else '_fit.npz'))
            fit = dict(np.load(fit_path))
            displayed_raw = np.array([c['representations'][f'{family}_{arm}_raw']['vector'] for c in data['cells']])
            # The frozen benchmark standardizes the encoder's native array
            # dtype: Point-MAE/Lite are float32, Allen norms are float64.
            # JSON numbers decode as float64, so verify the raw cache join
            # exactly and retain its original precision for this check.
            if family == 'allen':
                cache = np.load(out/'allen_codes.npz')
                raw = cache[arm]
            elif family == 'point_mae':
                source_arm, kind = arm.split('_', 1)
                cache = np.load(out/f'point_mae_{source_arm}.npz')
                raw = cache[kind]
            else:
                cache = np.load(out/f'lite_{arm}.npz')
                raw = cache['mean']
            np.testing.assert_array_equal(displayed_raw, raw.astype(float))
            adjusted = np.array([c['representations'][f'{family}_{arm}_adjusted']['vector'] for c in data['cells']])
            np.testing.assert_allclose(adjusted, residual(frame, (raw-fit['center'])/fit['scale'], fit), atol=1e-9, rtol=1e-9)
    # Independent CPU inference checks catch model-key/input mixups and large
    # Metal numerical errors without repeating the expensive complete run.
    torch.set_num_threads(2)
    index = next(i for i, c in enumerate(data['cells']) if c['qc'])
    errors = {}
    def check(name, actual, saved):
        error = float(np.linalg.norm(actual-saved)/max(np.linalg.norm(saved), 1e-12))
        assert error < .001, (name, error)
        errors[name] = error
    variants = [('simclr3d', 'mask'), ('simclr3d_raw', 'context'), ('simclr3d_raw_isolated', 'isolated'),
                ('icone3d_raw_isolated', 'isolated'), ('icone3d_raw_context', 'context')]
    for key, source in variants:
        checkpoint = torch.load(ROOT/'results'/key/'best.pt', map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        model = ResNet3DSimCLR(cfg['base_channels'], cfg['embedding_dim'], cfg.get('projection_dim', 64))
        if key.startswith('icone'):
            model = model.encoder; model.load_state_dict(checkpoint['backbone'])
        else:
            model.load_state_dict(checkpoint['model']); model = model.encoder
        model.eval()
        a = np.load(out/'image_inputs'/f'{source}.npy', mmap_mode='r')
        with torch.no_grad():
            actual = model(torch.tensor(np.array(a[index:index+1], copy=True), dtype=torch.float32)[:, None]).numpy()[0]
        actual /= max(np.linalg.norm(actual), 1e-12)
        check(key, actual, np.array(data['cells'][index]['representations'][key]['vector']))
    from pointnet_shape_core import PointCloudAutoencoder, sample_surface, normalize_cloud
    cfg = json.loads((ROOT/'config/pointnet_shape.json').read_text())
    c = data['cells'][index]
    v, f = read_mesh(atlas/c['sample']/f"{c['label']}.mesh")
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{c['id']}".encode()).digest()[:8], 'little')
    cloud = sample_surface(v, f, cfg['points'], np.random.default_rng(seed))
    normalized, _, _, _ = normalize_cloud(cloud, cloud)
    state = torch.load(ROOT/'results/pointnet_shape/baseline/best.pt', map_location='cpu', weights_only=False)
    model = PointCloudAutoencoder(cfg).eval(); model.load_state_dict(state['model'])
    with torch.no_grad():
        actual = model(torch.tensor(normalized[None]))[0].numpy()[0]
    actual /= max(np.linalg.norm(actual), 1e-12)
    check('pointnet_shape128', actual, np.array(c['representations']['pointnet_shape128']['vector']))
    row = frame.iloc[index]
    with np.load(out/'geometry'/f'{index:04d}.npz') as a:
        pool = a['points']
    candidates = pool[np.random.default_rng(20261022+4*int(row.inference_index)).choice(32768, 2048, replace=False)]
    points = candidates[fps_indices(candidates[None], 128)[0]]
    for kind in ['invariant', 'cnn']:
        state = torch.load(ROOT/'results/shapeembed3d'/kind/'best.pt', map_location='cpu', weights_only=False)
        model = ShapeEmbed3D(kind).eval(); model.load_state_dict(state['model'])
        with torch.no_grad():
            actual = model.encode(edm(torch.tensor(points[None])))[0].numpy()[0]
        check('shapeembed3d_'+kind, actual, np.array(data['cells'][index]['representations'][f'shapeembed3d_{kind}_raw']['vector']))
    from point_mae_reference import model as point_model, encode
    from run_point_mae_reference import draw
    from spatial_benchmark_core import SIGNS
    with np.load(out/'geometry'/f'{index:04d}.npz') as a:
        axes = a['axes']
    cloud = draw(pool, int(row.inference_index))[None]
    for arm in ['pretrained', 'adapted']:
        model = point_model()
        if arm == 'adapted':
            state = torch.load(ROOT/'results/point_mae/best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            native = encode(model, torch.tensor(cloud)).numpy()[0]
            pose = np.mean([encode(model, torch.tensor((cloud@axes*sign).astype(np.float32))).numpy()[0] for sign in SIGNS], axis=0)
        for kind, actual in [('native', native), ('pose_average', pose)]:
            check(f'point_mae_{arm}_{kind}', actual, np.array(data['cells'][index]['representations'][f'point_mae_{arm}_{kind}_raw']['vector']))
    from train_allen_reference import CachedAllen
    a = np.load(out/'allen'/f'{index:04d}.npz')
    for arm in ['pretrained', 'head_decoder']:
        model = CachedAllen().eval()
        if arm == 'head_decoder':
            state = torch.load(ROOT/'results/allen_reference/head_decoder/best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
        with torch.no_grad():
            actual = model.encode(torch.tensor(a['features'][None])).norm(dim=-1).numpy()[0]
        check('allen_'+arm, actual, np.array(data['cells'][index]['representations'][f'allen_{arm}_raw']['vector']))
    subprocess.run([str(ROOT/'.cache/cajal-env/bin/python'), str(Path(__file__)), str(folder), '--cajal-only'], check=True)
    notes = json.loads((out/'image_inputs/complete.json').read_text())['notes']
    fallback = [n for n in notes if n['subvoxel_fragment_fallback']]
    report = dict(status='pass', organoids=summary, objects=len(frame), methods=36, source_raw_files_unchanged=True,
                  physical_measurements_unchanged=True, frozen_sources_unchanged=True, complete_finite_vectors=True,
                  independent_cpu_relative_errors=errors, subvoxel_fragment_preprocessing_exceptions=fallback,
                  caveat='Numerical and data-integrity validation; no manual segmentation accuracy or biological phenotype validation.')
    save = out/'validation.json'
    save.write_text(json.dumps(report, indent=2))
    text = '# Five-organoid analysis\n\n| Organoid | Candidate objects | QC passing |\n|---|---:|---:|\n'
    text += ''.join(f"| {s['organoid']} | {s['objects']} | {s['qc_passing']} |\n" for s in summary)
    text += f'\nTotal: {len(frame)} candidate objects with all 36 transferable comparison spaces. All five raw copies are hash-identical to the originals. Measurements and original surfaces were preserved.\n\n'
    text += f'{len(fallback)} objects needed explicitly recorded conservative occupied-voxel transfer because the standard normalized crop resampling lost their tiny foreground. Inspect these preprocessing exceptions before interpreting their image embeddings.\n\n'
    text += 'Joint `cell_measurements.csv` and `organoid_measurements.csv` tables accompany the per-organoid source tables. Organoid totals sum segmented cell measurements; they are not the outer tissue surface area or lumen-inclusive organoid volume. `representations/vectors.npz` contains all 36 spaces with explicit cell IDs.\n\n'
    text += 'Open `atlas_dataset/` with Command-O in Atlas. PCA and clusters are exploratory; models were not retrained. The old 48-cell CAJAL validation reference remains scoped to the original dataset.\n'
    (folder/'RESULTS.md').write_text(text)
    # Analysis-ready tables retain only actual measurements. Missing solidity
    # remains blank here; imputation is confined to representation inputs.
    measurements = []
    for c in data['cells']:
        measurements.append(dict(cell_id=c['id'], organoid=c['sample'].replace('GFP24h_', ''),
            sample=c['sample'], label=c['label'], qc_pass=c['qc'], quality_flags=c['flags'], **c['metrics']))
    columns = ['cell_id', 'organoid', 'sample', 'label', 'qc_pass', 'quality_flags']
    columns += sorted(set().union(*(r.keys() for r in measurements))-set(columns))
    with (folder/'cell_measurements.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(measurements)
    organoids = []
    for s in summary:
        cells = [c for c in data['cells'] if c['sample'] == 'GFP24h_'+s['organoid']]
        qc = [c for c in cells if c['qc']]
        organoids.append(dict(organoid=s['organoid'], candidate_objects=len(cells), qc_passing=len(qc),
            total_cell_volume_um3=sum(c['metrics']['volume_um3'] for c in cells),
            sum_cell_surface_area_um2=sum(c['metrics']['surface_area_um2'] for c in cells),
            qc_mean_cell_volume_um3=float(np.mean([c['metrics']['volume_um3'] for c in qc])) if qc else '',
            qc_mean_sphericity=float(np.mean([c['metrics']['sphericity'] for c in qc])) if qc else '',
            qc_mean_elongation=float(np.mean([c['metrics']['elongation'] for c in qc])) if qc else ''))
    with (folder/'organoid_measurements.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(organoids[0]))
        writer.writeheader(); writer.writerows(organoids)
    launcher = folder/'Open in Organoid Atlas.command'
    launcher.write_text('#!/bin/zsh\nset -eu\nanalysis_folder="${0:A:h}"\nopen -n "/Applications/Organoid Atlas.app" --args "$analysis_folder/atlas_dataset"\n')
    launcher.chmod(0o755)
    code = folder/'code'
    code.mkdir(exist_ok=True)
    snapshots = {}
    for name in ['run_five_organoids.py', 'run_joint_representation_pipeline.py',
                 'infer_joint_organoid_methods.py', 'encode_standalone_atlas.py',
                 'export_standalone_atlas.py', 'validate_joint_organoids.py']:
        source = ROOT/'scripts'/name
        (code/name).write_bytes(source.read_bytes())
        snapshots[name] = sha256(source)
    (code/'snapshot_hashes.json').write_text(json.dumps(snapshots, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--cajal-only', action='store_true')
    args = parser.parse_args()
    if args.cajal_only:
        cajal_checks(args.folder)
    else:
        main(args.folder)
