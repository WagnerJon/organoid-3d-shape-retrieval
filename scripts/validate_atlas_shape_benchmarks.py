"""Check exact source readouts, partial coverage and preservation of prior analyses."""
import json
import numpy as np
from export_atlas_shape_benchmarks import OUT, ATLAS, readouts, benchmark_readouts
from pointnet_shape_core import ROOT, sha256


def validate(data=None):
    if data is None:
        data = json.loads((ATLAS / 'dataset.json').read_text())
    frame, methods, sources = readouts()
    lookup = {c['id']: c for c in data['cells']}
    ids = set(frame.cell_id)
    assert len(ids) in (609, 1088) and len(methods) == 16
    assert data['shapeBenchmarks']['sources'] == sources
    benchmark_frame, benchmark_methods, _ = benchmark_readouts()
    assert data['shapeBenchmarks']['benchmarkSplits'] == dict(zip(benchmark_frame.cell_id, benchmark_frame.split))
    if len(ids) == 1088:
        from extend_atlas_shapes import verify_frozen
        verify_frozen()
        assert data['shapeBenchmarks']['fullCoverage']
        full = ROOT/'results/atlas_full_shape'; missing = frame[frame.benchmark_index < 0]
        raw_sources = {}
        for arm in ('pretrained', 'head_decoder'):
            raw_sources['allen_'+arm] = (full/'allen_codes.npz', arm, ROOT/'results/allen_reference/evaluation'/(arm+'_readout_fit.npz'))
        for arm in ('pretrained', 'adapted'):
            for kind in ('native', 'pose_average'):
                raw_sources['point_mae_'+arm+'_'+kind] = (full/('point_mae_'+arm+'.npz'), kind, ROOT/'results/point_mae'/(arm+'_'+kind+'_fit.npz'))
        for kind in ('invariant', 'cnn'):
            raw_sources['shapeembed3d_'+kind] = (full/('lite_'+kind+'.npz'), 'mean', ROOT/'results/shapeembed3d'/(kind+'_fit.npz'))
        from evaluate_spatial_benchmark import residual_apply
        for key, (path, field, fit_path) in raw_sources.items():
            source = np.load(path); fit = dict(np.load(fit_path))
            assert source['ids'].tolist() == missing.cell_id.tolist()
            raw = np.array([lookup[c]['representations'][key+'_raw']['vector'] for c in missing.cell_id])
            np.testing.assert_array_equal(raw, source[field])
            # Preserve source dtype during standardization, as in the original benchmark.
            raw = raw.astype(source[field].dtype)
            expected = residual_apply(missing, (raw-fit['center'])/fit['scale'], fit)
            actual = np.array([lookup[c]['representations'][key+'_adjusted']['vector'] for c in missing.cell_id])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for meta, values in methods:
        key = meta['id']
        spec = next(m for m in data['methods'] if m['id'] == key)
        assert spec['dimensions'] == values.shape[1] and spec['scope'] == 'cells'
        actual = np.array([lookup[c]['representations'][key]['vector'] for c in frame.cell_id])
        np.testing.assert_array_equal(actual, values)
        assert {c['id'] for c in data['cells'] if key in c['representations']} == ids
        assert all(key not in organoid['representations'] for organoid in data['organoids'])
        for cid in ids:
            rep = lookup[cid]['representations'][key]
            assert len(rep['point']) == 2 and np.isfinite(rep['point']).all()
            assert 0 <= rep['cluster'] < 6
    export = json.loads((OUT / 'export.json').read_text())
    backup = ROOT / export['backup']
    assert sha256(backup) == export['before']
    old = json.loads(backup.read_text())
    assert {m['id'] for m in old['methods']} <= {m['id'] for m in data['methods']}
    assert old['samples'] == data['samples']
    assert len(old['cells']) == len(data['cells'])
    for cell in old['cells']:
        current = lookup[cell['id']]
        for key, value in cell.items():
            if key != 'representations':
                assert current[key] == value, (cell['id'], key)
        for key, rep in cell['representations'].items():
            np.testing.assert_array_equal(current['representations'][key]['vector'], rep['vector'])
    print(f'PASS: 16 methods × {len(ids)} genuine codes; exact fixed-fit and original benchmark parity; finite projections; cell-only scope; prior vectors, measurements and membership preserved.')


if __name__ == '__main__':
    validate()
