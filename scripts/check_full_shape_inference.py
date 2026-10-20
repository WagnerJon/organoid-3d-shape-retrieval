"""Independent CPU spot checks of frozen full-data inference (no Atlas writes)."""
import json
import time
import numpy as np
import pandas as pd
import torch
from extend_atlas_shapes import OUT, geometry, verify_frozen
from pointnet_shape_core import ROOT, sha256, sample_surface
from prepare_pointnet_shape import read_mesh
from spatial_benchmark_core import SIGNS


def relative(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-12))


def main():
    torch.set_num_threads(1); verify_frozen(); started = time.monotonic()
    frame = pd.read_csv(OUT/'manifest.csv'); missing = frame[frame.benchmark_index < 0]
    assert len(frame) == 1088 and len(missing) == 479 and len(set(frame.cell_id)) == 1088
    row = missing.iloc[0].to_dict(); pool, axes = geometry(row)
    v, f = read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f"{row['label']}.mesh")
    v = v.astype(float); normalized = (v-(v.min(0)+v.max(0))/2)/np.ptp(v, axis=0).max()
    expected = sample_surface(normalized, f, 32768, np.random.default_rng(20261110+int(row['inference_index'])))
    np.testing.assert_array_equal(pool, expected)
    results = {}
    for name, dimensions, fields in [('lite_invariant.npz', 128, ['mean']), ('lite_cnn.npz', 128, ['mean']),
                                     ('point_mae_pretrained.npz', 768, ['native', 'pose_average']),
                                     ('point_mae_adapted.npz', 768, ['native', 'pose_average'])]:
        a = np.load(OUT/name)
        assert a['ids'].tolist() == missing.cell_id.tolist()
        for field in fields:
            assert a[field].shape == (479, dimensions) and np.isfinite(a[field]).all()
    from shapeembed3d_core import ShapeEmbed3D, fps_indices, edm
    candidates = pool[np.random.default_rng(20261022+4*int(row['inference_index'])).choice(32768, 2048, replace=False)]
    points = candidates[fps_indices(candidates[None], 128)[0]]
    for kind in ('invariant', 'cnn'):
        state = torch.load(ROOT/'results/shapeembed3d'/kind/'best.pt', map_location='cpu', weights_only=False)
        model = ShapeEmbed3D(kind).eval(); model.load_state_dict(state['model'], strict=True)
        with torch.no_grad(): actual = model.encode(edm(torch.tensor(points[None])))[0].numpy()[0]
        expected = np.load(OUT/('lite_'+kind+'.npz'))['mean'][0]
        error = relative(actual, expected); assert error < .001, (kind, error)
        results['lite_'+kind] = error
    from point_mae_reference import model as point_model, encode
    from run_point_mae_reference import draw
    cloud = draw(pool, int(row['inference_index']))[None]
    for arm in ('pretrained', 'adapted'):
        model = point_model()
        if arm == 'adapted':
            state = torch.load(ROOT/'results/point_mae/best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
        model.eval()
        with torch.no_grad():
            actual = encode(model, torch.tensor(cloud)).numpy()[0]
            poses = np.mean([encode(model, torch.tensor((cloud@axes*sign).astype(np.float32))).numpy()[0] for sign in SIGNS], axis=0)
        a = np.load(OUT/('point_mae_'+arm+'.npz'))
        for kind, observed in [('native', actual), ('pose_average', poses)]:
            error = relative(observed, a[kind][0]); assert error < .001, (arm, kind, error)
            results['point_mae_'+arm+'_'+kind] = error
    # Re-evaluate saved Allen pre-head features through independently loaded heads.
    from train_allen_reference import CachedAllen
    available = sorted((OUT/'allen').glob('*.npz'))
    assert available
    for arm in ('pretrained', 'head_decoder'):
        model = CachedAllen().eval()
        if arm != 'pretrained':
            state = torch.load(ROOT/'results/allen_reference'/arm/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
        errors = []
        for path in available[:3]:
            a = np.load(path)
            with torch.no_grad(): actual = model.encode(torch.tensor(a['features'][None])).norm(dim=-1).numpy()[0].astype(float)
            error = relative(actual, a[arm]); assert error < 1e-6, (arm, path, error)
            errors.append(error)
        results['allen_'+arm+'_cached_head_max'] = max(errors)
    verify_frozen()
    result = dict(status='pass', source_cell=row['cell_id'], additional_cells=len(missing),
                  relative_errors=results, seconds=time.monotonic()-started, frozen_sources_unchanged=True)
    (OUT/'spot_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
