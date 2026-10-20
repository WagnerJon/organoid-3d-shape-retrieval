"""Validate the complete local explorer export against its source experiments."""
from pathlib import Path
import json
import struct
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'results/explorer_dataset'
    data = json.loads((folder/'dataset.json').read_text())
    morph = pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id')
    assert len(data['cells']) == 1088 and len(data['samples']) == 108
    cells = {c['id']: c for c in data['cells']}
    assert len(cells) == len(data['cells'])
    for sample in data['samples']:
        source = folder/sample['id']
        size = np.prod(sample['shape'])
        assert (source/'labels.u16').stat().st_size == size*2
        labels = np.fromfile(source/'labels.u16', dtype='<u2')
        ids = {c['label'] for c in cells.values() if c['sample'] == sample['id']}
        assert set(np.unique(labels)) - {0} <= ids
        if sample['rawAvailable']:
            assert (source/'raw.u8').stat().st_size == size
    for cid, cell in cells.items():
        assert cell['metrics']['volume_um3'] == morph.loc[cid, 'volume_um3']
        content = (folder/cell['sample']/f"{cell['label']}.mesh").read_bytes()
        nv, nf = struct.unpack('<II', content[:8])
        assert len(content) == 8+nv*24+nf*12
        vertices = np.frombuffer(content, dtype='<f4', count=nv*6, offset=8)
        assert np.isfinite(vertices).all()
        indices = np.frombuffer(content, dtype='<u4', offset=8+nv*24)
        assert indices.max() < nv
    if any(m.get('family') for m in data['methods']):
        from validate_atlas_spatial import validate
        validate(data)
    if 'harmonicsCajal' in data:
        from validate_harmonics_cajal import validate as validate_harmonics
        validate_harmonics(data)
    if 'shapeBenchmarks' in data:
        from validate_atlas_shape_benchmarks import validate as validate_shape_benchmarks
        validate_shape_benchmarks(data)
    for method in data['methods']:
        if method.get('family'):
            continue
        if method['id'] == 'traditional':
            continue
        source_path = ROOT/'results'/method['id']/'embeddings.csv'
        if method['id'] == 'pointnet_shape128': source_path = Path(data['geometryModels'][method['id']]['embeddings_csv'])
        source = pd.read_csv(source_path).set_index('cell_id')
        columns = [k for k in source if k.startswith('embedding_')]
        for cid, cell in cells.items():
            vector = source.loc[cid, columns].to_numpy(dtype=float, copy=True)
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            np.testing.assert_allclose(vector, cell['representations'][method['id']]['vector'], atol=6e-8)
    for cell in data['cells']:
        for relative in cell.get('pointReconstructions', {}).values():
            payload = (folder/relative).read_bytes()
            count, = struct.unpack('<I', payload[:4])
            assert count == 2048 and len(payload) == 4+count*12
            assert np.isfinite(np.frombuffer(payload[4:],dtype='<f4')).all()
    print(f'PASS: all 1,088 meshes; 108 raw/label previews; morphology and embeddings from {len(data["methods"])-1} models matched by cell_id; learned point reconstructions validated.')


if __name__ == '__main__':
    main()
