"""Prepare physical cell surfaces, independent resampled targets and cached PointNet++ neighborhoods."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT, read_config, sample_surface, normalize_cloud, hierarchy, assert_split, sha256, save_json


def read_mesh(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError('Truncated mesh: ' + str(path))
    vertices, faces = struct.unpack('<II', data[:8])
    if len(data) != 8+vertices*24+faces*12:
        raise ValueError('Invalid mesh length: ' + str(path))
    xyz = np.frombuffer(data, dtype='<f4', count=vertices*3, offset=8).reshape(-1, 3)
    triangles = np.frombuffer(data, dtype='<u4', count=faces*3, offset=8+vertices*24).reshape(-1, 3)
    if not len(triangles) or triangles.max() >= len(xyz):
        raise ValueError('Invalid mesh topology: ' + str(path))
    return xyz, triangles


def prepare(config):
    atlas = ROOT/config['atlas_dataset']; output = ROOT/config['output_dir']/'data'
    if (output/'dataset.json').exists():
        raise FileExistsError(f'{output} already exists; use a new output_dir to prepare a different dataset')
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads((atlas/'dataset.json').read_text())
    groups = pd.read_csv(ROOT/config['split_groups']).set_index('sample', verify_integrity=True)
    records = []
    for cell in data['cells']:
        if 'inMainOrganoid' not in cell:
            raise ValueError('Classify main-organoid membership before preparing shape learning')
        if cell['sample'] not in groups.index:
            raise ValueError('Missing grouped split for ' + cell['sample'])
        split = groups.loc[cell['sample']]
        records.append(dict(cell_id=cell['id'], sample=cell['sample'], label=cell['label'],
                            split=split['split'], similarity_group=int(split.similarity_group),
                            training_eligible=bool(cell['qc'] and cell['inMainOrganoid']),
                            in_main_organoid=cell['inMainOrganoid'], shape_qc_pass=cell['qc'],
                            **cell['metrics']))
    manifest = pd.DataFrame(records); assert_split(manifest)
    n, count = len(manifest), config['points']; first, second = config['centers']; k = config['neighbors']
    shapes = dict(points=(n,count,3), targets=(n,count,3), centers1=(n,first), neighbors1=(n,first,min(k,count)),
                  centers2=(n,second), neighbors2=(n,second,min(k,first)))
    arrays = {name:np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=np.float32 if name in {'points','targets'} else np.int32,shape=shape)
              for name,shape in shapes.items()}
    start = time.monotonic()
    for i, row in manifest.iterrows():
        mesh = atlas/row['sample']/f'{row.label}.mesh'
        vertices, triangles = read_mesh(mesh)
        seed = int.from_bytes(hashlib.sha256(f'{config["seed"]}:{row.cell_id}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        points, targets = sample_surface(vertices,triangles,count,rng), sample_surface(vertices,triangles,count,rng)
        points, targets, center, scale = normalize_cloud(points,targets)
        arrays['points'][i] = points; arrays['targets'][i] = targets
        for name, value in hierarchy(points,config['centers'],k).items(): arrays[name][i] = value
        manifest.loc[i,'normalization_radius_um'] = scale
        for axis, value in zip('xyz', center): manifest.loc[i,'normalization_center_'+axis+'_um'] = value
        manifest.loc[i,'source_mesh_sha256'] = sha256(mesh)
        if (i+1)%50 == 0 or i == n-1:
            print(f'PREPARED {i+1}/{n} cell surfaces · {time.monotonic()-start:.1f}s',flush=True)
    for array in arrays.values(): array.flush()
    manifest.to_csv(output/'manifest.csv',index=False)
    save_json(output/'dataset.json',dict(version=1, points=count, embedding_dim=config['embedding_dim'], centers=config['centers'],neighbors=k,
        objects=n,eligible=int(manifest.training_eligible.sum()), eligible_split_counts=manifest[manifest.training_eligible].split.value_counts().to_dict(),
        array_sha256={name:sha256(output/(name+'.npy')) for name in arrays},
        manifest_sha256=sha256(output/'manifest.csv'), atlas_sha256=sha256(atlas/'dataset.json'), split_sha256=sha256(ROOT/config['split_groups']),
        geometry='Unsmoothed original predicted triangle meshes in physical XYZ micrometers, sampled proportional to triangle area.',
        normalization='Input cloud centroid and maximum radius; independently sampled target uses the same transform. Physical volume, area, dimensions and inverse transform retained in manifest.',
        targets='Independent surface resampling, not manual biological ground truth.',
        split='Existing TIFF-similarity connected groups; biological specimen identities remain unknown.',
        training='Only non-excluded, QC-passing cells; excluded/QC-failed cells prepared for inference only.',
        spacing='Inherited inferred physical calibration; no new TIFF verification.'))
    print(json.dumps(manifest[manifest.training_eligible].groupby('split').size().to_dict()),flush=True)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json')
    args=parser.parse_args();prepare(read_config(ROOT/args.config))
