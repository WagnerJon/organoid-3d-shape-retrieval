"""Verify Atlas vectors against cached source encoding and training-only fits."""
import json
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,sha256
from atlas_shape_space import pack
from spatial_benchmark_core import morphology,CLASSICAL,CONTROLS,summary_features
from evaluate_spatial_benchmark import residual_apply


def validate(data=None):
    folder=ROOT/'results/atlas_spatial'
    if data is None:data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text())
    export=json.loads((folder/'export.json').read_text());vectors=np.load(folder/'comparison_vectors.npz')
    manifest=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv').set_index('cell_id').loc[vectors['ids']].reset_index()
    train=manifest.index[manifest.cell_id.isin(export['training_ids'])].to_numpy();frame=manifest.iloc[train]
    assert len(train)==96
    assert sha256(ROOT/'results/spatial_benchmark/aligned_pca.npz')==export['checkpoint_sha256']
    sig=vectors['signatures'];orbit=vectors['orbits']
    expected={'spatial_robust':pack(vectors['gaps'],orbit,sig),'surface_signature':sig,
              'surface_signature_adjusted':residual_apply(manifest,sig,dict(np.load(folder/'signature_nuisance.npz'))),
              'spatial_adjusted':residual_apply(manifest,summary_features(orbit),dict(np.load(folder/'spatial_nuisance.npz')))}
    specs=[('morph_controls',CONTROLS),('morph_size',['volume_um3','surface_area_um2','major_axis_um','middle_axis_um','minor_axis_um']),
           ('morph_shape',['sphericity','elongation','flatness','mesh_solidity']),('morph_all',CLASSICAL)]
    for key,columns in specs:
        x=morphology(manifest,columns);expected[key]=(x-x[train].mean(0))/x[train].std(0).clip(1e-6)
    from evaluate_spatial_benchmark import residual_fit
    expected['morph_adjusted']=residual_apply(manifest,expected['morph_all'],residual_fit(frame,expected['morph_all'][train]))
    lookup={c['id']:c for c in data['cells']}
    for key,x in expected.items():
        actual=np.array([lookup[c]['representations'][key]['vector'] for c in vectors['ids']])
        np.testing.assert_allclose(actual,x,rtol=1e-10,atol=1e-10,err_msg=key)
    assert all('spatial_robust' not in o['representations'] for o in data['organoids'])
    print(f'PASS: {len(expected)} Atlas comparison spaces exactly match source encodings and training-only transforms.')


if __name__=='__main__':validate()
