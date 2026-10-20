"""Prepare development data only. Test geometry is sealed until freeze.json exists."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,save_json,sha256
from spatial_benchmark_core import mesh_for,surface_frame,orbit_grids

OUT=ROOT/'results/spatial_benchmark'


def prepare_one(item):
    index,row=item;v,f=mesh_for(row);frame=surface_frame(v,f)
    return index,orbit_grids(v,f,frame),frame


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'dev.json').exists():raise FileExistsError('Prepared development experiment exists')
    manifest=pd.read_csv(ROOT/'results/sdf_shape_pilot/manifest.csv')
    assert set(manifest.split)=={'train','validation'}
    records=[];grids=[];centers=[];axes=[];eigenvalues=[];gaps=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for index,fields,frame in pool.map(prepare_one,manifest.to_dict('index').items()):
            grids.append(fields);centers.append(frame[0]);axes.append(frame[1]);eigenvalues.append(frame[2]);gaps.append(frame[3])
    manifest['axis_gap']=gaps;manifest.to_csv(OUT/'dev_manifest.csv',index=False)
    np.save(OUT/'dev_grids.npy',np.asarray(grids));np.savez(OUT/'dev_frames.npz',centers=centers,axes=axes,eigenvalues=eigenvalues)
    save_json(OUT/'dev.json',dict(manifest_sha256=sha256(OUT/'dev_manifest.csv'),grids_sha256=sha256(OUT/'dev_grids.npy'),
        atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),test_geometry_opened=False,
        preprocessing='Area-weighted mesh centroid/covariance; four proper axis-sign frames; radius from prior normalization unchanged.',
        protocol=dict(rotation_trials=8,frame_stress_trials=2,frame_stress_surface_points=2048,near_degenerate_gap=.05,
            cnn_epochs=120,pca_dimensions=95,cnn_dimensions=128,selection='Lowest validation Chamfer among models with exact-rotation identity >=95%; no selection by test or retrieval.',
            retrieval='Cross-group, control distance <=1 training SD, >=5 candidates, cap20; log volume/sphericity/elongation.',
            residualization='Training-only degree-2 polynomial Ridge alpha1 on orbit mean/std, no whitening.',
            primary_test='Selected encoder residual features vs nine classical measures; paired group bootstrap 2000; geometry proxy, not biological ground truth.')))
    print('Prepared 96 train / 48 validation; test geometry unopened.',flush=True)


if __name__=='__main__':prepare()
