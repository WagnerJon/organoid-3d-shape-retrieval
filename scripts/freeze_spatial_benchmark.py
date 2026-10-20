"""Freeze development-selected model and analysis before opening test geometry."""
import json
import shutil
from importlib.metadata import version
import pandas as pd
from pointnet_shape_core import ROOT,save_json,sha256
from prepare_spatial_benchmark import OUT


def freeze():
    if (OUT/'freeze.json').exists():raise FileExistsError('Selection already frozen')
    metrics=json.loads((OUT/'development/metrics.json').read_text())
    candidates=[kind for kind in ['pca','cnn'] if metrics['pose'][kind+'_aligned']['rotation_identity']>=.95]
    if not candidates:raise RuntimeError('No model passed the prespecified rotation gate; do not open test')
    chosen=min(candidates,key=lambda kind:metrics['reconstruction'][kind+'_aligned']['squared_chamfer'])
    original=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    test=original[original.split.eq('test')&original.training_eligible]
    dev=pd.read_csv(OUT/'dev_manifest.csv');assert not set(test.similarity_group)&set(dev.similarity_group)
    # Audit identities and exposure only; do not inspect earlier test scores.
    earlier=pd.read_csv(ROOT/'results/boundary_shape/manifest.csv')
    previous=set(earlier.loc[earlier.split=='test','source_cell_id'].dropna())
    sources=['spatial_benchmark_core.py','prepare_spatial_benchmark.py','train_spatial_benchmark.py',
        'evaluate_spatial_benchmark.py','freeze_spatial_benchmark.py','spatial_sdf.py','sdf_shape_core.py',
        'pointnet_shape_core.py','evaluate_spatial_sdf.py','evaluate_sdf_shape.py']
    snapshot=OUT/'frozen_source';snapshot.mkdir()
    for name in sources:shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    nuisance='development/'+chosen+'_nuisance_fit.npz'
    artifacts=['aligned_pca.npz','aligned_cnn.pt',nuisance,'dev_manifest.csv','dev.json',
        '../spatial_sdf/pca95_encoder.npz','../spatial_sdf/boundary/best.pt']
    save_json(OUT/'freeze.json',dict(selected=chosen,selection_rule='Lowest validation base Chamfer among aligned encoders passing >=95% exact-rotation identity; retrieval and test do not choose model.',
        development_metrics_sha256=sha256(OUT/'development/metrics.json'),
        test_cell_ids=test.cell_id.tolist(),test_groups=sorted(map(int,test.similarity_group.unique())),
        test_manifest_sha256=sha256(ROOT/'results/pointnet_shape/data/manifest.csv'),
        test_history=dict(previous_project_test_cell_overlap=len(set(test.cell_id)&previous),test_cells=len(test),
            interpretation='Held out from training and selection of these spatial encoders. Earlier project experiments used test groups; not a globally untouched dataset.'),
        nuisance_files={chosen:nuisance},primary_comparison=chosen+'_residual vs nine_measures',
        sources={name:sha256(ROOT/'scripts'/name) for name in sources},artifacts={name:sha256(OUT/name) for name in artifacts},
        versions={name:version(name) for name in ['numpy','scipy','torch','scikit-learn','libigl']},test_geometry_opened=False))
    print(f'FROZEN {chosen}; {len(test)} test cells remain unopened until the test stage.',flush=True)


if __name__=='__main__':freeze()
