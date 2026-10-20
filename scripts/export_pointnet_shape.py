"""Publish a trained surface encoder and its decoded points into an existing Atlas export."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
import pandas as pd
import torch
from pointnet_shape_core import ROOT,read_config,load_data,tensors,PointCloudAutoencoder,select_device,save_json,sha256
from organoid_membership import rebuild

METHOD='pointnet_shape128'


def export(config,device_override=None):
    torch.set_num_threads(config['threads']);device=select_device(device_override or config['device'])
    data=ROOT/config['output_dir']/'data';run=ROOT/config['output_dir']/'baseline';atlas=ROOT/config['atlas_dataset']
    status=json.loads((run/'status.json').read_text())
    if status['smoke'] or status['status']!='trained':raise ValueError('Only a completed non-smoke baseline may be published')
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    manifest,arrays=load_data(data)
    if checkpoint['manifest_sha256']!=sha256(data/'manifest.csv'):raise ValueError('Model and surface data differ')
    atlas_signature=sha256(atlas/'dataset.json')
    document=json.loads((atlas/'dataset.json').read_text());lookup=manifest.set_index('cell_id')
    revision=sha256(run/'best.pt')[:12]
    row_for={cid:i for i,cid in enumerate(manifest.cell_id)}
    if set(lookup.index)!=set(c['id'] for c in document['cells']):raise ValueError('Atlas / training cell IDs differ')
    for cell in document['cells']:
        original=lookup.loc[cell['id']]
        if sha256(atlas/cell['sample']/f'{cell["label"]}.mesh')!=original.source_mesh_sha256:
            raise ValueError('An original surface changed after preparation; refusing to attach stale shape predictions')
        if bool(original.in_main_organoid)!=cell['inMainOrganoid'] or bool(original.shape_qc_pass)!=cell['qc']:
            raise ValueError('Membership or QC changed during training; prepare and train a new version before publication')
    model=PointCloudAutoencoder(config).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    embeddings=[];clouds=[]
    with torch.no_grad():
        for start in range(0,len(manifest),config['batch_size']):
            ids=np.arange(start,min(len(manifest),start+config['batch_size']))
            points,_,groups=tensors(arrays,ids,device)
            latent,prediction=model(points,groups)
            embeddings.extend(latent.cpu().numpy());clouds.extend(prediction.cpu().numpy())
    embeddings=np.asarray(embeddings);clouds=np.asarray(clouds)
    if not np.isfinite(embeddings).all() or not np.isfinite(clouds).all():raise ValueError('Non-finite predictions')
    np.save(run/'latent128.npy',embeddings);np.save(run/'decoded_normalized.npy',clouds)
    norm=embeddings/np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    table=pd.DataFrame(norm,columns=[f'embedding_{i:03d}' for i in range(128)])
    table.insert(0,'cell_id',manifest.cell_id.to_numpy());table.to_csv(run/'embeddings.csv',index=False)
    for cell in document['cells']:
        index=row_for[cell['id']]
        row=manifest.iloc[index]
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz'])
        physical=(clouds[index]*row.normalization_radius_um+center).astype('<f4')
        relative=f'{METHOD}/{revision}/{cell["sample"]}/{cell["label"]}.points'
        path=atlas/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(struct.pack('<I',len(physical))+physical.tobytes())
        cell.setdefault('pointReconstructions',{})[METHOD]=relative
        cell['representations'][METHOD]=dict(vector=np.asarray(norm[index],dtype=float).round(7).tolist(),point=[0.,0.],cluster=0)
    document['methods']=[m for m in document['methods'] if m['id']!=METHOD]
    document['methods'].append(dict(id=METHOD,name='Surface PointNet++ · 128D',metric='cosine',dimensions=128,variance=[0.,0.],stackVariance=[0.,0.]))
    document.setdefault('membershipStackTemplates',document['organoids'])
    document['organoids']=document['membershipStackTemplates']
    document.setdefault('geometryModels',{})[METHOD]=dict(checkpoint=str(run/'best.pt'),checkpoint_sha256=sha256(run/'best.pt'),
        embeddings_csv=str(run/'embeddings.csv'),data_manifest_sha256=checkpoint['manifest_sha256'],epoch=int(checkpoint['epoch']),rotation_invariant=False,
        input='Normalized surface XYZ only; no microscopy or hand-designed morphometry inputs',decoder='2048 surface points',
        scope='Trained on included QC cells in training similarity groups; all other cells encoded for inference only',
        stage2_decision=json.loads((run/'stage2_decision.json').read_text()) if (run/'stage2_decision.json').exists() else None)
    base=document['provenance'].split(' Main-organoid membership:')[0].split(' Surface PointNet++:')[0]
    document['provenance']=base+' Surface PointNet++: normalized predicted-cell surface geometry only, 128D, grouped training/validation, exploratory point-cloud reconstruction; not rotation invariant. Original physical measurements remain separate.'
    rebuild(document)
    if sha256(atlas/'dataset.json')!=atlas_signature:
        raise ValueError('Atlas changed during inference; no analysis data was replaced. Retry after reviewing membership.')
    save_json(atlas/'dataset.json',document,compact=True)
    print(f'Published {len(manifest)} 128D embeddings and physical point reconstructions to Atlas',flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');parser.add_argument('--device')
    args=parser.parse_args();export(read_config(ROOT/args.config),args.device)
