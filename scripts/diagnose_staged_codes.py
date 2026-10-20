"""Reproduce auxiliary code/readout diagnostics; not a feature-selection sweep."""
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from pointnet_shape_core import save_json
from prepare_staged_sdf import OUT,load
from staged_sdf_core import ReferenceVectorEncoder,stability
from sdf_shape_core import SDFModel


def main():
    torch.set_num_threads(4);torch.manual_seed(21)
    manifest,arrays=load();ids=manifest.index[manifest.split=='validation'].to_numpy()
    folder=OUT/'decoder_validation'
    with np.load(folder/'codes.npz') as data:
        if not np.array_equal(data['cell_ids'],manifest.iloc[ids].cell_id.to_numpy(str)):raise ValueError('Cell IDs changed')
        z=data['fitted_zero'];fresh=data['fitted_zero_fresh'];rotated=data['fitted_zero_rotated']
    mix=np.random.default_rng(21).normal(size=(128,128)).astype(np.float32)/np.sqrt(128)
    triangle=np.triu_indices(128)
    readouts=dict(norms=lambda a:np.linalg.norm(a,axis=-1),
        mixed_norms128=lambda a:np.linalg.norm(np.einsum('ij,bjc->bic',mix,a),axis=-1),
        gram=lambda a:(a@a.transpose(0,2,1))[:,triangle[0],triangle[1]])
    result={name:stability(fn(z),fn(fresh),fn(rotated)) for name,fn in readouts.items()}
    save_json(folder/'invariant_readout_diagnostic.json',result)
    model=SDFModel('vn');saved=torch.load(OUT/'decoder/best.pt',weights_only=False,map_location='cpu')
    model.decoder.load_state_dict(saved['decoder']);model.eval();errors=[]
    with torch.no_grad():
        for first in range(0,len(ids),8):
            query=torch.tensor(arrays['query'][ids[first:first+8]])
            a=model.decode(torch.tensor(z[first:first+8]),query)
            b=model.decode(torch.tensor(fresh[first:first+8]),query)
            errors.extend((a-b).abs().mean(1).numpy().tolist())
    save_json(folder/'field_consistency.json',dict(mean_same_shape_field_difference_l1=float(np.mean(errors)),
        per_cell_differences=errors,interpretation='Same frozen decoder and cell; disjoint balanced SDF observation sets. Function consistency, not identity.'))
    base=np.random.default_rng(22).normal(size=(16,3));base=base/np.linalg.norm(base,axis=1,keepdims=True)*[1,.7,.4]
    signs=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    p=torch.tensor((base[None]*signs[:,None]).reshape(-1,3),dtype=torch.float32)[None]
    graph=torch.tensor(cKDTree(p[0].numpy()).query(p[0].numpy(),k=20)[1])[None]
    with torch.no_grad():value=float(ReferenceVectorEncoder()(p,graph).norm())
    save_json(folder/'symmetry_diagnostic.json',dict(global_vector_code_norm=value,expected='approximately zero',
        scope='Untrained architecture check on D2-symmetric input; arbitrary nonzero free vector codes need not obey this constraint.'))
    print(json.dumps(dict(readouts=result,field_difference=float(np.mean(errors)),symmetric_vector_code_norm=value)))


if __name__=='__main__':main()
