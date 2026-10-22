"""Explicit supplementary general-mapping normalization after distortion audit."""
from diagnose_spherical_shape_study import area_distortion
from spherical_shape_study_core import *

def main():
    d,Y=fib_basis(4096,12);sphere=trimesh.creation.icosphere(subdivisions=4);Ye=basis(sphere.vertices,12)
    records=[];ids=[];features=[];folder=OUT/'balanced_mapping';folder.mkdir(exist_ok=True)
    sources=sorted((OUT/'mapping_cells').glob('*.npz'))+sorted((OUT/'arbitrary_controls').glob('*.npz'))
    for i,path in enumerate(sources):
        name=path.stem;key=('real_'+name if path.parent.name=='mapping_cells' else name)
        z=np.load(path);v,f=z['vertices'],z['faces']
        if path.parent.name=='mapping_cells':
            old=json.loads(path.with_suffix('.json').read_text());cid=old['cell_id'];mapped=z['mapped'];valid=old['mapping']['valid']
        else:
            cid=name;mapped,info=map_sphere(v,f);valid=info['valid']
        row=dict(id=cid,source=key,eligible=valid)
        feat=np.full(13,np.nan)
        if valid:
            mapped,quality=balance_sphere(mapped,v,f);row['balance']=quality
            if quality['valid']:
                _,values,qc=ray_samples(mapped,f,d,attribute=v);c=coefficients(values,Y);reconstruction=(Ye@c).real
                row['reconstruction']=reconstruction_metrics(reconstruction,v,f);row['area_distortion']=area_distortion(v,f,mapped)
                feat=np.array([np.linalg.norm(c[l*l:(l+1)**2]) for l in range(13)]);feat/=max(np.linalg.norm(feat),1e-12)
                np.savez_compressed(folder/f'{key}.npz',reconstruction=reconstruction,mapped=mapped)
        records.append(row)
        if path.parent.name=='mapping_cells':ids.append(cid);features.append(feat)
        if i%8==0:print(f'Balanced mapping {i+1}/{len(sources)}',flush=True)
    save_json(OUT/'balanced_mapping.json',records);np.savez_compressed(OUT/'general_balanced_features.npz',ids=ids,features=features)
if __name__=='__main__':main()
