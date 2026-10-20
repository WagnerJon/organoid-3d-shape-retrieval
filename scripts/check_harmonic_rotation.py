"""Fresh-mesh rotation probes for harmonic spectra; unchanged triangulation."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial.distance import cdist
from pointnet_shape_core import ROOT,sha256
from spatial_benchmark_core import mesh_for
from harmonics_cajal_core import harmonics,angular_basis
from check_cajal_validation import candidate_sets

OUT=ROOT/'results/harmonics_cajal'


def probe(args):
    i,row=args;path=OUT/'rotation_probes'/f'{i:03d}.npz'
    if path.exists():
        with np.load(path) as a:return a['codes']
    v,f=mesh_for(row);rng=np.random.default_rng(20261102+i);basis=angular_basis();codes=[]
    for trial in range(8):
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        codes.append(harmonics(v@q,f,basis))
    codes=np.asarray(codes);np.savez_compressed(path,codes=codes);return codes


def main():
    frame,_,_=candidate_sets();(OUT/'rotation_probes').mkdir(exist_ok=True)
    source=np.load(OUT/'surfaces.npz');index={cid:i for i,cid in enumerate(source['ids'])};base=source['harmonics'][[index[c] for c in frame.cell_id]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rotated=[]
        for i,codes in enumerate(pool.map(probe,enumerate(frame.to_dict('records')))):
            rotated.append(codes)
            if i%8==0:print(f'Rotation probes {i+1}/{len(frame)} cells',flush=True)
    rotated=np.stack(rotated);identity=[];jaccard=[];groups=frame.similarity_group.to_numpy();d0=cdist(base,base)
    for i in range(len(frame)):
        allowed=np.flatnonzero(groups!=groups[i]);reference=set(allowed[np.argsort(d0[i,allowed])[:5]])
        for code in rotated[i]:
            d=cdist(code[None],base)[0];identity.append(int(d.argmin()==i));neighbors=set(allowed[np.argsort(d[allowed])[:5]])
            jaccard.append(len(reference&neighbors)/len(reference|neighbors))
    result=dict(cells=len(frame),rotations_per_cell=8,identity=float(np.mean(identity)),neighbor_jaccard=float(np.mean(jaccard)),
        mean_code_change=float(np.linalg.norm(rotated-base[:,None],axis=-1).mean()),max_code_change=float(np.linalg.norm(rotated-base[:,None],axis=-1).max()),
        scope='Validation cells only, source mesh rotated before descriptor computation. Finite angular quadrature and radial sampling mean numerical, not exact, invariance.',source_sha256=sha256(OUT/'surfaces.npz'))
    (OUT/'rotation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
