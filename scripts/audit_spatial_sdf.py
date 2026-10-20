"""Cross-check compiled mesh distances with cached grids and prior SDF targets."""
import numpy as np
import igl
from spatial_sdf import OUT,load,EXTENT
from pointnet_shape_core import ROOT,save_json
from prepare_pointnet_shape import read_mesh


def audit():
    manifest,arrays=load();axis=np.linspace(-EXTENT,EXTENT,32,dtype=np.float32)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3).astype(np.float64);rows=[]
    for index,row in manifest.iterrows():
        v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);v=(v-center)/row.normalization_radius_um
        def distance(q):return igl.signed_distance(q.astype(np.float64),v.astype(np.float64),f.astype(np.int64),igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].clip(-.15,.15)
        observed=arrays['target'][index].clip(-.15,.15);predicted=distance(arrays['query'][index]);cached=arrays['grids'][index,0].reshape(-1)
        expected_grid=distance(grid)
        result=dict(index=index,cell_id=row.cell_id,query_max_error=float(np.abs(observed-predicted).max()),
            grid_max_error=float(np.abs(cached-expected_grid).max()),
            query_sign_disagreements=int((((observed<0)!=(predicted<0))&(np.abs(observed)>1e-4)).sum()),
            grid_sign_disagreements=int((((cached<0)!=(expected_grid<0))&(np.abs(cached)>1e-4)).sum()))
        rows.append(result)
    valid=all(row['query_max_error']<1e-4 and row['grid_max_error']<1e-4 and row['query_sign_disagreements']==0 and row['grid_sign_disagreements']==0 for row in rows)
    save_json(OUT/'distance_audit.json',dict(passed=valid,cells=len(rows),records=rows,
        query_checks='All 3072 original SDF targets per cell',grid_checks='All 32768 cached grid values per cell',
        tolerance=1e-4,near_zero_sign_tolerance=1e-4))
    if not valid:raise ValueError('Distance audit failed; inspect per-cell records before training')
    print(f"AUDIT PASSED: {len(rows)} cells, max query error {max(r['query_max_error'] for r in rows):.3g}, max grid error {max(r['grid_max_error'] for r in rows):.3g}",flush=True)


if __name__=='__main__':audit()
