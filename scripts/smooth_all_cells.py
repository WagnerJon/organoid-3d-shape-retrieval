"""Apply the accepted 100-pass Taubin pilot to every original cell mesh."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import trimesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/smoothing_all'


def main():
    OUT.mkdir(exist_ok=True)
    source_paths=sorted([p for root in ['cpsam_v2_first5','cpsam_v2_remaining']
        for p in (ROOT/'results'/root).glob('run_*/*/meshes/cell_*.ply')],key=lambda p:(p.parent.parent.name,p.name))
    assert len(source_paths)==1088
    rows=[];start=time.time()
    for index,path in enumerate(source_paths,1):
        sample=path.parent.parent.name;label=int(path.stem.split('_')[-1]);folder=OUT/sample;folder.mkdir(exist_ok=True)
        output=folder/path.name;meta=folder/(path.stem+'.json')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if meta.exists() and output.exists():
            row=json.loads(meta.read_text())
            if row.get('source_sha256')==digest and row.get('safeguard_version')==1:
                rows.append(row);continue
        original=trimesh.load(path,process=False);smoothed=original.copy()
        status='smoothed'
        try:
            trimesh.smoothing.filter_taubin(smoothed,lamb=.5,nu=.53,iterations=100)
            if not np.isfinite(smoothed.vertices).all() or smoothed.area<=1e-12:
                raise ValueError('Degenerate result')
            if smoothed.area < .5*original.area or (original.is_watertight and original.volume>0 and abs(smoothed.volume/original.volume-1)>.1):
                raise ValueError('Tiny fragment distorted by strong smoothing')
        except Exception as error:
            smoothed=original.copy();status='original_retained: '+str(error)
        smoothed.export(output)
        check=trimesh.load(output,process=False)
        assert np.isfinite(check.vertices).all() and np.array_equal(original.faces,check.faces)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        movement=np.linalg.norm(smoothed.vertices-original.vertices,axis=1)
        row=dict(safeguard_version=1,cell_id=f'{sample}:L{label}',sample=sample,label=label,status=status,source=str(path),source_sha256=digest,
            output=str(output),passes=100,lamb=.5,nu=.53,
            original_watertight=original.is_watertight,smoothed_watertight=check.is_watertight,
            original_mesh_volume_um3=float(original.volume),smoothed_mesh_volume_um3=float(smoothed.volume),
            volume_change_pct=float(100*(smoothed.volume/original.volume-1)) if abs(original.volume)>1e-12 else None,
            original_area_um2=float(original.area),smoothed_area_um2=float(smoothed.area),
            area_change_pct=float(100*(smoothed.area/original.area-1)),
            vertex_movement_p95_um=float(np.percentile(movement,95)),vertex_movement_max_um=float(movement.max()),
            enclosed_volume_reliable=bool(original.is_watertight and check.is_watertight and original.volume>0 and smoothed.volume>0))
        meta.write_text(json.dumps(row,indent=2));rows.append(row)
        if index%20==0 or index==len(source_paths):
            print(f'{index}/{len(source_paths)} smoothed: {sample} L{label}; elapsed {(time.time()-start)/60:.1f} min',flush=True)
            pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(objects=len(rows),seconds=time.time()-start,
        original_retained=sum(r['status']!='smoothed' for r in rows),method='Taubin',passes=100,lamb=.5,nu=.53,
        originals_preserved=True,segmentation_masks_modified=False),indent=2))
    print('Completed all smoothing exports',flush=True)


if __name__=='__main__':main()
