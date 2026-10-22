"""Add review flags and closed-mesh solidity without altering segmentation labels."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,json,math,shutil,hashlib
from pathlib import Path
import numpy as np,trimesh,tifffile
from run_standalone_cellpose import native_plane,save


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);a=ap.parse_args();p=a.folder
    cfg=json.loads((p/'run.json').read_text());summary=json.loads((p/'summary.json').read_text());labels=np.load(p/'labels_processed.npy',mmap_mode='r')
    source=p/'cell_measurements_pipeline.csv'
    if not source.exists():shutil.copyfile(p/'cell_measurements.csv',source)
    rows=list(csv.DictReader(source.open()));expected={int(r['label']) for r in rows};actual=set(map(int,np.unique(labels)))-{0};assert expected==actual
    meshes=list((p/'meshes').glob('cell_*.ply'));assert len(meshes)==len(rows)==summary['predictedObjects']
    counts={};flagged=[]
    for r in rows:
        flags=[];cid=int(r['label']);closed=r['mesh_watertight']=='True';volume=float(r['volume_um3'])
        if int(r['voxel_count'])<100:flags.append('small_fragment_under_100_processed_voxels')
        if not 0<float(r['sphericity'])<=1:flags.append('sphericity_outside_0_to_1')
        if r['touches_border']=='True':flags.append('touches_image_border')
        if not closed:flags.append('non_watertight_mesh')
        mesh=trimesh.load(p/'meshes'/f'cell_{cid:04d}.ply',process=False);assert bool(mesh.is_watertight)==closed
        r['mesh_volume_um3']='';r['mesh_solidity']=''
        if closed:
            mv=abs(float(mesh.volume));r['mesh_volume_um3']=mv
            try:
                hull=abs(float(mesh.convex_hull.volume));solid=mv/hull
                assert hull>0 and math.isfinite(solid)
                r['mesh_solidity']=solid
            except Exception:flags.append('convex_hull_measurement_failed')
        r['equivalent_diameter_um']=(6*volume/np.pi)**(1/3);r['surface_to_volume_per_um']=float(r['surface_area_um2'])/volume
        r['requires_review']=bool(flags);r['quality_flags']=';'.join(flags) if flags else 'none'
        for f in flags:counts[f]=counts.get(f,0)+1
        if flags:flagged.append(dict(label=cid,flags=flags))
    with (p/'cell_measurements.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Verify the native label TIFF at five representative Z planes without
    # loading the 2.4 GB label stack or modifying it.
    checked=sorted({0,cfg['nativeShape'][0]//4,cfg['nativeShape'][0]//2,3*cfg['nativeShape'][0]//4,cfg['nativeShape'][0]-1})
    with tifffile.TiffFile(p/'labels_native.ome.tif') as t:
        assert t.series[0].shape==tuple(cfg['nativeShape']) and t.is_ome and t.series[0].dtype==np.uint32
        for z in checked:np.testing.assert_array_equal(t.pages[z].asarray(),native_plane(labels,cfg['nativeShape'],z))
    result=dict(predictedObjects=len(rows),objectsRequiringReview=len(flagged),flagCounts=counts,flaggedObjects=flagged,
        tinyFragmentThreshold='100 processed voxels (about 100 cubic micrometers); an operational review flag, not a biological size cutoff',
        segmentationLabelsUnchanged=True,excludedObjects=0,nativeGridValidationPlanes=checked,
        solidityPolicy='Mesh volume / convex-hull volume for watertight meshes only; blank for open meshes. Voxel-derived volumes are retained for every object.',
        observation='Middle-plane overlays show stronger detection in bright regions and weaker coverage in faintly labeled regions. No manual accuracy benchmark is available.',
        scriptSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(p/'qc_summary.json',result);(p/'qc_script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (p/'QUALITY_CONTROL.md').write_text(f'''# Quality review

The run completed with {len(rows)} predicted objects. These are candidate objects, not a validated biological cell count.

{len(flagged)} objects carry at least one review flag. All labels and objects are retained; none have been silently removed.

'''+''.join(f'- {key}: {value}\n' for key,value in counts.items())+'''
The <100 processed-voxel flag is an operational way to find tiny fragments, not a validated cell-size threshold. Review these alongside the raw image. Some very small objects have sphericity above one because voxel volume and marching-cubes surface area disagree at that resolution.

`cell_measurements.csv` includes the review flags and additional shape measures. `cell_measurements_pipeline.csv` preserves the original automated export. Mesh solidity and mesh volume are left blank for non-watertight surfaces. Primary voxel-derived volumes are preserved.

The overlays show better coverage in bright regions, with missed or weakly segmented structures in faint regions. This should be treated as a first pass requiring manual inspection. Optional whole-volume binary hole filling was disabled to avoid forcing a possible central lumen into foreground. Native-grid labels are nearest-neighbor resampled from the approximately 1 µm processed grid; they do not recover native-resolution detail.

Calibration is copied from the TIFF. Five native Z planes were checked exactly against the expected resampling. The raw source was hash-verified unchanged by the analysis runner. The segmentation label files have not been modified by this quality review.
''')
    print(json.dumps({k:v for k,v in result.items() if k!='flaggedObjects'},indent=2))

if __name__=='__main__':main()
