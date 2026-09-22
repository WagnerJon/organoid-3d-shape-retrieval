"""Create binary, isotropic, cubic 64^3 inputs from every predicted cell label."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.transform import resize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simclr3d'


def standardize(binary,spacing,size=64,padding=1.15):
    binary=np.asarray(binary,dtype=bool)
    if binary.ndim!=3 or not binary.any():raise ValueError('Expected nonempty 3D binary mask')
    bounds=ndimage.find_objects(binary.astype(np.uint8))[0]
    cropped=binary[bounds]
    spacing=np.asarray(spacing,dtype=float)
    pitch=float(spacing.min())
    iso=ndimage.zoom(cropped.astype(np.uint8),spacing/pitch,order=0,prefilter=False)>0
    # Fits the crop bounding-box diagonal, protecting the complete object during arbitrary rotations.
    side=int(np.ceil(np.linalg.norm(iso.shape)*padding))
    pads=[((side-n)//2,side-n-(side-n)//2) for n in iso.shape]
    cube=np.pad(iso,pads)
    soft=resize(cube.astype(np.float32),(size,size,size),order=1,preserve_range=True,anti_aliasing=side>size)
    standardized=(soft>=.5).astype(np.uint8)
    if not standardized.any():raise ValueError('Cell vanished during resampling')
    return standardized,dict(crop_shape_zyx=list(cropped.shape),isotropic_shape_zyx=list(iso.shape),
        isotropic_pitch_um=pitch,cube_side_voxels=side,cube_fov_um=side*pitch,
        normalized_voxel_pitch_um=side*pitch/size,foreground_voxels=int(standardized.sum()))


def main():
    OUT.mkdir(exist_ok=True)
    config=json.loads((ROOT/'config/simclr3d.json').read_text())
    cells=pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').sort_values(['sample','label'])
    paths={p.parent.name:p for root in ['cpsam_v2_first5','cpsam_v2_remaining']
        for p in (ROOT/'results'/root).glob('run_*/*/complete.json')}
    stacks=sorted(paths)
    cut=int(np.floor(len(stacks)*(1-config['validation_stack_fraction'])))
    train=set(stacks[:cut-config['split_gap_stacks']]);val=set(stacks[cut:])
    array=np.lib.format.open_memmap(OUT/'masks64.npy',mode='w+',dtype=np.uint8,
                                 shape=(len(cells),config['input_size'],config['input_size'],config['input_size']))
    rows=[];index=0
    for sample,group in cells.groupby('sample',sort=True):
        p=paths[sample];labels=tifffile.imread(p.parent/'labels_isotropic.tif')
        info=json.loads(p.read_text());spacing=info['effective_spacing_zyx_um']
        source_hash=hashlib.sha256((p.parent/'labels_isotropic.tif').read_bytes()).hexdigest()
        for _,cell in group.iterrows():
            binary=labels==int(cell.label)
            standardized,metadata=standardize(binary,spacing,config['input_size'],config['rotation_safe_padding'])
            array[index]=standardized
            eligible=bool(cell.shape_qc_pass) if config['training_cohort']=='shape_qc_pass' else True
            split=('train' if sample in train else 'validation' if sample in val else 'gap') if eligible else 'excluded_qc'
            rows.append(dict(index=index,cell_id=cell.cell_id,sample=sample,label=int(cell.label),split=split,
                training_eligible=eligible,shape_qc_pass=bool(cell.shape_qc_pass),original_volume_um3=float(cell.volume_um3),
                source_label_file=str(p.parent/'labels_isotropic.tif'),source_sha256=source_hash,**metadata))
            index+=1
        print(f'Prepared {sample}: {index}/{len(cells)} masks',flush=True)
    array.flush();del array
    manifest=pd.DataFrame(rows);manifest.to_csv(OUT/'manifest.csv',index=False)
    counts=manifest.split.value_counts().to_dict()
    metadata=dict(objects=len(rows),config=config,split_counts=counts,train_stacks=sorted(train),validation_stacks=sorted(val),
        gap_stacks=[s for s in stacks if s not in train|val],input_source='original predicted binary masks, not smoothed meshes or reference masks',
        size_normalization=True,physical_size_retained_in_manifest=True,
        validation_caveat='Filename-blocked diagnostic split; biological independence is unknown. Not a held-out organoid benchmark.')
    (OUT/'dataset.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
