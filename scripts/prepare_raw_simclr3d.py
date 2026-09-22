"""Fixed physical-FOV raw crops. Predicted segmentation supplies centres only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates,gaussian_filter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/simclr3d_raw'

def raw_crop(image,center,spacing,fov=40,size=64):
    image=np.asarray(image,dtype=np.float32);spacing=np.asarray(spacing)
    # Antialias axes when downsampling; no artificial increase in native Z resolution is implied.
    sigma=np.maximum((fov/size/spacing-1)/2,0)
    filtered=gaussian_filter(image,sigma)
    axes=[center[k]+((np.arange(size)+.5)/size-.5)*fov/spacing[k] for k in range(3)]
    coords=np.meshgrid(*axes,indexing='ij')
    result=map_coordinates(filtered,coords,order=1,mode='constant',cval=0,prefilter=False)
    outside=np.zeros(result.shape,dtype=bool)
    for k in range(3):outside|=(coords[k]<0)|(coords[k]>image.shape[k]-1)
    return result,float(outside.mean())

def main():
    OUT.mkdir(exist_ok=True);cfg=json.loads((ROOT/'config/simclr3d_raw.json').read_text())
    m=pd.read_csv(ROOT/'results/simclr3d/manifest.csv');a=pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id')
    spacing=np.array(json.loads((ROOT/'config/organoids.json').read_text())['spacing_zyx_um'])
    array=np.lib.format.open_memmap(OUT/'raw64.npy',mode='w+',dtype=np.float32,shape=(len(m),64,64,64));rows=[]
    for sample,g in m.groupby('sample',sort=False):
        folder=Path(g.iloc[0].source_label_file).parent;info=json.loads((folder/'complete.json').read_text());path=Path(info['image'])
        image=tifffile.imread(path);assert image.ndim==3
        lo,hi=np.percentile(image,[1,99.8]);assert hi>lo
        normalized=np.clip((image.astype(np.float32)-lo)/(hi-lo),0,1)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();effective=np.array(info['effective_spacing_zyx_um'])
        for idx,row in g.iterrows():
            cell=a.loc[row.cell_id];center=(cell[['centroid_z_um','centroid_y_um','centroid_x_um']].to_numpy(dtype=float)+.5*effective)/spacing-.5
            crop,outside=raw_crop(normalized,center,spacing,cfg['raw_fov_um']);array[idx]=crop
            q=np.percentile(crop,[10,50,90,99]);extra=dict(raw_image=str(path),raw_sha256=digest,raw_fov_um=cfg['raw_fov_um'],outside_fraction=outside,normalization_low=float(lo),normalization_high=float(hi),intensity_mean=float(crop.mean()),intensity_std=float(crop.std()),intensity_p10=q[0],intensity_p50=q[1],intensity_p90=q[2],intensity_p99=q[3],bright_fraction=float((crop>.5).mean()))
            rows.append(dict(**row.to_dict(),**extra))
        print(f'Prepared {sample}: {g.index.max()+1}/{len(m)}',flush=True)
    array.flush();pd.DataFrame(rows).sort_values('index').to_csv(OUT/'manifest.csv',index=False)
    (OUT/'dataset.json').write_text(json.dumps(dict(objects=len(m),source='raw TIFF intensities, unmasked; predicted segmentation centres only',fixed_fov_um=cfg['raw_fov_um'],spacing_zyx_um=spacing.tolist(),spacing_confirmed_from_metadata=False,normalization='per-stack percentiles 1 and 99.8 clipped to [0,1]',split='identical to mask model',caveats=['Not segmentation-free localization','Neighbour cells and background remain visible','Fixed physical field of view retains size; binary model used per-cell size normalization','Validation stacks may be related to training stacks']),indent=2))
if __name__=='__main__':main()
