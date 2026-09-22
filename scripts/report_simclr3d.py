"""Create scientific QC figures and a readable report for this training run."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(__file__).resolve().parents[1]/".cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from simclr3d_core import augment
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/simclr3d'

def main():
    torch.set_num_threads(2)
    meta=pd.read_csv(OUT/'manifest.csv');masks=np.load(OUT/'masks64.npy',mmap_mode='r')
    selected=[0,1,2];x=torch.from_numpy(np.array(masks[selected])).unsqueeze(1).float();g=torch.Generator().manual_seed(55)
    a=augment(x,g).numpy()[:,0];b=augment(x,g).numpy()[:,0]
    fig,axes=plt.subplots(3,3,figsize=(8,8))
    for r,i in enumerate(selected):
        for c,arr in enumerate([masks[i],a[r],b[r]]):
            axes[r,c].imshow(arr.max(0),cmap='gray',vmin=0,vmax=1);axes[r,c].axis('off')
            axes[r,c].set_title(f'{meta.iloc[i].cell_id}: '+['Input','View A','View B'][c])
    fig.suptitle('64³ binary masks — Z maximum projections\nIndependent 3D rotations, small scale changes and translations');fig.tight_layout();fig.savefig(OUT/'augmentation_preview.png',dpi=160);plt.close(fig)
    status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {'status':'starting'}
    extra=''
    if (OUT/'training_history.csv').exists():
        hist=pd.read_csv(OUT/'training_history.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(hist.epoch,hist.train_loss,label='Training');v=hist.dropna(subset=['validation_loss']);axes[0].plot(v.epoch,v.validation_loss,'o-',label='Validation');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='NT-Xent loss')
        axes[1].plot(v.epoch,v.validation_pair_retrieval*100,'o-');axes[1].set(xlabel='Epoch',ylabel='Paired-view retrieval (%)',title='Within validation minibatches')
        fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    if status['status']=='complete':
        from sklearn.decomposition import PCA
        h=np.load(OUT/'embeddings.npy');qc=meta.shape_qc_pass.to_numpy();pca=PCA(n_components=2).fit(h[qc]);xy=pca.transform(h)
        fig,ax=plt.subplots(figsize=(7,5));s=ax.scatter(xy[qc,0],xy[qc,1],c=meta.original_volume_um3[qc],s=12,cmap='viridis');fig.colorbar(s,ax=ax,label='Original volume (µm³)');ax.set(xlabel='Embedding PC1',ylabel='Embedding PC2',title='Learned shape embeddings — quality-passing objects');fig.tight_layout();fig.savefig(OUT/'embedding_pca.png',dpi=160);plt.close(fig)
        sim=h@h.T;rows=[]
        for i in np.where(qc)[0]:
            eligible=np.where(qc&(meta['sample'].to_numpy()!=meta.iloc[i]['sample']))[0]
            nearest=eligible[np.argsort(sim[i,eligible])[-5:][::-1]]
            for rank,j in enumerate(nearest,1):rows.append(dict(cell_id=meta.iloc[i].cell_id,neighbor=meta.iloc[j].cell_id,rank=rank,cosine_similarity=float(sim[i,j])))
        pd.DataFrame(rows).to_csv(OUT/'nearest_cells_across_samples.csv',index=False)
        extra=f"Completed {status['epochs']} epochs; selected epoch {status['best_epoch']} by validation loss. All {len(meta)} objects have 128-dimensional encoder embeddings.\n\nThese are an exploratory representation, not validated cell types or biological classes. Near-identical cells can be treated as negatives by SimCLR, especially if stacks contain related observations.\n"
    text=f'''# 3D cell shape encoder: ResNet + SimCLR

Run status: **{status['status']}**. Live status is in `status.json`; epoch details are in `training.log` and `training_history.csv`.

{extra}
## Inputs and preprocessing

All 1,088 predicted cell masks from 108 stacks were extracted, cropped, resampled with nearest-neighbor interpolation to isotropic spacing, padded to a rotation-safe cubic field of view and resized to 64 × 64 × 64. Linear resizing with antialiasing for downsampling is thresholded at 0.5 to retain binary inputs. The cube follows each object's bounding-box diagonal, so relative scale is normalized; absolute size is retained as `original_volume_um3` and physical field of view in `manifest.csv`.

Input masks are original cpsam_v2/u-Segment3D predictions. Reference annotations are never network inputs or targets. The strong smoothing exports are separate meshes; they are not voxelized into these CNN inputs. Isotropic resampling cannot recover subvoxel information. Physical calibration inherits the prior reconstruction's assumed source Z/Y/X spacing of 1.0/0.1733/0.1733 µm, which was not confirmed in TIFF metadata.

## Training

Small residual 3D encoder (channels 8/16/32/64; GroupNorm; 128-dimensional embedding), plus a 64-dimensional projection head. Two independently rotated, translated and slightly scaled binary views are trained with NT-Xent contrastive loss, temperature 0.2; AdamW; 50 epochs; batch size 16; fixed recorded seed. Arbitrary rotations remove orientation as a target feature. No eroding, cutting away cell regions, or manual class labels are used. Exported embeddings come from the encoder, not the projection head.

674 quality-passing cells train the model; 267 are validation objects; 44 lie in a four-stack buffer; 103 quality-flagged objects are excluded from fitting and validation. All receive embeddings. Split membership is by whole stack in filename order. Biological independence is unknown: this is a diagnostic validation split, not an independent-organoid benchmark. Do not interpret individual objects as independent biological replicates.

Validation uses fixed independent augmentations for comparable checkpoints. Paired-view retrieval is within minibatches, with chance approximately 1/31 for a full batch; it is not cell classification accuracy. The best checkpoint is selected using validation loss; no separate test cohort is available.

## Files and use

- `best.pt`, `last.pt`: selected model and resumable last checkpoint, including configuration and optimizer state.
- `masks64.npy`, `manifest.csv`, `dataset.json`: every prepared object, provenance, calibration and split.
- `embeddings.npy`: unit-normalized encoder embeddings in manifest row order; `embeddings_raw.npy`: unnormalized features; `embeddings.csv`: IDs, physical volume, QC and embedding dimensions.
- `nearest_cells_across_samples.csv`: five nearest quality-passing objects in other stacks, using cosine similarity (created after training).
- `augmentation_preview.png`, `training_curves.png`, `embedding_pca.png`: QC figures; PCA is descriptive, not evidence of discrete cell types.

Resume training from the project folder with `.venv/bin/python -u scripts/train_simclr3d.py`. On this Mac the process needs access to MPS outside the restricted execution sandbox. Do not regenerate the input dataset or change configuration during a resumed run. Refresh this report with `.venv/bin/python scripts/report_simclr3d.py`.

Follow progress in Terminal:

```sh
tail -f results/simclr3d/training.log
```

Method reference: [SimCLR](https://arxiv.org/abs/2002.05709), adapted here to binary 3D cell shapes.
'''
    (OUT/'REPORT.md').write_text(text)
    sm=pd.read_csv(ROOT/'results/smoothing_all/metrics.csv');good=sm[sm.enclosed_volume_reliable];ret=sm[sm.status!='smoothed']
    (ROOT/'results/smoothing_all/REPORT.md').write_text(f'''# Strong smoothing of all predicted cell meshes

Applied the X001 pilot's stronger setting: Taubin λ=0.5, ν=0.53, 100 alternating passes. All {len(sm)} original meshes were processed independently; {len(sm)-len(ret)} have smoothed exports and {len(ret)} retain originals because the strong setting degenerates or severely distorts tiny fragments. Exceptions: {', '.join(ret.cell_id)}.

Median enclosed-volume change among watertight positive-volume meshes: {good.volume_change_pct.median():+.3f}%; median surface-area change: {good.area_change_pct.median():+.2f}%. These statistics include safeguarded originals. Area and sphericity depend on smoothing and must not be mixed across measurement methods. Nonwatertight meshes do not provide reliable enclosed volumes. Per-object results and displacement are recorded in metrics.csv.

Each sample folder contains PLY meshes in physical XYZ µm and JSON provenance. Original meshes and segmentation TIFFs remain unchanged. Strong smoothing exports are visualization surfaces; CNN inputs are separately prepared from predicted binary masks. The operation smooths the sampled boundary and does not add microscope resolution. Safeguard: retain original if the result is nonfinite, degenerate, loses more than half its area or changes a watertight positive enclosed volume by over 10%.
''')
    print('Updated reports and figures')
if __name__=='__main__':main()
