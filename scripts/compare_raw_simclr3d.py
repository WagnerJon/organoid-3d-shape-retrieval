"""Matched downstream probes; no morphology labels are used to train the encoders."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/simclr3d_raw'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    m=pd.read_csv(OUT/'manifest.csv');bm=pd.read_csv(ROOT/'results/simclr3d/manifest.csv');assert m.cell_id.equals(bm.cell_id) and m.split.equals(bm.split)
    a=pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id').loc[m.cell_id]
    tr=m.split.eq('train').to_numpy();va=m.split.eq('validation').to_numpy()
    raw=np.load(OUT/'embeddings.npy');mask=np.load(ROOT/'results/simclr3d/embeddings.npy');untrained=np.load(OUT/'untrained_embeddings.npy');untrained/=np.maximum(np.linalg.norm(untrained,axis=1,keepdims=True),1e-12)
    intensity=m[['intensity_mean','intensity_std','intensity_p10','intensity_p50','intensity_p90','intensity_p99','bright_fraction']].to_numpy()
    representations={'Crop position/padding control':np.column_stack([m.outside_fraction.to_numpy(),a[['centroid_z_um','centroid_y_um','centroid_x_um']].to_numpy()]),'Mean-only baseline':m[['intensity_mean']].to_numpy(),'Intensity statistics':intensity,'Untrained raw CNN':untrained,'Trained mask CNN':mask,'Trained raw CNN':raw,'Raw + mask CNN':np.concatenate([raw,mask],axis=1)}
    targets=['sphericity','aspect_ratio','elongation','flatness','mesh_solidity','volume_um3'];rows=[]
    for name,x in representations.items():
        for target in targets:
            y=a[target].to_numpy();model=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[.1,1,10,100,1000]},cv=GroupKFold(5),scoring='neg_mean_squared_error',n_jobs=2)
            model.fit(x[tr],y[tr],groups=m.loc[tr,'sample']);pred=model.predict(x[va])
            observed=y[va].copy();centered_pred=pred.copy();groups=m.loc[va,'sample'].to_numpy()
            for sample in np.unique(groups):
                ii=groups==sample;observed[ii]-=observed[ii].mean();centered_pred[ii]-=centered_pred[ii].mean()
            within=float(r2_score(observed,centered_pred))
            rows.append(dict(representation=name,target=target,within_stack_r2=within,r2=float(r2_score(y[va],pred)),mae=float(mean_absolute_error(y[va],pred)),alpha=model.best_params_['ridge__alpha']))
        print('Evaluated '+name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'morphology_probe_comparison.csv',index=False)
    agreement=float(spearmanr(pdist(raw[va],metric='cosine'),pdist(mask[va],metric='cosine')).statistic)
    (OUT/'comparison.json').write_text(json.dumps(dict(validation_cells=int(va.sum()),distance_agreement_spearman=agreement,results=rows),indent=2))
    pivot=frame.pivot(index='target',columns='representation',values='r2').reindex(targets)
    ax=pivot.plot.bar(figsize=(13,6));ax.axhline(0,color='black',lw=.6);ax.set(ylabel='Validation R² (higher is better)',title='Predicting measured morphology from raw vs binary-mask embeddings');ax.legend(fontsize=8);plt.xticks(rotation=20);plt.tight_layout();plt.savefig(OUT/'comparison.png',dpi=160);plt.close()
    hist=pd.read_csv(OUT/'training_history.csv');fig,ax=plt.subplots(figsize=(7,4));ax.plot(hist.epoch,hist.train_loss,label='Training');v=hist.dropna(subset=['validation_loss']);ax.plot(v.epoch,v.validation_loss,'o-',label='Validation');ax.legend();ax.set(xlabel='Epoch',ylabel='Contrastive loss');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    text='''# Raw intensity experiment: completed comparison

Each cell is represented by a fixed 40 µm unmasked raw-intensity cube resampled to 64³. Predicted segmentation is used ONLY to locate the centre. Neighbours remain visible. Per-stack percentile normalization and geometric/photometric augmentation are used. Morphology measurements and binary embeddings are never encoder training targets.

The same training/validation cell IDs are used as in the binary-mask experiment. The raw crop retains absolute size, unlike the size-normalized binary inputs. Therefore shape targets and physical volume must be interpreted separately. Assumed source spacing is 1.0/0.1733/0.1733 µm, not verified TIFF calibration.

Frozen embeddings are assessed with ridge probes predicting six existing morphology measurements. Probe regularization is chosen using five whole-stack folds WITHIN the training subset. Reported R² and MAE are on 267 validation cells. This same diagnostic set selected CNN checkpoints, so it is not an untouched test cohort. Biological independence remains unknown. Negative R² means worse than predicting the validation target mean; scores are not classification accuracy.

Controls include crop position/padding, intensity statistics and the identical untrained raw CNN. Comparison with the trained mask CNN asks whether the raw input encodes similar recoverable morphology. Comparison with controls asks whether self-supervised learning adds information accessible to these probes. Prediction of the same segmentation-derived measurements is evidence of morphological information, not proof of accurate cell segmentation, true shape, cell types, or superior biological utility.

'''
    text+='Within-stack R² subtracts each validation stack mean from both observed and predicted measurements for evaluation only. This tests cell-to-cell variation beyond the shared organoid context; it is not an independently deployable calibration step.\n\n'
    text+=f'Raw-versus-mask pair-distance Spearman correlation on validation cells: {agreement:.3f}. Dependent cell pairs are descriptive, not independent observations.\n\n'
    text+='| Representation | Target | Validation R² | Within-stack R² | MAE |\n|---|---|---:|---:|---:|\n'
    for r in rows:text+=f"| {r['representation']} | {r['target']} | {r['r2']:.3f} | {r['within_stack_r2']:.3f} | {r['mae']:.3f} |\n"
    text+='\nA promising result must still be checked on independent organoids, multiple training seeds, and crop-centre/context controls. Similar retrieval alone is insufficient: the network might identify neighbourhoods or intensity patterns. This experiment is segmentation-assisted localization, not a segmentation-free cell pipeline.\n'
    (OUT/'COMPARISON.md').write_text(text);print('Comparison completed',flush=True)
if __name__=='__main__':main()
