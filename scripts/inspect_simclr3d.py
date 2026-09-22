"""Inspect learned shape variation without interpreting embeddings as cell classes."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/simclr3d'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

m=pd.read_csv(OUT/'manifest.csv');h=np.load(OUT/'embeddings.npy');raw=np.load(OUT/'embeddings_raw.npy')
a=pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id').loc[m.cell_id].reset_index()
qc=m.shape_qc_pass.to_numpy();ids=np.where(qc)[0];pca=PCA().fit(h[qc]);xy=pca.transform(h)
rawpca=PCA().fit(raw[qc]);v=pca.explained_variance_ratio_;rank=float(np.exp(-np.sum(v*np.log(v+1e-20))))
correlations={name:float(spearmanr(xy[qc,0],a.loc[qc,name]).statistic) for name in ['volume_um3','sphericity','aspect_ratio','elongation','flatness']}
correlations['foreground_voxels']=float(spearmanr(xy[qc,0],m.loc[qc,'foreground_voxels']).statistic)
# Whole-stack similarity is descriptive; cells/stacks may not be independent biological observations.
rows=[]
for sample,g in m[qc].groupby('sample'):
    ii=g.index.to_numpy();cent=h[ii].mean(0);cent/=np.linalg.norm(cent)
    rows.append(dict(sample=sample,cells=len(ii),mean_pc1=float(xy[ii,0].mean()),mean_pc2=float(xy[ii,1].mean()),mean_cosine_distance_to_centroid=float((1-h[ii]@cent).mean()),mean_volume_um3=float(a.loc[ii,'volume_um3'].mean())))
samples=pd.DataFrame(rows);samples.to_csv(OUT/'sample_embedding_summary.csv',index=False)
summary=dict(objects=len(m),quality_passing=int(qc.sum()),finite_embeddings=bool(np.isfinite(h).all()),normalized_pc_variance=v[:5].tolist(),raw_pc1_variance=float(rawpca.explained_variance_ratio_[0]),normalized_effective_rank=rank,pc1_spearman=correlations)
(OUT/'embedding_diagnostics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,key,label in zip(axes,['aspect_ratio','sphericity','volume_um3'],['Axis aspect ratio','Sphericity (unsmoothed)','Original volume (µm³)']):
    sc=ax.scatter(xy[qc,0],xy[qc,1],c=a.loc[qc,key],s=10,cmap='viridis');fig.colorbar(sc,ax=ax,label=label);ax.set(xlabel=f'PC1 ({v[0]:.1%})',ylabel=f'PC2 ({v[1]:.1%})')
fig.suptitle('Learned embeddings: relation to measured cell properties');fig.tight_layout();fig.savefig(OUT/'embedding_shape_relations.png',dpi=160);plt.close(fig)
# Examples span the central shape distribution rather than selecting only extreme fragments.
order=ids[np.argsort(xy[ids,0])];anchors=[order[int(q*(len(order)-1))] for q in [.1,.5,.9]]
masks=np.load(OUT/'masks64.npy',mmap_mode='r');sim=h@h.T
fig=plt.figure(figsize=(12,9));pairs=[]
for row,i in enumerate(anchors):
    eligible=ids[m.loc[ids,'sample'].to_numpy()!=m.iloc[i]['sample']];ranked=eligible[np.argsort(sim[i,eligible])]
    cols=[i,ranked[-1],ranked[-2],ranked[0]]
    for col,j in enumerate(cols):
        ax=fig.add_subplot(3,4,row*4+col+1,projection='3d');verts,faces,_,_=marching_cubes(masks[j],.5,step_size=2)
        mesh=Poly3DCollection(verts[:,::-1][faces],facecolor=['#668db3','#68b5a0','#68b5a0','#df947a'][col],edgecolor='none');ax.add_collection3d(mesh)
        ax.set(xlim=(0,64),ylim=(0,64),zlim=(0,64));ax.set_box_aspect((1,1,1));ax.view_init(25,40);ax.set_axis_off()
        title=['Query','Nearest in other stack','Second nearest','Most distant'][col]
        ax.set_title(f'{title}\n{m.iloc[j].cell_id}'+(f' | cosine {sim[i,j]:.3f}' if col else ''),fontsize=10)
        pairs.append(dict(query=m.iloc[i].cell_id,role=title,cell_id=m.iloc[j].cell_id,cosine=float(sim[i,j])))
fig.suptitle('Examples from quality-passing cells — standardized binary shapes\nFixed viewing direction; the encoder is trained to ignore 3D orientation',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'cell_similarity_examples.png',dpi=160);plt.close(fig)
pd.DataFrame(pairs).to_csv(OUT/'cell_similarity_examples.csv',index=False)
(OUT/'NEXT_STEPS.md').write_text(f'''# Interpretation after the first training run

Training completed 50 epochs, selecting epoch 50. All 1,088 cells have finite 128-dimensional encoder embeddings; 985 passed the prior shape QC. See training_history.csv for the complete trace.

The validation paired-view retrieval rose from 6.4% before training to 42.7%; the contrastive loss fell from 3.416 to 1.640. Retrieval measures recognition of an augmented view within a minibatch, not cell classification or biological validity.

The normalized embeddings have PC1 variance {v[0]:.1%} and effective rank {rank:.2f}; raw encoder features have PC1 variance {rawpca.explained_variance_ratio_[0]:.1%}. This indicates a concentrated representation, so 128 exported coordinates should not be interpreted as 128 independent biological features. Further validation is needed before clustering into cell types.

PC1 Spearman associations (sign depends on PCA orientation): {json.dumps(correlations)}. Associations are descriptive and may reflect repeated related cells across stacks. Size was normalized during preprocessing, so a remaining volume association may reflect shape-size correlations rather than direct size encoding.

Next: inspect cell_similarity_examples.png and the nearest-cell list to judge whether similarities match the microscopy. Combine the learned shape representation with physical volume and conventional morphology for the original size-and-shape question. Confirm which stacks belong to the same organoid, animal or time series before testing differences statistically. A second seed and a genuinely independent biological holdout should precede claims of stable clusters or cell phenotypes.

Strong smoothing was applied to exported meshes; this model was trained on the separate predicted binary-mask inputs. Smoothed surfaces were not rasterized as training inputs.
''')
