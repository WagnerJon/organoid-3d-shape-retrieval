"""Report Point-MAE shape discrimination separately from conditional inpainting."""
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import itertools
from scipy.spatial import cKDTree
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from point_mae_reference import OUT,ROOT
from pointnet_shape_core import sha256,sample_surface
from spatial_benchmark_core import mesh_for


def align_neighbor(reference,neighbor,vertices):
    """Same proper-rotation ICP schedule as the proxy; return a display transform."""
    ac=reference.mean(0);bc=neighbor.mean(0);a=reference-ac;b=neighbor-bc
    _,aa=np.linalg.eigh(a.T@a);_,bb=np.linalg.eigh(b.T@b);a=a@aa;b=b@bb;ta=cKDTree(a);best=float('inf');transform=None
    for perm in itertools.permutations(range(3)):
        for sign in itertools.product([-1,1],repeat=3):
            r=np.eye(3)[:,perm]*sign
            if np.linalg.det(aa@r@bb.T)<0:continue
            total=r.T;moved=b@total
            for _ in range(8):
                _,idx=ta.query(moved);u,_,vt=np.linalg.svd(moved.T@a[idx]);step=u@vt
                if np.linalg.det(step)<0:u[:,-1]*=-1;step=u@vt
                total=total@step;moved=moved@step
            cost=float(np.square(ta.query(moved)[0]).mean()+np.square(cKDTree(moved).query(a)[0]).mean())
            if cost<best:best=cost;transform=total.copy()
    return (vertices-bc)@bb@transform@aa.T+ac


def neighbor_figure(queries):
    rows=pd.read_csv(OUT/'retrieval.csv');frame=pd.read_csv(OUT/'manifest.csv').set_index('cell_id')
    methods=[None,'spatial_robust','spherical_harmonics_adjusted','point_mae_adapted_pose_average_raw','point_mae_adapted_pose_average_adjusted'];titles=['Query cell','Spatial PCA neighbor','Spherical harmonics neighbor','Point-MAE unadjusted · secondary','Point-MAE adjusted · primary']
    fig=plt.figure(figsize=(15,9))
    for i,cid in enumerate(queries):
        vq,fq=mesh_for(frame.loc[cid]);reference=sample_surface(vq,fq,384,np.random.default_rng(20261030+int(frame.loc[cid].original_index)))
        for j,method in enumerate(methods):
            chosen=cid if method is None else rows[(rows.cell_id==cid)&(rows.method==method)].iloc[0].neighbor
            v,f=mesh_for(frame.loc[chosen])
            if method is not None:
                p=sample_surface(v,f,384,np.random.default_rng(20261030+int(frame.loc[chosen].original_index)));v=align_neighbor(reference,p,v)
            ax=fig.add_subplot(3,5,i*5+j+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f],facecolors='#247e91' if method is None else '#e98b37',linewidths=0,antialiased=False,shade=True,lightsource=LightSource(315,45),rasterized=True));ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off();ax.set_title(titles[j]+'\n'+chosen,fontsize=9)
    fig.suptitle('Same matched candidate pools · fixed query examples\nOriginal source meshes; neighbors aligned to query by proper rotation and translation only',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'shape_neighbors.png',dpi=170);plt.close(fig)


def main():
    results=json.loads((OUT/'retrieval.json').read_text());table={r['id']:r for r in results['methods']};primary=results['primary'];checks=json.loads((OUT/'stability.json').read_text());patches=pd.read_csv(OUT/'patch_reconstruction.csv');history=pd.read_csv(OUT/'history.csv');protocol=json.loads((OUT/'protocol.json').read_text());complete=json.loads((OUT/'evaluation.completed.json').read_text());best=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    raw=table['morph_all']['geometry_cost'];adjusted=table['morph_adjusted']['geometry_cost'];pca=table['spatial_robust']['geometry_cost'];harmonic=table['spherical_harmonics_adjusted']['geometry_cost'];ci=primary['vs_morph_adjusted']
    names={'point_mae_pretrained_native_adjusted':'Point-MAE pretrained · native · adjusted','point_mae_pretrained_pose_average_raw':'Point-MAE pretrained · pose average · unadjusted','point_mae_adapted_pose_average_raw':'Point-MAE adapted · pose average · unadjusted','point_mae_pretrained_pose_average_adjusted':'Point-MAE pretrained · pose average · adjusted','point_mae_adapted_native_adjusted':'Point-MAE adapted · native · adjusted','point_mae_adapted_pose_average_adjusted':'Point-MAE adapted · pose average · adjusted'}
    def name(r):return names.get(r['id'],r['name'])
    lines=['# Standalone Point-MAE benchmark','',
        'Organoid Atlas was not changed. This evaluates the official ShapeNet-pretrained Point-MAE and a 30-epoch full-network self-supervised adaptation on organoid cell surfaces. It is not a reproduction of the 300-epoch ShapeNet pretraining study.','',
        f"The prespecified primary readout (adapted, proper PCA-sign averaged, standardized and nuisance-adjusted) has geometric neighbor cost **{primary['geometry_cost']:.6f}**: {1-primary['geometry_cost']/raw:+.1%} improvement versus raw nine-feature morphometry, and {1-primary['geometry_cost']/adjusted:+.1%} versus identically adjusted morphometry. The paired difference versus adjusted morphometry is {ci['mean']:+.6f}, descriptive 95% group-bootstrap interval [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}].",'',
        f"Existing spatial PCA has observed cost {pca:.6f}; adjusted spherical harmonics {harmonic:.6f}; adapted Allen {table['allen_head_decoder_adjusted']['geometry_cost']:.6f}. Lower cost means geometrically closer retrieved neighbors. These are geometric proxy results, not cell-type or phenotype classification accuracies.",'',
        f"The prespecified secondary **unadjusted** adapted pose-averaged readout is more promising for general similarity: cost {table['point_mae_adapted_pose_average_raw']['geometry_cost']:.6f}, or {1-table['point_mae_adapted_pose_average_raw']['geometry_cost']/raw:+.1%} versus traditional features. Its advantage drops after nuisance adjustment. This suggests useful geometry similarity but does not establish information additional to volume, elongation and sphericity; regression can also remove useful signal. Raw/unadjusted here means no descriptor scaling/regression, not raw microscopy input.",'',
        '## Scope and decision','',
        ('The primary Point-MAE readout has the lowest observed cost of these existing reference methods. Independent biological validation is still needed before claiming useful phenotype separation.' if primary['geometry_cost']<min(pca,harmonic) else 'The primary Point-MAE readout does not surpass the best existing reference method on observed neighbor cost. These results do not justify replacing the current retrieval defaults.'),'',
        'Checkpoint selection and this comparison reuse validation data. Experimental condition/phenotype labels are absent. No test cells were processed. Descriptor averaging/standardization/nuisance removal are explicit readout choices, not inherent properties of Point-MAE.','',
        '## Matched shape neighbors','',
        '46 eligible queries from the same 48 validation cells / 11 similarity groups as the Allen, spherical-harmonic and CAJAL comparisons. Candidate cells come from another similarity group and lie within one training SD in the three-control space (log volume, elongation, sphericity), with at least five and at most twenty candidates. Controls are approximately matched, not identical.','',
        '| Method | Geometry cost ↓ | Improvement vs raw nine features | Paired improvement vs adjusted nine, 95% CI |','|---|---:|---:|---|']
    selected=['morph_all','morph_adjusted','spatial_robust','spatial_adjusted','spherical_harmonics_adjusted','allen_head_decoder_adjusted','cajal_gw_validation','pointnet_shape128']+[r['id'] for r in results['methods'] if r['id'].startswith('point_mae_')]
    for key in selected:
        r=table[key];c=r['vs_morph_adjusted'];lines.append(f"| {name(r)} | {r['geometry_cost']:.6f} | {1-r['geometry_cost']/raw:+.1%} | {c['mean']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
    lines+=['','Intervals use 2,000 paired group-bootstrap draws; positive differences favor the row method. They are exploratory, unadjusted for multiple readouts. Baseline inputs and costs are frozen from earlier studies. Models have unequal external pretraining, training data and compute budgets.','',
        '## Sampling and rotation stability','',
        '| Model / readout | Fresh sampling identity / 48 | Neighbor Jaccard | Rotated identity / 12 | Relative rotation feature error | Training effective rank |','|---|---:|---:|---:|---:|---:|']
    for arm in ['pretrained','adapted']:
        for kind in ['native','pose_average']:
            r=checks[arm][kind];lines.append(f"| {arm} / {kind} | {r['resampling_self_top1']:.1%} | {r['neighbor_jaccard']:.3f} | {r['rotation_self_top1']:.1%} | {r['rotation_relative_code_error']:.3g} | {r['effective_rank']:.2f} |")
    gaps=np.load(OUT/'frames.npz')['gap']
    lines+=['',
        f"The native network has no built-in SO(3) invariance. The pose readout uses area-weighted surface covariance eigenvectors and averages features over all four proper sign frames; no reflections are used. After rotation, covariance eigensystems are recomputed. Nearly degenerate eigenvalues can destabilize the frame: {int((gaps<.06).sum())} of 609 source cells have a relative minimum eigengap below 0.06. This threshold is descriptive and triggers no fallback or result selection.",'',
        'Fresh sampling draws a new 1,024 points from each fixed 32,768-point area-sampled surface pool and recomputes sample-mean/radius normalization. Rotation tests use the same twelve cells as the Allen benchmark, against the 48-cell gallery. Stability uses raw descriptor distances and does not test remeshing or segmentation errors.','',
        '## Masked-patch reconstruction (a separate task)','',
        '**These scores are not comparable to Allen/PCA whole-shape surface F1.** Point-MAE predicts local missing patches while receiving their true center coordinates and the visible patches. The visual figure retains the visible input. It does not decode a whole cell solely from the 768-dimensional pooled embedding.','',
        '| Model | Masked-patch squared Chamfer ↓ | Center-only baseline ↓ |','|---|---:|---:|']
    for arm in ['pretrained','adapted']:
        sub=patches[patches.arm==arm];lines.append(f"| {arm} | {sub.patch_cd.mean():.6f} | {sub.center_only_cd.mean():.6f} |")
    lines+=['','The center-only reference repeats the known patch center 32 times. Patch overlap and known missing-patch centers make this an inpainting diagnostic, not evidence that a pooled global embedding preserves all geometry. Fixed visual queries were chosen in the earlier spatial study, independent of Point-MAE outcomes.','',
        '## What ran','',
        '- Strict loading of the official 29,006,432-parameter pretrained checkpoint. Original encoder/decoder class bodies execute from pinned author source. Portable Euclidean FPS/kNN and squared-Chamfer implementations replace CUDA-only extensions; geometric definitions, gradients, original forward consistency and checkpoint loading have dedicated checks. No CUDA numerical-parity claim.',
        '- Full published 1,024-point / 64-patch / 32-point-per-patch / 384-channel / 12-layer encoder and 4-layer decoder configuration, with 60% random masking. Neural layers use the Mac GPU; FPS indices are computed on CPU.',
        '- All 469 eligible training cells from 21 groups, 140 validation cells from 11 disjoint groups. Thirty full-network adaptation epochs, batch 8, AdamW 0.0001, weight decay 0.05, cosine schedule, seed 42; fresh point draws and proper SO(3) augmentation, no anisotropic shape distortion.',
        f"- Selected checkpoint: epoch {best['epoch']} by fixed-mask validation patch Chamfer, including epoch 0 and every fifth epoch. Neither nearest-neighbor geometry nor rotation scores select weights.",
        '- Custom embedding: concatenated mean and maximum of all unmasked final tokens (768 dimensions). No untrained classification token or morphology values are added. Pretrained and adapted embeddings exist for all 609 training/validation cells, but neighbor/stability/patch diagnostics use the fixed validation cohort.',
        '- Original 96 training cells fit descriptor scaling and degree-two Ridge nuisance regression for log volume, elongation and sphericity. No descriptor scaling or nuisance fitting on validation cells.',
        f"- Atlas export preserved: {complete['atlas_unchanged']}. No native-app source, Atlas dataset or existing baseline models were changed.",'',
        '[Official Point-MAE source and checkpoint](https://github.com/Pang-Yatian/Point-MAE). [Experiment protocol](protocol.json). [Implementation notes](../../docs/POINT_MAE_BENCHMARK.md).','']
    (OUT/'report.md').write_text('\n'.join(lines))
    chosen=['morph_all','spatial_robust','spherical_harmonics_adjusted','allen_head_decoder_adjusted','cajal_gw_validation','point_mae_pretrained_native_adjusted','point_mae_pretrained_pose_average_raw','point_mae_pretrained_pose_average_adjusted','point_mae_adapted_pose_average_raw','point_mae_adapted_pose_average_adjusted']
    values=[table[k] for k in chosen];mean=np.array([r['vs_morph_adjusted']['mean'] for r in values]);interval=np.array([r['vs_morph_adjusted']['ci95'] for r in values]);ys=np.arange(len(values))
    fig,ax=plt.subplots(figsize=(12,10));ax.barh(ys,mean,color=['#009b97' if k.startswith('point_mae') else '#95a5b8' for k in chosen]);ax.errorbar(mean,ys,xerr=np.maximum(np.stack([mean-interval[:,0],interval[:,1]-mean]),0),fmt='none',ecolor='#23394c',capsize=3);ax.set_yticks(ys,[name(r).replace(' · ','\n') for r in values]);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.8);ax.set_xlabel('Reduction in geometric neighbor cost vs adjusted nine-feature baseline →');ax.set_title('Point-MAE standalone comparison · exploratory validation',loc='left',weight='bold');ax.spines[['right','top']].set_visible(False);fig.text(.02,.01,'46 queries / 11 groups. Descriptive 95% intervals. Only adjusted readouts are nuisance-regressed. No biological labels.',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'retrieval.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(history.epoch,history.validation_patch_cd,'o-',label='Validation masked-patch CD');ax.plot(history.epoch,history.training_patch_cd,'o-',label='Training masked-patch CD');ax.axvline(best['epoch'],color='#888',ls='--',label='Selected epoch');ax.set(xlabel='Local adaptation epoch',ylabel='Squared Chamfer',title='Full-network adaptation · checkpoint objective');ax.legend();fig.tight_layout();fig.savefig(OUT/'adaptation.png',dpi=170);plt.close(fig)
    a=np.load(OUT/'pretrained_patch_visuals.npz');b=np.load(OUT/'adapted_patch_visuals.npz');queries=[r['query'] for r in json.loads((ROOT/'results/spatial_benchmark/development/visual_queries.json').read_text())]
    fig=plt.figure(figsize=(10,9))
    for i,cid in enumerate(queries):
        for j,(data,suffix,title) in enumerate([(a,'_target','Original missing patches'),(a,'_predicted','Pretrained completion'),(b,'_predicted','Adapted completion')]):
            ax=fig.add_subplot(3,3,i*3+j+1,projection='3d');visible=data[cid+'_visible'];p=data[cid+suffix];ax.scatter(*visible.T,s=2,color='#247e91',alpha=.7);ax.scatter(*p.T,s=2,color='#e98b37',alpha=.8);ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off();ax.set_title(title+('\n'+cid if j==0 else ''),fontsize=10)
    fig.suptitle('Masked-patch inpainting with known patch centers\nBlue: visible input retained · Orange: missing target or predicted patches',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'patch_completion.png',dpi=170);plt.close(fig)
    neighbor_figure(queries)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').glob('*point_mae_reference.py'):shutil.copy2(p,snapshot/p.name)
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=complete['atlas_unchanged'],test_used=False,embedding_cells=609,evaluated_cells=48,report_sha256=sha256(OUT/'report.md'),protocol_sha256=sha256(OUT/'protocol.json'),primary=primary,script_hashes={p.name:sha256(p) for p in snapshot.glob('*.py')}),indent=2));print('Point-MAE report complete',flush=True)
    (OUT/'status.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=complete['atlas_unchanged'])))


if __name__=='__main__':main()
