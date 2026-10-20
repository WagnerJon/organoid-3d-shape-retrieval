"""A candid report for the unofficial, Lite-based 3D EDM-VAE experiment."""
import json,shutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapeembed3d_core import OUT,ROOT,SOURCE
from pointnet_shape_core import sha256,sample_surface
from spatial_benchmark_core import mesh_for
from evaluate_shapeembed3d import align_cloud


def label(key):
    return {'shapeembed3d_invariant_raw':'Lite3D invariant · unadjusted','shapeembed3d_invariant_standardized':'Lite3D invariant · standardized','shapeembed3d_invariant_adjusted':'Lite3D invariant · adjusted (primary)','shapeembed3d_cnn_raw':'Lite3D CNN reference · unadjusted','shapeembed3d_cnn_standardized':'Lite3D CNN reference · standardized','shapeembed3d_cnn_adjusted':'Lite3D CNN reference · adjusted','shapeembed3d_direct_edm_gw':'Direct EDM hard matching · no encoder'}.get(key,key)


def neighbor_figure(queries):
    frame=pd.read_csv(OUT/'manifest.csv').set_index('cell_id');rows=pd.read_csv(OUT/'retrieval.csv')
    methods=[None,'spatial_robust','shapeembed3d_invariant_raw','shapeembed3d_invariant_adjusted','shapeembed3d_direct_edm_gw'];titles=['Query cell','Spatial PCA neighbor','Lite3D unadjusted neighbor','Lite3D adjusted · primary','Direct EDM matching neighbor']
    fig=plt.figure(figsize=(15,9))
    for i,cid in enumerate(queries):
        vq,fq=mesh_for(frame.loc[cid]);reference=sample_surface(vq,fq,384,np.random.default_rng(20261030+int(frame.loc[cid].original_index)));shapes=[]
        for method in methods:
            chosen=cid if method is None else rows[(rows.cell_id==cid)&(rows.method==method)].iloc[0].neighbor;v,f=mesh_for(frame.loc[chosen])
            if method is not None:
                p=sample_surface(v,f,384,np.random.default_rng(20261030+int(frame.loc[chosen].original_index)));aligned,_=align_cloud(reference,p,False);u,_,vt=np.linalg.svd((p-p.mean(0)).T@(aligned-aligned.mean(0)));rotation=u@vt;v=(v-p.mean(0))@rotation+aligned.mean(0)
            shapes.append((chosen,v,f))
        extent=max(1.2,max(float(np.abs(v).max()) for _,v,f in shapes)*1.05)
        for j,(chosen,v,f) in enumerate(shapes):
            ax=fig.add_subplot(3,5,i*5+j+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f],facecolors='#247e91' if j==0 else '#e98b37',linewidths=0,antialiased=False,shade=True,lightsource=LightSource(315,45),rasterized=True));ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off();ax.set_title(titles[j]+'\n'+chosen,fontsize=9)
    fig.suptitle('Fixed query examples · same matched candidate pools\nOriginal source meshes; neighbors aligned by proper rotation and translation only',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'shape_neighbors.png',dpi=170);plt.close(fig)


def main():
    data=json.loads((OUT/'retrieval.json').read_text());table={r['id']:r for r in data['methods']};primary=data['primary'];baseline=table['morph_all']['geometry_cost'];adjusted=table['morph_adjusted']['geometry_cost'];checks=json.loads((OUT/'stability.json').read_text());recon=pd.read_csv(OUT/'reconstruction.csv');calibration=json.loads((OUT/'matching_audit.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text());complete=json.loads((OUT/'evaluation.completed.json').read_text());checkpoints={k:torch.load(OUT/k/'best.pt',map_location='cpu',weights_only=False) for k in ['invariant','cnn']};ci=primary['vs_morph_adjusted']
    lines=['# ShapeEmbedLite3D — experimental adaptation','',
        '**Unofficial research prototype. Atlas was not changed.** This adapts ShapeEmbedLite primarily by preserving normalized Euclidean distance matrices (EDMs), a 128D variational bottleneck, the original Lite coordinate-decoder class, and distance-only reconstruction supervision. It is not a published or author-endorsed 3D version.','',
        f"The declared primary invariant/adjusted descriptor has neighbor geometry cost **{primary['geometry_cost']:.6f}**, {1-primary['geometry_cost']/baseline:+.1%} versus ordinary nine-feature morphometry and {1-primary['geometry_cost']/adjusted:+.1%} versus equally adjusted morphometry. The paired improvement over adjusted morphometry is {ci['mean']:+.6f}, descriptive 95% group-bootstrap interval [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}].",'',
        f"The unadjusted invariant descriptor improves observed neighbor cost by {1-table['shapeembed3d_invariant_raw']['geometry_cost']/baseline:+.1%} over ordinary traditional features. This trails spatial PCA and adjusted spherical harmonics. The prototype is more stable than the literal CNN reference, but does not justify replacing the current retrieval defaults. The primary interval versus equally adjusted morphometry is only narrowly positive and is exploratory; biological added value is not established.",'',
        'These scores concern similarity of segmentation-derived shapes, not experimental-condition classification or additional biological information. Validation also selects checkpoints.','',
        '## What stayed close to ShapeEmbedLite','',
        '- Surface coordinates are converted to pairwise Euclidean distances. Frobenius normalization times the number of points removes uniform scale; no PCA orientation alignment is needed.',
        '- A 128-dimensional VAE compresses the normalized EDM. This uses the explicit paper target, without the upstream CLI experimental latent-size doubling. A deterministic latent mean is the descriptor and can drive reconstruction alone.',
        '- The original `DecoderMLP` source class predicts 128 three-dimensional points. Their pair distances produce a valid EDM by construction. The class-default three-hidden-layer recipe is retained, with widths 1,536 in this configuration.',
        '- Learning compares reconstructed and input distances; no Chamfer, coordinate-alignment, raw-intensity or morphology supervision enters training. The KL weight is 1e-10.',
        '- Recorded scale can be restored outside the latent. Mirror/reflection invariance is retained; chirality is intentionally not represented.','',
        '## What genuinely changes in 3D','',
        'A surface is not an ordered cyclic contour. The original cyclic/reversal reindexing does not handle arbitrary surface-point permutations. The main encoder therefore uses scalar EDM message passing and symmetric pooling. A companion reference uses the official unpretrained ResNet18 factory with circular convolutions disabled; its pooling remains as in the source. Both arms share the same decoder and generalized EDM loss.','',
        'Reindexing is generalized to approximate hard one-to-one matching. A soft GW proposal is rounded to feasible marginals and projected with Hungarian assignment; a distance-profile Hungarian candidate is also checked. The lower actual reindexed EDM MSE trains the model. Soft averaging is not the reconstruction objective, and the permutation optimum is not certified. This matching algorithm and the surface-set encoder are the principal extensions.','',
        '```mermaid','flowchart LR','  S[3D surface sample] --> D[Normalized EDM]','  D --> E[Permutation-equivariant EDM encoder]','  E --> Z[128D VAE]','  Z --> P[Lite MLP: 3D points]','  P --> R[Valid decoded EDM]','  D --> L[Reindexing-aware EDM loss]','  R --> L','```','',
        '## Matched geometric neighbors','',
        'Same 48 validation cells / 46 eligible queries / 11 similarity groups as the previous studies. Candidates belong to another group and are approximately matched in log volume, elongation and sphericity. Descriptor scaling and degree-two Ridge nuisance adjustment use the original 96 training cells. No biology labels or validation fitting of these transforms.','',
        '| Method | Geometry cost ↓ | Improvement vs nine traditional features | Paired difference vs adjusted nine, 95% CI |','|---|---:|---:|---|']
    chosen=['morph_all','morph_adjusted','spatial_robust','spatial_adjusted','spherical_harmonics_adjusted','allen_head_decoder_adjusted','point_mae_adapted_pose_average_raw','point_mae_adapted_pose_average_adjusted','cajal_gw_validation','shapeembed3d_direct_edm_gw']+[r['id'] for r in data['methods'] if r['id'].startswith('shapeembed3d_') and r['id']!='shapeembed3d_direct_edm_gw']
    for key in chosen:
        r=table[key];stat=r['vs_morph_adjusted'];name=label(key) if key.startswith('shapeembed3d') else r['name'];lines.append(f"| {name} | {r['geometry_cost']:.6f} | {1-r['geometry_cost']/baseline:+.1%} | {stat['mean']:+.6f} [{stat['ci95'][0]:+.6f}, {stat['ci95'][1]:+.6f}] |")
    direct=table['shapeembed3d_direct_edm_gw'];delta=primary['vs_shapeembed3d_direct_edm_gw']
    lines+=['',f"Direct EDM matching without a learned bottleneck has cost {direct['geometry_cost']:.6f}. Primary learned-descriptor improvement over that input-only baseline is {delta['mean']:+.6f}, interval [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}]. This separates benefits of the invariant input representation from benefits of learning.",'',
        'All intervals are descriptive paired group-bootstrap estimates (2,000 draws), uncorrected for examining multiple readouts. Models differ in data, external pretraining and compute budgets. Positive paired differences favor the row method; percentages are geometric-cost reductions, not classification accuracies.','',
        '## Invariance versus sampling stability','',
        '| Encoder | Resampling identity / 48 | Neighbor Jaccard | Permutation identity / 48 | Rotation / 12 | Reflection / 12 | Translation + scale / 12 | Training effective rank |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ['invariant','cnn']:
        r=checks[kind];lines.append(f"| {kind} | {r['resampling_self_top1']:.1%} | {r['neighbor_jaccard']:.3f} | {r['permutation_self_top1']:.1%} | {r['rotation_self_top1']:.1%} | {r['reflection_self_top1']:.1%} | {r['translate_scale_self_top1']:.1%} | {r['effective_rank']:.2f} |")
    lines+=['','The invariant architecture is designed for fixed-sample translation, positive uniform scale, O(3) transforms and arbitrary simultaneous row/column permutations. This does not guarantee a complete invariant for every graph, or stable descriptors under new surface sampling, remeshing or segmentation changes. The CNN reference has no arbitrary-permutation guarantee. Mean latent codes, not stochastic samples, are compared.','',
        '## Global 3D reconstruction','',
        '**These are sparse 128-point outputs, not reconstructed meshes.** Known normalization scale is restored and clouds are aligned for evaluation without fitting scale. Both proper rotations and O(3), including reflection, are reported. Reflection is allowed by the EDM representation. The 128-point source reference exposes the sampling-resolution limit and should be considered alongside dense-surface errors.','',
        '| Point cloud / alignment | Cells | Squared Chamfer ↓ | Surface F1 at 5% radius ↑ | P95 distance (inferred µm) ↓ | Reflected alignments |','|---|---:|---:|---:|---:|---:|']
    for key in ['source_sampling_128','invariant_proper','invariant_o3','cnn_proper','cnn_o3']:
        sub=recon[recon.method==key];lines.append(f"| {key} | {len(sub)} | {sub.squared_chamfer.mean():.6f} | {sub.f1_at_005_radius.mean():.3f} | {sub.p95_surface_error_um.mean():.3f} | {int(sub.alignment_reflected.sum())} |")
    source_error=recon[recon.method=='source_sampling_128'].squared_chamfer.mean();model_error=recon[recon.method=='invariant_o3'].squared_chamfer.mean()
    lines+=['',f"The invariant decoder has {model_error/source_error:.2f} times the source 128-point reference Chamfer error. Thus low sampling density alone does not explain the weak reconstruction: the trained decoder also underfits geometry. The fixed examples retain broad round/elongated structure but lose characteristic details. Valid EDMs guarantee geometric consistency, not fidelity to a particular cell.",'',
        'Comparison with dense SDF meshes is not a matched-resolution reconstruction contest. Orientation alignment uses a finite multi-start ICP heuristic, not a certified global optimum. No watertightness, connectivity or topology claim is made for a point set.','',
        '## Matching calibration','',
        '| Arm | Identical-source matched EDM MSE | Independent-sample matched EDM MSE | Maximum relative marginal error |','|---|---:|---:|---:|']
    for kind,r in calibration.items():lines.append(f"| {kind} | {r['self_gw_mean']:.6f} | {r['resampling_gw_mean']:.6f} | {r['max_relative_marginal_error']:.3g} |")
    lines+=['','An approximate assignment can miss a correspondence even for identical geometry. These references are diagnostics, not universal lower bounds or a claim of globally optimal permutation matching. Reconstruction uses only EDMs during training; ICP is used only for rendering/evaluation.','',
        '## Training and development record','',
        '- All 469 eligible training cells / 21 groups and 140 validation cells / 11 disjoint groups. Four independent area-pool/FPS views, 128 points each. Source geometry hashes checked. No test or excluded cells.',
        '- From scratch: 100 epochs per arm, Adam 0.001, batch 16, seed 42; random surface-view selection and arbitrary row/column permutations. Neither raw image data nor conventional features are supplied to the model.',
        f"- Selected validation-reconstruction epochs: invariant {checkpoints['invariant']['epoch']}, CNN {checkpoints['cnn']['epoch']}. Checkpoints include epoch 0 and every fifth epoch; retrieval never selects weights.",
        '- The invariant encoder uses distance RBF features and channel LayerNorm. Both arms initialize log-variance bias to -6. These are explicit adaptation/conditioning choices, not properties claimed for the published method.',
        '- Early probes exposed unbalanced transport weights and a near-constant latent mean under default sampling noise. Their weights were isolated and discarded before final retrieval evaluation. A soft-GW draft was also discarded when the final objective was tightened to actual hard-reindexed EDM MSE. The protocol records the training-sample diagnostic and corrections; this is exploratory model development, not an untouched confirmatory test.',
        '- The stated single-reconstruction-term VAE loss is used; the repository training helper adds that term twice. No duplicated term is used here. Decoder/encoder source and adaptation hashes are preserved.',
        f"- Atlas preserved: {complete['atlas_unchanged']}; embeddings for {complete['embedding_cells']} training/validation cells, evaluation on {complete['evaluated_cells']}. Biological usefulness remains untested.",'',
        '[Method, fidelity table and reproduction](../../docs/SHAPEEMBED_LITE_3D.md). [Original ShapeEmbedLite source](https://github.com/uhlmanngroup/ShapeEmbedLite). [Original ShapeEmbed paper](https://arxiv.org/abs/2507.01009).','']
    (OUT/'report.md').write_text('\n'.join(lines))
    chart=['morph_all','spatial_robust','spherical_harmonics_adjusted','point_mae_adapted_pose_average_raw','shapeembed3d_direct_edm_gw','shapeembed3d_cnn_raw','shapeembed3d_cnn_adjusted','shapeembed3d_invariant_raw','shapeembed3d_invariant_adjusted'];values=[table[k] for k in chart];ys=np.arange(len(chart));mean=np.array([r['vs_morph_adjusted']['mean'] for r in values]);interval=np.array([r['vs_morph_adjusted']['ci95'] for r in values]);names=[label(k) if k.startswith('shapeembed3d') else table[k]['name'] for k in chart]
    fig,ax=plt.subplots(figsize=(12,9));ax.barh(ys,mean,color=['#009b97' if k.startswith('shapeembed3d') else '#95a5b8' for k in chart]);ax.errorbar(mean,ys,xerr=np.maximum(np.stack([mean-interval[:,0],interval[:,1]-mean]),0),fmt='none',ecolor='#23394c',capsize=3);ax.set_yticks(ys,[n.replace(' · ','\n') for n in names]);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.8);ax.set_xlabel('Reduction in neighbor geometry cost vs adjusted nine-feature baseline →');ax.set_title('ShapeEmbedLite3D · exploratory matched-validation comparison',loc='left',weight='bold');ax.spines[['right','top']].set_visible(False);fig.text(.02,.01,'46 queries / 11 groups. Descriptive 95% intervals. Only adjusted descriptors are nuisance-regressed. No biological labels.',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'retrieval.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for kind in ['invariant','cnn']:
        h=pd.read_csv(OUT/kind/'history.csv');ax.plot(h.epoch,h.validation_gw,'o-',ms=3,label=kind+' validation mean-code reconstruction')
    ax.set(xlabel='Training epoch',ylabel='Hard-reindexed EDM MSE',yscale='log',title='Checkpoint criterion · no neighbor-based selection');ax.legend();fig.tight_layout();fig.savefig(OUT/'training.png',dpi=170);plt.close(fig)
    a=np.load(OUT/'invariant_reconstructions.npz');b=np.load(OUT/'cnn_reconstructions.npz');queries=[r['query'] for r in json.loads((ROOT/'results/spatial_benchmark/development/visual_queries.json').read_text())];fig=plt.figure(figsize=(12,9))
    for i,cid in enumerate(queries):
        extent=max(1.2,max(float(np.abs(p).max()) for p in [a[cid+'_reference'],a[cid+'_source_sampling_128'],a[cid+'_invariant_o3'],b[cid+'_cnn_o3']])*1.05)
        for j,(source,key,title) in enumerate([(a,'_reference','Dense source surface sample'),(a,'_source_sampling_128','Source 128-point sample'),(a,'_invariant_o3','Invariant Lite3D · 128 points'),(b,'_cnn_o3','CNN reference · 128 points')]):
            p=source[cid+key];ax=fig.add_subplot(3,4,i*4+j+1,projection='3d');ax.scatter(*p.T,s=2 if j==0 else 8,color='#247e91' if j<2 else '#e98b37',alpha=.8);ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off();ax.set_title(title+('\n'+cid if j==0 else ''),fontsize=9)
    fig.suptitle('Global latent-only reconstruction · recorded scale restored\nDecoded points aligned allowing reflections; sparse points are not surface meshes',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'reconstructions.png',dpi=170);plt.close(fig)
    neighbor_figure(queries)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').glob('*shapeembed3d*.py'):shutil.copy2(p,snapshot/p.name)
    shutil.copy2(SOURCE,snapshot/'official_lite_models.py');shutil.copy2(SOURCE.parent.parent/'LICENSE',snapshot/'OFFICIAL_LICENSE')
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=complete['atlas_unchanged'],test_used=False,embedding_cells=609,evaluated_cells=48,primary=primary,report_sha256=sha256(OUT/'report.md'),protocol_sha256=sha256(OUT/'protocol.json'),script_hashes={p.name:sha256(p) for p in snapshot.glob('*.py')}),indent=2));(OUT/'status.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=complete['atlas_unchanged'])));print('ShapeEmbedLite3D report complete',flush=True)


if __name__=='__main__':main()
