"""Create a standalone benchmark report, scientific figures and completion record."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pointnet_shape_core import ROOT,sha256
from allen_reference_core import OUT,source_hashes
from spatial_benchmark_core import mesh_for


def main():
    evaldir=OUT/'evaluation';geometry=json.loads((evaldir/'geometry.json').read_text());retrieval=json.loads((evaldir/'retrieval.json').read_text());stability=json.loads((evaldir/'stability.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text());completion=json.loads((evaldir/'completed.json').read_text());runtime=json.loads((OUT/'runtime_audit.json').read_text())
    selection={name:json.loads((OUT/name/'completed.json').read_text()) for name in ['decoder_only','head_decoder']}
    checkpoints={name:dict(epoch=int(__import__('torch').load(OUT/name/'best.pt',map_location='cpu',weights_only=False)['epoch']),sha256=sha256(OUT/name/'best.pt')) for name in selection}
    table={r['id']:r for r in retrieval['methods']};baseline=table['morph_all']['geometry_cost'];adjusted=table['morph_adjusted']['geometry_cost']
    primary=table['allen_head_decoder_adjusted'];ci=primary['vs_morph_adjusted'];gain=1-primary['geometry_cost']/adjusted
    lines=['# Allen Institute reference benchmark','',
        'Standalone experiment. Organoid Atlas was not updated. This evaluates the released pretrained NPM1 model and two explicitly limited local adaptation arms; it is not a full reproduction of the paper’s end-to-end training.','',
        'The tested adaptation modestly improves reconstructed geometry, but does not lead the shape-neighbor benchmark. Spatial PCA and adjusted spherical harmonics retain lower observed neighbor costs. I would keep those as the current retrieval methods and retain Allen as an offline generative reference; these results do not establish additional biological information.','',
        f"Primary comparison: adapted standardized/adjusted invariant norms versus identically adjusted nine-feature morphometry. Mean group geometry-cost reduction: {gain:+.1%}; paired absolute difference {ci['mean']:+.6f}, 95% group-bootstrap interval [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}].",'',
        'This is exploratory validation evidence. The same validation data select checkpoints; source geometry is from predicted masks, and experimental condition/phenotype labels are unavailable.','',
        '## Reconstruction','',
        'Meshes are extracted at 56³, transformed back to the previous common normalization and compared to fresh source-surface samples. F1 tolerance is 5% of the common radius. Means below cover successful reconstructions; failures and open/box-crossing surfaces must be considered alongside those means.','',
        '| Method | Surface F1 ↑ | Squared Chamfer ↓ | Volume error | Failed / 48 | Open | Box crossing |','|---|---:|---:|---:|---:|---:|---:|']
    names={'pretrained':'Released pretrained','decoder_only':'Decoder-only adaptation','head_decoder':'Encoder-head + decoder adaptation','spatial_pca':'Existing aligned spatial PCA ·95D'}
    for key in ['pretrained','decoder_only','head_decoder','spatial_pca']:
        r=geometry[key];lines.append(f"| {names[key]} | {r['f1_at_005_radius']:.3f} | {r['squared_chamfer']:.6f} | {r['volume_relative_error']:.1%} | {r['failed_surfaces']} | {r['nonwatertight']} | {r['boundary_crossings']} |")
    lines+=['','## Matched geometry retrieval','',
        '46 eligible queries from the fixed 48-cell validation cohort, 11 similarity groups; identical other-group candidate pools, matched within one training SD in log volume/sphericity/elongation. Matching is not exact size matching. Lower proxy cost is better. Intervals are descriptive and not adjusted for exploring multiple readouts.','',
        '| Readout | Geometry cost ↓ | Reduction vs raw 9 features | Paired improvement vs adjusted 9 baseline, 95% CI |','|---|---:|---:|---|']
    for r in retrieval['methods']:
        stat=r['vs_morph_adjusted'];lines.append(f"| {r['name']} | {r['geometry_cost']:.6f} | {1-r['geometry_cost']/baseline:+.1%} | {stat['mean']:+.6f} [{stat['ci95'][0]:+.6f}, {stat['ci95'][1]:+.6f}] |")
    lines+=['','## Stability','',
        '| Arm | Fresh sampling identity / 48 | Neighbor Jaccard | Fresh rotation identity / 12 | Effective rank on training norms |','|---|---:|---:|---:|---:|']
    for name in ['pretrained','decoder_only','head_decoder']:
        r=stability[name];lines.append(f"| {names[name]} | {r['resampling_self_top1']:.1%} | {r['resampling_neighbor_jaccard']:.3f} | {r['fresh_rotation_self_top1']:.1%} | {r['effective_rank']:.2f} |")
    lines+=['',
        'Rotation tests rerun the full 8,192-point backbone on twelve prespecified rotated input clouds, using fixed normalization transforms, against the 48-cell gallery. They do not recompute axis-aligned bounding boxes after rotation. Resampling uses a fresh draw from each original 32,768-point surface pool, not remeshing or segmentation perturbation.','',
        '## What was actually run','',
        '- Strict loading of the published 512-vector encoder and residual SDF decoder: 14,306,163 parameters. Original model classes are executed from the pinned source; disabled optional imports are adapted without replacing model mathematics.',
        '- Full 8,192-point input, k = 20, FP32. Memory-bounded pointwise and exact-kNN scheduling passed CPU float64 output/gradient/equivariance tests. The standard unchunked Metal run hit an invalidating memory error. Lower-precision probes were rejected.',
        f"- Full-size CPU/Metal cache agreement check: relative feature error {runtime['cpu_feature_relative_error']:.3g}; cached final-head maximum error {runtime['cached_head_maximum_error']:.3g}.",
        '- All 469 eligible training cells, 21 training groups; all 140 validation cells from 11 disjoint groups select checkpoints. No test cells are prepared or evaluated. Published NPM1 pretraining provides additional external training and is not budget-matched to our existing PCA baseline.',
        '- Local adaptation freezes the backbone. Decoder-only and final-encoder-layer-plus-decoder arms each receive 30 epochs, batch 8, 2,048 queried SDF observations per training cell-step, Adam 0.0002. Their checkpoint selection includes epoch 0 and every fifth epoch on fixed 20,000-query validation observations.',
        f"- Selected epochs: decoder-only {checkpoints['decoder_only']['epoch']}; head+decoder {checkpoints['head_decoder']['epoch']}. A selected epoch 0 means adaptation did not improve the chosen validation objective.",
        '- Preprocessing follows the bounding-box center/largest-side normalization, with exact mesh SDF values replacing scanned approximations and float32 replacing float16 storage. Training has one fixed noised surface draw per cell; proper rotations of cached features preserve equivariance but do not provide fresh surface resampling each epoch.',
        '- Descriptor standardization and nuisance fits use the original 96 training cells for comparison with existing baselines. The backbone/decoder training populations and pretraining budgets differ; this is not a controlled architecture ablation.',
        '- Primary retrieval uses 512 scalar vector norms; SDF reconstruction uses 512×3 vector codes. Good reconstruction does not by itself show that the norm readout retains the reconstructed geometry. A prespecified secondary diagnostic aligns full vector codes by proper SO(3) before comparison; it is not the published norm readout and is not nuisance-regressed.',
        f"- Atlas analysis export unchanged: {completion['atlas_unchanged']}. No per-cell latent optimization, hand-designed morphology input, or microscopy intensity input was supplied to this model.",'',
        'The full original training recipe specifies 400–2000 epochs on A100 GPUs. The local head/decoder study cannot establish how that complete training run would perform. A positive geometry result would still need independent biological labels; a negative result would not reject the published framework across its intended intracellular-structure applications.','',
        '[Protocol and implementation details](../../docs/ALLEN_REFERENCE_BENCHMARK.md). [Official paper](https://www.nature.com/articles/s41592-025-02729-9). [Author benchmark source](https://github.com/AllenCell/benchmarking_representations).','']
    if (OUT/'symmetry_diagnostic.json').exists() and (OUT/'symmetry_float64.json').exists():
        a=json.loads((OUT/'symmetry_diagnostic.json').read_text());b=json.loads((OUT/'symmetry_float64.json').read_text())
        lines += ['## Engineered symmetry diagnostic','',
            f"For two exactly D2-symmetric ellipsoid point sets, float32 vector-code norms were {a['latent_norms']}; a float64 check on the first ellipsoid produced norm {b['latent_norm']:.3g}. In exact arithmetic, global equivariant vectors must vanish on these symmetric inputs. Numerical symmetry breaking can therefore influence outputs. This is a structural diagnostic, not a natural-cell failure estimate or a basis for selecting the adaptation checkpoints.",'']
    (OUT/'report.md').write_text('\n'.join(lines))
    # Reconstruction figure uses fixed low/mid/high-sphericity queries chosen in
    # the earlier benchmark, independent of the present model’s performance.
    a=np.load(evaldir/'visuals.npz');methods=['references','pretrained','decoder_only','head_decoder','spatial_pca'];titles=['Original segmented surface','Released pretrained','Decoder adaptation','Head + decoder adaptation','Spatial PCA ·95D']
    manifest=pd.read_csv(OUT/'manifest.csv').set_index('cell_id')
    fig=plt.figure(figsize=(15,9))
    for i,cid in enumerate(a['ids']):
        for j,key in enumerate(methods):
            ax=fig.add_subplot(3,5,i*5+j+1,projection='3d')
            if key=='references':vertices,faces=mesh_for(manifest.loc[cid])
            else:
                with np.load(evaldir/'meshes'/f"{cid.replace(':','_')}_{key}.npz") as mesh:
                    vertices,faces=mesh['vertices'],mesh['faces']
            if len(faces):
                ax.add_collection3d(Poly3DCollection(vertices[faces],facecolors='#247e91' if key=='references' else '#e98b37',
                    linewidths=0,antialiased=False,shade=True,lightsource=LightSource(azdeg=315,altdeg=45),rasterized=True))
            else:ax.text(0,0,0,'No extracted zero surface',fontsize=8)
            ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),zlim=(-1.15,1.15),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off();ax.set_title(titles[j]+('\n'+cid if j==0 else ''),fontsize=9)
    fig.suptitle('Full 8,192-point published backbone · local partial adaptation · fixed validation queries',fontsize=13);fig.tight_layout();fig.savefig(OUT/'reconstructions.png',dpi=170);plt.close(fig)
    chosen=[r for r in retrieval['methods'] if r['id'] in ['morph_all','morph_adjusted','spatial_robust','spatial_adjusted','spherical_harmonics_adjusted','cajal_gw_validation','cajal_profile_adjusted','allen_pretrained_adjusted','allen_head_decoder_adjusted','allen_pretrained_vector_alignment','allen_head_decoder_vector_alignment']]
    fig,ax=plt.subplots(figsize=(12,10));ys=np.arange(len(chosen));means=np.array([r['vs_morph_adjusted']['mean'] for r in chosen]);interval=np.array([r['vs_morph_adjusted']['ci95'] for r in chosen]);labels=[r['name'].replace(' · ','\n').replace('pose-aligned full vector code (secondary)','full vector-code alignment · secondary') for r in chosen]
    ax.barh(ys,means,color=['#009b97' if r['id'].startswith('allen') else '#95a5b8' for r in chosen]);ax.errorbar(means,ys,xerr=np.maximum(np.stack([means-interval[:,0],interval[:,1]-means]),0),fmt='none',ecolor='#22384c',capsize=3)
    ax.set_yticks(ys,labels);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.7);ax.set_xlabel('Reduction in geometric neighbor cost vs adjusted nine-feature baseline →');ax.set_title('Standalone Allen reference benchmark · exploratory validation',loc='left',weight='bold');ax.spines[['right','top']].set_visible(False)
    fig.text(.02,.01,'46 queries / 11 similarity groups. Group-bootstrap 95% intervals. Source-segmentation geometry, not biological labels. Only readouts labelled adjusted are residualized.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'retrieval.png',dpi=170);plt.close(fig)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').glob('*allen*py'):shutil.copy2(p,snapshot/p.name)
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=completion['atlas_unchanged'],test_used=False,
        training_scope='Released frozen-backbone transfer and partial final-head/decoder adaptation; not full end-to-end retraining.',
        source_hashes=source_hashes(),script_hashes={p.name:sha256(p) for p in snapshot.glob('*.py')},report_sha256=sha256(OUT/'report.md'),primary=ci),indent=2))
    print('Standalone report complete',flush=True)


if __name__=='__main__':main()
