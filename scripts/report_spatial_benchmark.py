"""Write benchmark results and predetermined neighbor contact sheets."""
import json
import os
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,sample_surface,sha256
from prepare_spatial_benchmark import OUT
from spatial_benchmark_core import mesh_for,surface_frame


def contact_sheet(stage,kind):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/stage;frame=pd.read_csv(folder/'manifest.csv');records=pd.read_csv(folder/'retrieval.csv')
    eligible=records.loc[records.method==kind,'query'].to_numpy()
    if not len(eligible):return
    order=eligible[np.argsort(frame.iloc[eligible].sphericity.to_numpy())];chosen=order[np.linspace(0,len(order)-1,3).astype(int)]
    methods=[kind,kind+'_residual','nine_measures'];fig=plt.figure(figsize=(14,11));assessment=[]
    for ri,index in enumerate(chosen):
        neighbors=[index]
        for method in methods:neighbors.append(int(records[(records['query']==index)&records.method.eq(method)].iloc[0].neighbor))
        for ci,neighbor in enumerate(neighbors):
            row=frame.iloc[neighbor];v,f=mesh_for(row);cloud=sample_surface(v,f,2048,np.random.default_rng(2030+int(row.original_index)))
            center,axes,_,_=surface_frame(v,f);cloud=(cloud-center)@axes
            signs=np.sign((cloud**3).mean(0));signs[signs==0]=1;signs[2]=signs[0]*signs[1];cloud*=signs
            ax=fig.add_subplot(3,4,ri*4+ci+1,projection='3d');ax.scatter(*cloud.T,s=2,alpha=.8,color=['#247e91','#ec8a35','#8a65ac','#697580'][ci])
            ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),zlim=(-1.15,1.15),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off()
            label=['Query','Encoder neighbor','Residual neighbor','Nine-measure neighbor'][ci]
            ax.set_title(f"{label}\n{row.cell_id}\nV={row.volume_um3:.0f} µm³ · S={row.sphericity:.2f} · E={row.elongation:.2f}",fontsize=9)
        assessment.append(dict(query=frame.iloc[index].cell_id,selection='Low/mid/high sphericity among matched queries',
            encoder_neighbor=frame.iloc[neighbors[1]].cell_id,residual_neighbor=frame.iloc[neighbors[2]].cell_id,classical_neighbor=frame.iloc[neighbors[3]].cell_id))
    fig.suptitle(f'{stage}: original cell surfaces · {kind.upper()} · display orientation aligned independently');fig.tight_layout();fig.savefig(folder/'neighbors.png',dpi=170);plt.close(fig)
    (folder/'visual_queries.json').write_text(json.dumps(assessment,indent=2)+'\n')


def pose_plot(development,test,kind):
    import matplotlib.pyplot as plt
    def values(metrics):
        return [metrics['pose'][kind+'_raw']['rotation_identity'],
            metrics['pose'][kind+'_aligned']['single_frame_rotation_identity'],
            metrics['pose'][kind+'_aligned']['rotation_identity'],
            metrics['pose'][kind+'_aligned']['sampled_frame_identity']]
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(4)
    for shift,label,metrics,color in [(-.18,'Validation',development,'#247e91'),(.18,'Held-out spatial test',test,'#ec8a35')]:
        bars=ax.bar(x+shift,np.array(values(metrics))*100,.36,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set_xticks(x,['Raw orientation','One alignment frame','Four proper frames','Fresh sampled frames'])
    ax.set(ylabel='Same-cell top-1 retrieval (%)',ylim=(0,115),title=kind.upper()+': orientation and frame sensitivity')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(OUT/'pose_summary.png',dpi=170);plt.close(fig)


def report():
    development=json.loads((OUT/'development/metrics.json').read_text());test=json.loads((OUT/'test/metrics.json').read_text());lock=json.loads((OUT/'freeze.json').read_text())
    chosen=lock['selected'];primary=test['retrieval']['methods'][chosen+'_residual']['nine_measures'];pose=test['pose'][chosen+'_aligned']
    retrieval_rows=pd.read_csv(OUT/'test/retrieval.csv')
    costs=retrieval_rows.groupby(['similarity_group','method']).geometry_cost.mean().unstack()
    relative=1-float(costs[chosen+'_residual'].mean()/costs.nine_measures.mean())
    ci=primary['ci95']
    if ci[0]>0:conclusion='The primary held-out geometry comparison supports additional geometric information beyond the nine-measure baseline under the stated adjustment.'
    elif ci[1]<0:conclusion='The primary held-out geometry comparison favors the nine-measure baseline; do not replace classical similarity with these residual features.'
    else:conclusion='The primary held-out geometry comparison is inconclusive: additional geometry beyond the nine-measure baseline is not established.'
    if pose['sampled_frame_identity']<.8:conclusion+=' Sample-estimated principal frames remain fragile; prefer exact mesh frames and do not claim robustness to segmentation changes.'
    residual_correlation=test['retrieval']['residual_diagnostics'][chosen]['max_residual_control_correlation']
    if residual_correlation>.5:conclusion+=' Some residual features still correlate strongly with the controls; this does not demonstrate complete morphometric independence.'
    symmetry=json.loads((OUT/'synthetic_symmetry.json').read_text()) if (OUT/'synthetic_symmetry.json').exists() else None
    adjusted=json.loads((OUT/'adjusted_classical.json').read_text()) if (OUT/'adjusted_classical.json').exists() else None
    if symmetry and symmetry['rotation_identity']<.95:conclusion+=' The synthetic degeneracy check exposes a remaining pose ambiguity: use an ambiguity flag/fallback before generalizing this alignment beyond the present dataset.'
    if adjusted and adjusted['result']['ci95'][0]<=0:conclusion+=' The equally adjusted classical-baseline sensitivity check does not establish a robust advantage; interpret the original primary contrast narrowly.'
    supplement=['## Supplementary diagnostics after the frozen primary test','']
    if symmetry:
        lo=min(r['sphericity'] for r in symmetry['records']);hi=max(r['sphericity'] for r in symmetry['records'])
        supplement+=[f"Eight engineered six-lobed, near-spherical meshes (sphericity {lo:.3f}–{hi:.3f}) have essentially isotropic surface covariance. Across eight rotations, same-shape identity is {symmetry['rotation_identity']:.1%} in this eight-shape gallery. Principal axes are undefined in this limit, and four sign choices do not remove arbitrary orientations. This is an adversarial diagnostic, not an estimated natural-cell failure rate or directly comparable to the 135-cell gallery. The real test cohort contains no cells with sphericity ≥0.9; the validation cohort contains only one. No fitting or model changes followed this check.",'']
    if adjusted:
        value=adjusted['result'];lo,hi=value['ci95']
        supplement+=[f"Nine classical features were also residualized using the same training-only degree-two Ridge procedure. PCA residual features versus equally adjusted classical features: improvement {value['mean']:.6f}, 95% group interval [{lo:.6f}, {hi:.6f}], {adjusted['queries']} queries; descriptive relative cost reduction {adjusted['relative_lower_cost']:.1%}. This sensitivity check was added after seeing size differences in contact sheets. It is not a new confirmatory test and did not select or tune a model.",'']
    audit=json.loads((OUT/'test/source_mesh_audit.json').read_text())
    supplement+=[f"All {audit['cells']} source meshes were independently audited: {audit['open_meshes']} open meshes, {audit['inconsistent_winding']} inconsistent winding cases, {audit['outside_canonical_box']} outside the canonical input box. Frozen source and model hashes remained unchanged.",'',
        'Visual review: the predetermined sheets contain plausible profile/indentation matches and some ambiguous alternatives. The size-normalized encoder can retrieve cells with different physical volumes; one round test query has more than twice its retrieved neighbor’s volume. Matching within one population standard deviation is not exact size matching. Physical size should remain visible alongside shape similarity. These images do not establish biological preference or replace the qualified quantitative comparisons above.','']
    lines=['# Spatial shape benchmark: pose, conditional geometry and held-out groups','',conclusion,'',
        'Atlas remains unchanged. These are mesh-derived geometry experiments, not raw-microscopy biological phenotype validation.','',
        '## Frozen selection and data history','',
        f"The development-only rule selected **{chosen.upper()}** using validation reconstruction error among aligned models with at least 95% exact-rotation identity. Source, model and nuisance-regression hashes were frozen before test geometry was opened. Test retrieval did not choose the model.",'',
        f"Development: 96 training cells, 48 validation cells. Test: {test['cells']} cells in {test['groups']} TIFF-similarity groups, separate from development. Earlier project pipelines evaluated {lock['test_history']['previous_project_test_cell_overlap']} of these test-cell identities. The test set is held out for these spatial encoders, not globally untouched project data.",'',
        '## Rotation and frame robustness','',
        '| Development pipeline | Eight-rotation identity ↑ | Top-5 neighbor Jaccard ↑ | Fresh sampled-frame identity ↑ |',
        '|---|---:|---:|---:|']
    for name,value in development['pose'].items():
        stress=f"{value['sampled_frame_identity']:.1%}" if 'sampled_frame_identity' in value else '—'
        lines.append(f"| {name} | {value['rotation_identity']:.1%} | {value['rotation_neighbor_jaccard']:.1%} | {stress} |")
        if name.endswith('_aligned'):lines.append(f"| {name}, naive single frame | {value['single_frame_rotation_identity']:.1%} | — | — |")
    lines+=['',f"Selected test encoder: rotation identity **{pose['rotation_identity']:.1%}**, neighbor Jaccard **{pose['rotation_neighbor_jaccard']:.1%}**, sampled-frame identity **{pose['sampled_frame_identity']:.1%}**. Each rotation trial uses the full fixed test gallery, which is larger than the validation gallery.",'',
        '![Pose robustness](pose_summary.png)','',
        'Exact triangle-area covariance avoids random frame estimation. Four proper sign frames address eigenvector signs, but not all rotations within exactly degenerate eigenspaces. Fresh-frame stress estimates axes from two independent 2048-point samples and is distinct from the production exact-mesh frame. It does not simulate real segmentation errors. Same-cell identity is stricter than shape similarity for nearly indistinguishable cells; neighbor overlap provides additional context.','']
    for stage,metrics in [('development',development),('test',test)]:
        for name,value in metrics['pose'].items():
            if not name.endswith('_aligned'):continue
            for subgroup in ['near_degenerate','other','nearly_spherical']:
                if subgroup in value:
                    sub=value[subgroup];lines.append(f"- {stage}, {name}, {subgroup}: {sub['cells']} cells; sampled-frame identity {sub['sampled_frame_identity']:.1%}.")
    lines+=['','## Reconstruction','',
        '| Cohort / model | Base Chamfer ↓ | Rotated Chamfer ↓ | Base surface F1 ↑ | Missing / nonclosed base surfaces |',
        '|---|---:|---:|---:|---:|']
    for stage,metrics in [('validation',development),('test',test)]:
        for name,value in metrics['reconstruction'].items():
            if name.endswith('_rotated'):continue
            rotated=metrics['reconstruction'][name+'_rotated']
            lines.append(f"| {stage} / {name} | {value['squared_chamfer']:.5f} | {rotated['squared_chamfer']:.5f} | {value['f1_at_005_radius']:.1%} | {value['failed_surfaces']} / {value['nonwatertight']} |")
    lines+=['','Surface F1 tolerance is 5% of normalization radius. Reconstructions use a 56³ extraction grid and independent 2048-point surface samples. Only the first of the eight predetermined rotations is reconstructed; all eight contribute to embedding robustness.','',
        '## Additional geometry after morphometry adjustment','',
        'Candidates are from other groups and matched within distance 1 in training-standardized log volume, sphericity and elongation, with at least five and at most twenty candidates. Residual features subtract a training-only polynomial Ridge prediction from invariant orbit mean/std features. Nine-measure controls add area, flatness, solidity and axis lengths.','',
        f"**Primary held-out comparison:** {chosen.upper()} residual features versus nine measures: group-mean geometry-cost improvement **{primary['mean']:.6f}**, 95% group-bootstrap interval **[{ci[0]:.6f}, {ci[1]:.6f}]**, {primary['groups']} groups; {test['retrieval']['queries']} eligible test queries. Positive favors the residual embedding.",'',
        f"Descriptive ratio of group-mean costs: {relative:.1%} lower error for residual retrieval (negative would mean higher error). The confidence interval above concerns the paired absolute difference, not this ratio.",'',
        '| Cohort / method | Improvement vs three controls (95% CI) | Improvement vs nine measures (95% CI) |',
        '|---|---:|---:|']
    for stage,metrics in [('validation',development),('test',test)]:
        for name,baselines in metrics['retrieval']['methods'].items():
            formatted=[]
            for baseline in ['three_measures','nine_measures']:
                result=baselines[baseline];lo,hi=result['ci95'];formatted.append(f"{result['mean']:.6f} [{lo:.6f}, {hi:.6f}]")
            lines.append(f"| {stage} / {name} | {' | '.join(formatted)} |")
    lines+=['','Only the selected residual-versus-nine comparison is the prespecified primary held-out contrast; other intervals are exploratory. Geometry uses independently sampled surfaces and multi-start proper-rotation ICP, permitting neither scale changes nor reflections. It remains a proxy derived from predicted segmentations. No biological-condition labels are available.','',
        'Residual adjustment is not proof of complete statistical independence. Remaining absolute correlations with the three controls:','']
    for stage,metrics in [('validation',development),('test',test)]:
        for kind,value in metrics['retrieval']['residual_diagnostics'].items():
            lines.append(f"- {stage} {kind}: mean {value['mean_residual_control_correlation']:.3f}; maximum {value['max_residual_control_correlation']:.3f}.")
    lines+=['','## Predetermined visual checks','',
        'Contact sheets use low/mid/high-sphericity queries among matched queries, not queries chosen for success. Columns show the source query and the source cells retrieved by the encoder, residual features and nine measures. Display alignment is independent from the retrieval ranking. These are qualitative inspections, not a blinded human study.','',
        '![Validation neighbors](development/neighbors.png)','', '![Held-out neighbors](test/neighbors.png)','',
        *supplement,'## Decision and limits','',conclusion,'',
        'Sign-frame orbit comparison stores four codes per cell (380 PCA or 512 CNN values); it is not a single 95D/128D invariant embedding. Residual summaries use 190/256 values and lose some information. One dataset, one training seed, sparse z-derived predicted surfaces and reused project test groups limit external validity. Similarity groups are not verified biological specimen identities. Biological utility and robustness to real segmentation changes require new specimens and condition-labelled data.','',
        f"Atlas unchanged: development `{development['atlas_unchanged']}`, test `{test['atlas_unchanged']}`. Six targeted pose/geometry/residualization/seal checks passed. Frozen pipeline manifest SHA-256: `{sha256(OUT/'freeze.json')}`.",'',
        'Full protocol: `docs/SPATIAL_POSE_BENCHMARK.md`. Per-cell reconstruction, retrieval choices, raw code arrays, source snapshots and the opening record remain in this experiment directory.']
    for stage in ['development','test']:contact_sheet(stage,chosen)
    pose_plot(development,test,chosen)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(OUT/'report.md')


if __name__=='__main__':report()
