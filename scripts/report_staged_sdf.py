"""Summarize staged shape coding, preserving encoder/fitting distinctions."""
import json
import os
import pandas as pd
from pointnet_shape_core import ROOT,sha256
from prepare_staged_sdf import OUT


def main():
    manifest=pd.read_csv(OUT/'manifest.csv');metadata=json.loads((OUT/'data.json').read_text())
    lines=['# Staged SDF shape experiment','',
        '**Organoid Atlas is unchanged.** This experiment exports no models, embeddings or meshes to the app.','',
        'The shared decoder learns from 96 training cells in 21 similarity groups. Evaluation uses 48 cells in 11 '
        'other groups. No test cells are used. The encoder receives 2,048 surface points, up from 512 in the preceding pilot.','']
    names=dict(baseline='Existing PointNet++ AE',encoder='Staged VN encoder only',refined='Encoder + SDF code fitting',fitted_zero='SDF code fitting, no encoder')
    complete=OUT/'evaluation/metrics.json'
    current=complete if complete.exists() else OUT/'decoder_validation/metrics.json'
    if current.exists():
        data=json.loads(current.read_text());methods=data['methods'];folder=current.parent
        if complete.exists():
            refined=methods['refined'];baseline=methods['baseline']
            decision=['## Decision','',
                '**Keep the SDF reconstruction direction; do not integrate these embeddings into Atlas yet.** '
                f"Encoder-initialized fitting reduces sampled-surface Chamfer by {1-refined['squared_chamfer']/baseline['squared_chamfer']:.1%}, "
                f"with {refined['f1_at_005_radius']:.1%} surface F1 and {refined['volume_relative_error']:.2%} mean volume error. "
                f"Its independent-observation identity retrieval is only {refined['resampling_self_top1']:.1%}. "
                'The encoder alone still loses geometry, and matched-morphometry retrieval does not establish an advantage over classical features.','',
                'The best geometry requires fitting a per-object code to distance observations from its known segmentation. '
                'It is a useful reconstruction result, not a fast encoder or a validated biological similarity space.','',
                '**Next direction:** avoid forcing an equivariant encoder to reproduce arbitrary free vector codes. '
                'Test symmetry-compatible shape/pose coding or field-level training with explicit independent-sampling consistency. '
                'Faithful reconstruction, repeatable similarity and independent biological validation remain separate requirements.','']
            lines[4:4]=decision
        best=min((k for k in methods if k!='baseline' and methods[k]['squared_chamfer'] is not None),key=lambda k:methods[k]['squared_chamfer'])
        ratio=methods[best]['squared_chamfer']/methods['baseline']['squared_chamfer']
        lines+=['## Observed result','',f"Lowest reconstruction error: **{names[best]}**. "
            f"Its sampled-surface squared Chamfer is {ratio:.2f}× the existing AE on these validation cells.",'',
            '**Fitted codes use signed-distance observations from the known segmentation.** '
            'Their shared decoder weights remain frozen, but each new object receives an optimization step. '
            'They are not encoder-only inference or proof of generalization from microscopy intensities. '
            'The alternate stability view uses disjoint observations and a new input-point sample.','',
            '| Method | Squared Chamfer ↓ | P95 error (µm) ↓ | Surface F1 ↑ | Resampling identity ↑ | Rotation identity ↑ | Effective rank |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method,m in methods.items():
            def number(key):return f"{m[key]:.5f}" if m.get(key) is not None else '—'
            def percent(key):return f"{m[key]:.1%}" if m.get(key) is not None else '—'
            rank=f"{m['effective_rank']:.2f}" if m.get('effective_rank') is not None else '—'
            lines.append(f"| {names[method]} | {number('squared_chamfer')} | {number('p95_surface_error_um')} | "
                         f"{percent('f1_at_005_radius')} | {percent('resampling_self_top1')} | {percent('rotation_self_top1')} | {rank} |")
        lines+=['','F1 uses a tolerance of 5% of the cell’s normalization radius. Point-sampling error is included; '
                'micrometer units inherit the dataset’s inferred calibration. Identity retrieval is not biological classification. '
                'Surface means exclude failed extractions; failures are reported separately and prevent accepting a method.','']
        for method,m in methods.items():
            if method=='baseline':continue
            volume=f"{m['volume_relative_error']:.1%}" if m.get('volume_relative_error') is not None else 'n/a'
            query=f"{m['heldout_query_sdf_l1']:.5f}" if m.get('heldout_query_sdf_l1') is not None else 'n/a'
            lines.append(f"- {names[method]}: volume error {volume}; "
                         f"held-out SDF-query L1 {query}; "
                         f"failed surfaces {m['failed_surfaces']}; nonclosed meshes {m['nonwatertight']}; extraction-box crossings {m['boundary_crossings']}.")
        lines+=['',f"Atlas fingerprint unchanged: **{data['atlas_unchanged']}**.",'',
                f"![Validation comparison]({folder.name}/comparison.png)",'',
                '## Paired reconstruction improvements','',
                'Positive mean and interval favor the new method over the original AE. Means are over similarity groups; '
                'intervals are 95% group-bootstrap intervals.','']
        for method,value in data['paired_group_improvements'].items():
            lo,hi=value['ci95'];lines.append(f"- {names[method]}: {value['mean']:.6f} [{lo:.6f}, {hi:.6f}], {value['groups']} groups.")
        lines+=['','## Geometry retrieval after matching basic morphometry','',
                'Exploratory cross-group geometry proxy among cells matched for volume, sphericity and elongation; '
                'PCA alignment is ambiguous for near-symmetric shapes. This cannot establish biological usefulness.','']
        for method,m in methods.items():
            r=m.get('conditional_retrieval')
            if not r:continue
            lines.append(f"- {names[method]}: {r['queries']} queries in {r['groups']} groups; "
                         f"mean geometry advantage {r['mean_group_improvement']}; 95% group interval {r['group_bootstrap_95_ci']}.")
    diagnostic=OUT/'decoder_validation/invariant_readout_diagnostic.json'
    consistency=OUT/'decoder_validation/field_consistency.json'
    if diagnostic.exists():
        values=json.loads(diagnostic.read_text())
        lines+=['','## Code-consistency diagnostic','',
                'The direct-code fits were also compared using two additional invariant readouts. '
                'This checks whether instability is simply caused by discarding relative vector angles; '
                'these are diagnostics, not selected or integrated feature methods.','']
        for name,value in values.items():
            lines.append(f"- {name}: independent-observation identity retrieval {value['resampling_self_top1']:.1%}.")
    if consistency.exists():
        value=json.loads(consistency.read_text())['mean_same_shape_field_difference_l1']
        lines+=['',f'The two direct-code fits differ by {value:.5f} mean normalized SDF value on the same query locations. '
                'Similar decoded fields with different latent nearest neighbours are consistent with code ambiguity. '
                'This is not a proof of mathematical non-identifiability or biological equivalence.','']
    histories=OUT/'decoder_validation/fitting_histories.json'
    if histories.exists():
        entries=json.loads(histories.read_text()).get('fitted_zero',[])
        if isinstance(entries,list) and entries:
            seconds=sum(x.get('seconds',0) for x in entries);cells=sum(len(x['cell_ids']) for x in entries)
            lines+=['',f'Direct code-fitting throughput for one observation set: approximately {seconds/cells:.2f} '
                    'seconds per cell in batches of eight on this run. This excludes preparation and surface meshing; '
                    'it is not interactive UI latency.','']
    lines+=['','## Method and limits','',
        'The decoder is trained jointly with free 128×3 vector codes, then frozen. The reference-style vector encoder '
        'learns those codes using both code regression and reconstruction. This is independently implemented and smaller '
        'than the published model; it is not a paper reproduction. Code norms yield 128 invariant scalar features. '
        'They need not contain all the information in the vector code, and geometry-fitting codes need not be unique.','',
        'The original AE used 469 training cells and a different architecture. This follow-up changes input density, '
        'architecture and training objective simultaneously. It does not isolate which change caused an effect. '
        'Validation also selects encoder checkpoints, so these are development results, not final independent test estimates. '
        'Existing segmentations are reference shapes, not manual biological ground truth; no sparse-Z detail can be recovered '
        'that is absent from them. Biological usefulness requires independent condition/phenotype data.','',
        'Detailed protocol: [STAGED_SDF_EXPERIMENT.md](../../docs/STAGED_SDF_EXPERIMENT.md). '
        'Source snapshots, fixed settings, checkpoints and per-cell metrics are stored alongside this report.','']
    lines+=['## Symmetry and code-target diagnostic','',
        'A synthetic point cloud invariant under 180-degree rotations about two different axes forces every '
        'permutation-invariant SO(3)-equivariant global vector to be zero: the vector must remain unchanged '
        'under both rotations. The encoder passes this check. Unconstrained nonzero free vector codes need not '
        'obey that constraint, even if they decode a symmetric surface. Thus arbitrary auto-decoder vector '
        'codes can be incompatible regression targets for an equivariant encoder. Nearly symmetric real '
        'cells may also make such orientation choices unstable. This is a modeling risk, not proof that all '
        'observed failures have this cause, and does not apply as a criticism of the published jointly trained model.','']
    training=OUT/'encoder/history.csv'
    if training.exists():
        history=pd.read_csv(training)
        lines+=['',f"The encoder’s final recorded normalized code-regression MSE was {history.iloc[-1].train_code_mse:.3f}. "
                'A zero-vector predictor has training-set MSE 1 under this per-channel RMS normalization. '
                'These are training averages during optimization, not an independent regression benchmark; they further '
                'illustrate the difficulty of recovering the chosen free-code targets.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,name,xcol in [(axes[0],'decoder','step'),(axes[1],'encoder','epoch')]:
        path=OUT/name/'history.csv'
        if not path.exists():continue
        frame=pd.read_csv(path)
        for column in frame.columns:
            if 'sdf_l1' in column:ax.plot(frame[xcol],frame[column],label=column)
        ax.set(title=name,xlabel=xcol,ylabel='SDF L1');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    print(OUT/'report.md')


if __name__=='__main__':main()
