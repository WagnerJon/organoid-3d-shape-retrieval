"""Build a compact experiment report, without changing the Atlas dataset."""
import json
import os
import pandas as pd
from pointnet_shape_core import ROOT
from prepare_sdf_shape import OUT


def main():
    manifest=pd.read_csv(OUT/'manifest.csv')
    lines=['# SDF shape pilot results','',
        'This experiment is isolated from Organoid Atlas. No new model, embedding, mesh, or UI was added to the app.','',
        '## Scope','',
        f"Training: {sum(manifest.split=='train')} cells from {manifest[manifest.split=='train'].similarity_group.nunique()} similarity groups. "
        f"Validation: {sum(manifest.split=='validation')} cells from {manifest[manifest.split=='validation'].similarity_group.nunique()} separate groups. "
        'The overfit sanity check uses 12 training cells. No test cells were selected.','',
        'Paper-inspired reduced-compute experiment, not an exact reproduction of the published model. '
        'Both new models use 512 surface points and 128 analysis features. VN training retains 128 three-dimensional vectors; '
        'PointNet++ training retains 128 scalars. Architectures are not parameter matched. '
        'The existing point-cloud AE used 2,048 input points and 469 training cells.','']
    validation_path=OUT/'pilot_evaluation/metrics.json'
    if validation_path.exists():
        v=json.loads(validation_path.read_text())['methods'];old=v['baseline']['squared_chamfer']
        conclusion=['## Decision: do not integrate this pilot','',
            '**Neither new end-to-end model improves reconstruction on the validation cells.** '
            f"PointNet++/SDF has {v['pointnet']['squared_chamfer']/old-1:.0%} higher sampled-surface Chamfer error than the existing AE; "
            f"VN-DGCNN/SDF has {v['vn']['squared_chamfer']/old-1:.0%} higher error. "
            'The predefined example figures show persistent average rounded shapes and loss of individual geometry.','',
            f"VN-DGCNN reaches {v['vn']['rotation_self_top1']:.0%} rotation identity retrieval but only "
            f"{v['vn']['resampling_self_top1']:.1%} under independent surface resampling. "
            'Rotation invariance alone is insufficient; the learned representation is not reliable for shape retrieval.','',
            'The free-code diagnostic shows that these SDF decoders can capture the cell geometry much better. '
            'It suggests an encoder/optimization bottleneck, but free codes, a small training set, and different update counts '
            'prevent a definitive attribution. Those fits are not transferable embeddings and are not validation results.','',
            '**Recommended next experiment:** keep SDF reconstruction, first train a shared decoder with optimized shape codes, '
            'then test whether an encoder can predict those codes without losing geometry. Compare that strategy with a fuller '
            'reference VN-DGCNN setup and larger surface samples. Require both faithful held-out reconstruction and resampling '
            'stability before Atlas integration. This is a proposed follow-up, not implemented evidence.','',
            'This reduced-compute pilot does not reproduce the paper’s full point budget, architecture, data or training setup. '
            'Its failure is evidence against integrating this implementation, not against the published approach.','']
        lines[4:4]=conclusion
    diagnostic_path=OUT/'decoder_diagnostic/metrics.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        lines+=['## Decoder-only capacity diagnostic','',
            'Free per-cell codes and the existing decoder were optimized on the same 12 training cells for 1,500 updates. '
            'These codes are not outputs of an encoder and are not usable for new cells. This is not held-out accuracy. '
            'The initial end-to-end sanity check had 500 updates, so this comparison does not isolate the encoder alone.','',
            '| Decoder | Squared Chamfer ↓ | P95 point error (µm) ↓ | Surface F1 ↑ |','|---|---:|---:|---:|']
        for kind,m in diagnostic['methods'].items():
            lines.append(f"| {kind} + free codes | {m['squared_chamfer']:.5f} | {m['p95_surface_error_um']:.3f} | {m['f1_at_005_radius']:.1%} |")
        lines+=['','![Decoder capacity](decoder_diagnostic/decoder_capacity.png)','']
    for mode in ['overfit','pilot']:
        folder=OUT/(mode+'_evaluation')
        if not (folder/'metrics.json').exists():continue
        data=json.loads((folder/'metrics.json').read_text())
        if mode=='pilot' and (OUT/'baseline_stability.json').exists():
            data['methods']['baseline'].update(json.loads((OUT/'baseline_stability.json').read_text()))
        lines += ['## '+('Training reconstruction sanity check' if mode=='overfit' else 'Held-out group validation'),'',
            '| Method | Squared Chamfer ↓ | P95 point error (µm) ↓ | Surface F1 ↑ | Resampling identity ↑ | Rotation identity ↑ | Effective rank |',
            '|---|---:|---:|---:|---:|---:|---:|']
        names=dict(baseline='Existing PointNet++ point AE',pointnet='PointNet++ + SDF',vn='VN-DGCNN + SDF')
        for kind,metrics in data['methods'].items():
            def percentage(key):return f"{100*metrics[key]:.1f}%" if key in metrics else '—'
            rank=f"{metrics['effective_rank']:.2f}" if 'effective_rank' in metrics else '—'
            lines.append(f"| {names[kind]} | {metrics['squared_chamfer']:.5f} | {metrics['p95_surface_error_um']:.3f} | {percentage('f1_at_005_radius')} | {percentage('resampling_self_top1')} | {percentage('rotation_self_top1')} | {rank} |")
        lines+=['', 'F1 tolerance is 5% of each original normalization radius. Point metrics include surface-sampling error. '
                 'Micrometer values inherit the dataset’s inferred calibration. Identity retrieval is not biological classification.','']
        for kind in ['pointnet','vn']:
            m=data['methods'][kind]
            lines += [f"- {names[kind]}: mean absolute relative volume error {m['volume_relative_error']:.1%}; "
                      f"failed surfaces {m['failed_surfaces']}; nonclosed meshes {m['nonwatertight']}; "
                      f"extraction-box crossings {m['boundary_crossings']}; "
                      f"maximum joint-rotation SDF discrepancy {m['maximum_joint_rotation_sdf_error']:.3g}."]
        lines+=['',f"Atlas dataset hash unchanged: **{data['atlas_unchanged']}**.",'',
            f'![Reconstruction comparison]({mode}_evaluation/reconstruction_comparison.png)','',
            f'![Midplane comparison]({mode}_evaluation/midplane_comparison.png)','']
        if mode=='pilot':
            lines+=['The pilot ran 120 epochs, warm-started from the corresponding 250-epoch training-only sanity check. '
                    'It did not use the diagnostic free codes or decoder weights.','']
            lines+=['### Paired reconstruction comparisons','',
                    'Positive differences favor the second method. Group means and 95% bootstrap intervals:','']
            for pair,result in data['paired_group_comparisons'].items():
                lo,hi=result['ci95'];lines.append(f"- {pair}: {result['mean']:.6f} [{lo:.6f}, {hi:.6f}], {result['groups']} groups.")
            lines+=['','### Retrieval beyond basic morphometry','',
                    'An exploratory geometry proxy among other groups, matched for volume, sphericity and elongation; '
                    'this is not evidence of biological usefulness.','']
            for kind in ['pointnet','vn']:
                retrieval=data['methods'][kind]['conditional_retrieval'];ci=retrieval.get('group_bootstrap_95_ci')
                lines.append(f"- {names[kind]}: {retrieval['queries']} queries, {retrieval['groups']} groups; "
                             f"mean geometry-cost advantage {retrieval['mean_group_improvement']}; 95% interval {ci}.")
    lines+=['## Interpretation limits','',
        'Validation selects checkpoints; these are exploratory development results, not final independent test estimates. '
        'Targets are existing segmentations, not manually established biological ground truth. '
        'The experiment does not recover information missing from sparse Z imaging. '
        'Comparison against the older AE also changes training-set size and input-point budget. '
        'Better reconstruction does not establish biological value beyond morphometry.','',
        'Methods, commands and safeguards: [SDF_SHAPE_PILOT.md](../../docs/SDF_SHAPE_PILOT.md).','',
        'Reference: [Vasan et al., Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02729-9).','']
    if validation_path.exists():
        lines+=['The PointNet++/SDF matched-morphometry proxy has a small positive group-bootstrap interval in this pilot. '
                'This exploratory signal does not outweigh its low resampling stability, worse reconstruction, or lack of '
                'biological outcome labels. No claim of biological information beyond traditional morphometry is established.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,['overfit','pilot']):
        for kind in ['pointnet','vn']:
            path=OUT/f'{mode}_{kind}'/'history.csv'
            if not path.exists():continue
            history=pd.read_csv(path)
            ax.plot(history.epoch,history.train_sdf_l1,label=kind+' train')
            valid=history.dropna(subset=['validation_sdf_l1']);ax.plot(valid.epoch,valid.validation_sdf_l1,'--',label=kind+' validation')
        ax.set(title=mode,xlabel='Epoch',ylabel='Clipped-target SDF L1');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    print(OUT/'report.md')


if __name__=='__main__':main()
