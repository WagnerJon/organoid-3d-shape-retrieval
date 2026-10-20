"""Produce a reviewable report without changing Atlas."""
import json
import shutil
from consistent_sdf import OUT,SOURCE
from pointnet_shape_core import ROOT,sha256,save_json


def report():
    metrics=json.loads((OUT/'evaluation/metrics.json').read_text());methods=metrics['methods']
    prior=json.loads((SOURCE/'evaluation/metrics.json').read_text())['methods']
    table=['| Method | Chamfer ↓ | Surface F1 ↑ | Volume error ↓ | Fresh-sample identity ↑ | Effective rank |',
           '|---|---:|---:|---:|---:|---:|']
    entries=[('Starting staged encoder',prior['encoder']),('Previous encoder + per-cell SDF fitting',prior['refined'])]
    labels={'control_best':'SDF-only, selected checkpoint','paired_best':'Paired consistency, selected checkpoint',
            'control_last':'SDF-only, final epoch','paired_last':'Paired consistency, final epoch'}
    entries.extend((labels[key]+f" (epoch {value['epoch']})",value) for key,value in methods.items())
    for name,value in entries:
        table.append(f"| {name} | {value['squared_chamfer']:.5f} | {value['f1_at_005_radius']:.1%} | {value['volume_relative_error']:.1%} | {value['resampling_self_top1']:.1%} | {value['effective_rank']:.2f} |")
    control=methods['control_last'];paired=methods['paired_last']
    verdict=(f"At the fixed final epoch, consistency changes fresh-sample identity retrieval from {control['resampling_self_top1']:.1%} to {paired['resampling_self_top1']:.1%}, "
             f"but changes Chamfer from {control['squared_chamfer']:.5f} to {paired['squared_chamfer']:.5f} and F1 from {control['f1_at_005_radius']:.1%} to {paired['f1_at_005_radius']:.1%}. "
             'This pilot does not justify adding these embeddings to Atlas. Sampling repeatability remains inadequate for reliable cell similarity search.')
    lines=['# Surface-sampling consistency experiment','',
           'Atlas is unchanged. This is an isolated development experiment, not an application feature.','',
           verdict,'',
           '## Question and controlled protocol','',
           'Can an encoder learn repeatable representations from independent surface samples while preserving SDF reconstruction, without predicting arbitrary fitted teacher vectors? Both arms warm-start the same staged encoder and use the same frozen decoder, 96 training cells, two independent 2048-point views, rotations, minibatches and 12 epochs. The paired arm adds vector agreement and norm-feature instance discrimination. No fitted teacher vectors are targets. No validation-cell code fitting is performed in either new arm.','',
           'The 48 validation cells occupy 11 separate TIFF similarity groups. A third independent point view is held out from training. Best checkpoints use validation SDF error, including epoch zero. Final-epoch results are also shown. No test cells were used.','',
           '## Results','',*table,'',
           'The previous per-cell fitting row is context, not an equal-information comparison: it optimizes each validation code against that cell’s known segmentation SDF. The starting and previous rows use the preceding evaluation’s independent surface samples; the four new arms share exactly matched fresh samples in this evaluation. Lower Chamfer is better; F1 uses a tolerance of 5% of the normalization radius.','',
           '![Matched final-epoch reconstructions](evaluation/comparison.png)','',
           '![Training trajectories](evaluation/training_curves.png)','',
           '## Paired geometry comparison','']
    for selection,value in metrics['paired_group_geometry_improvement'].items():
        lines.append(f"- {selection}: group-mean control minus paired Chamfer = {value['mean']:.6f}; 95% group bootstrap interval {value['ci95']}. Positive favors consistency; {value['groups']} groups.")
    lines+=['','## Rotation, surface integrity and morphometry-controlled retrieval','']
    for key,value in methods.items():
        retrieval=value['conditional_retrieval']
        lines.append(f"- {labels[key]}: rotation identity {value['rotation_self_top1']:.1%}; relative vector equivariance error {value['relative_rotation_error']:.3g}; {value['failed_surfaces']} missing surfaces, {value['nonwatertight']} nonclosed surfaces, {value['boundary_crossings']} box crossings. Retrieval diagnostic: `{json.dumps(retrieval)}`.")
    lines+=['','## Interpretation and limits','',
           'Improved sampling identity is useful only if shape fidelity and representation diversity are retained. Effective rank is a collapse diagnostic, not biological validation. The morphometry-controlled retrieval test uses a geometric proxy and does not establish biological information beyond volume, elongation or sphericity.','',
           'The visual comparison still shows rounded reconstructions replacing distinctive cell geometry. The paired arm’s best SDF checkpoint is epoch zero: none of its updates improves the starting validation SDF error. Its final code diversity is also lower than the control. These observations suggest an encoder/bottleneck problem that this consistency objective does not repair; they do not establish that all consistency methods or larger models would fail.','',
           'Next, isolate encoder capacity and geometry preservation with a small encoder overfit test before another similarity-learning run. Retain the successful per-cell SDF fitting as a reconstruction reference. A richer local representation is a candidate for that next experiment, not yet a validated solution.','',
           'This short single-seed pilot uses only three precomputed surface samples per cell and three within-minibatch negative cells. Validation data have been reused for development and checkpoint selection; these are not independent test estimates. The encoder is a reduced-compute paper-inspired implementation, not an exact published reproduction. Its global vector bottleneck still has the symmetry limitations documented in the preceding experiment.','',
           f"Atlas data SHA-256 remains `{sha256(ROOT/'results/explorer_dataset/dataset.json')}`. Verification: `{metrics['atlas_unchanged']}`.",
           '', 'Full metrics: `evaluation/metrics.json`; per-cell measurements: `evaluation/cell_metrics.csv`; protocol: `docs/CONSISTENT_SDF_EXPERIMENT.md` in the repository.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    files=['consistent_sdf.py','evaluate_consistent_sdf.py','report_consistent_sdf.py']
    for name in files:
        destination=OUT/'source_snapshot'/name
        if not destination.exists():shutil.copy2(ROOT/'scripts'/name,destination)
    save_json(OUT/'completed_source_hashes.json',{name:sha256(ROOT/'scripts'/name) for name in files})
    print(OUT/'report.md')


if __name__=='__main__':report()
