"""Create an honest baseline report and original/predicted surface-point figure."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,read_config


def report(config):
    run=ROOT/config['output_dir']/'baseline'
    metrics=json.loads((run/'validation_metrics.json').read_text());decision=json.loads((run/'stage2_decision.json').read_text())
    synthetic=json.loads((run/'synthetic_benchmark.json').read_text()) if (run/'synthetic_benchmark.json').exists() else None
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=np.load(run/'validation_reconstructions.npz')
    selected=np.linspace(0,len(examples['cell_ids'])-1,3,dtype=int)
    fig=plt.figure(figsize=(10,8))
    for row,index in enumerate(selected):
        for column,key in enumerate(['inputs','reconstructions']):
            ax=fig.add_subplot(3,2,row*2+column+1,projection='3d');points=examples[key][index]
            ax.scatter(*points[::2].T,s=1,c='#18a8b8' if column==0 else '#f09238',alpha=.6)
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1));ax.set_box_aspect((1,1,1));ax.set_axis_off()
            ax.set_title(f'{examples["cell_ids"][index]} · '+('Input surface' if column==0 else 'Decoded 128D embedding'),fontsize=10)
    fig.suptitle('PointNet++ baseline · normalized geometry, original orientation',fontsize=13)
    fig.tight_layout();fig.savefig(run/'reconstruction_examples.png',dpi=160);plt.close(fig)
    history=pd.read_csv(run/'history.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(history.epoch,history.train_chamfer,label='Training (random rotations)');ax.plot(history.epoch,history.validation_chamfer,label='Validation (fixed pose)')
    ax.set(xlabel='Epoch',ylabel='Symmetric squared Chamfer',title='Training and validation reconstruction');ax.legend();fig.tight_layout();fig.savefig(run/'training_curve.png',dpi=150);plt.close(fig)
    failed=[name for name,passed in decision['checks'].items() if not passed]
    text=f'''# Surface PointNet++ baseline

2,048 area-weighted surface points → PointNet++ (512/128 centers, 32 neighbors) → 128D latent → MLP point decoder.

Geometry only; original size is retained separately. The encoder is not rotation invariant. Reconstruction error uses normalized predicted segmentation surfaces, not biological ground truth. Training and model selection use separate similarity groups; original organism identities are unavailable.

| Validation measure | Result |
|---|---:|
| Cells / similarity groups | {metrics['cells']} / {metrics['groups']} |
| Best checkpoint epoch | {metrics['checkpoint_epoch']} |
| Symmetric squared Chamfer | {metrics['mean_squared_chamfer']:.6f} |
| Untrained model | {metrics['untrained_chamfer']:.6f} |
| Training-surface template | {metrics['training_template_chamfer']:.6f} |
| Improvement over template | {metrics['template_improvement']:.1%} |
| Resampling identity top-1 | {metrics['resampling_self_top1']:.1%} |
| Rotated identity top-1 | {metrics['rotation_self_top1']:.1%} |
| Effective embedding rank | {metrics['effective_rank']:.2f} |

Stage-two decision: **{decision['status']}**.

Unmet checks: {', '.join(failed) or 'none'}. Decisions use validation only; the held-out test set does not select the architecture. The geometry-matched retrieval diagnostic compares candidates from different similarity groups, controlling volume, sphericity and elongation. Its PCA/sign-aligned point Chamfer is an imperfect geometry proxy and can be ambiguous for nearly symmetric surfaces. Details and the group-bootstrap interval are recorded in `validation_metrics.json`.

'''
    if synthetic:text+=f"Synthetic rotated/resampled identity top-1: {synthetic['rotated_resampled_identity_top1']:.1%}; exact-descriptor baseline: {synthetic['classical_identity_top1']:.1%}. This is a stress test, not evidence of biological relevance.\n\n"
    text+='''![Input and reconstruction](reconstruction_examples.png)

![Learning curve](training_curve.png)

A future SDF stage must separate shape from pose: use equivariant vector features to produce an invariant latent code and supply orientation to the coordinate decoder separately. Use both near-surface and volume samples, a documented negative-inside sign convention and physically calibrated error. Never force an invariant code to reconstruct arbitrary poses without a pose pathway. Do not advance on reconstruction fidelity alone.
'''
    (run/'report.md').write_text(text)
    print('Report: '+str(run/'report.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');args=parser.parse_args();report(read_config(ROOT/args.config))
