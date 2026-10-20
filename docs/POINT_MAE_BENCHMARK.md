# Standalone Point-MAE experiment

This is an offline comparison, not an Atlas integration. Results and figures are
written to `results/point_mae/`. The experiment leaves existing Atlas exports,
native app code and prior benchmark weights unchanged.

## Sources and architecture

- Official source: https://github.com/Pang-Yatian/Point-MAE
- Pinned commit: `7445a68068d707da6d959b66773bfef84c4e3e32`
- Published ShapeNet pretraining checkpoint:
  https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth
- Source and weight hashes are recorded in `protocol.json` before outcomes.
- The full source configuration is retained: 1,024 input points, 64 groups of
  32 points, 384 channels, 12 encoder / 4 decoder blocks, six attention heads,
  60% masking; 29,006,432 parameters. This study does not reproduce the original
  300-epoch ShapeNet training schedule.

Original neural class bodies execute from the cached author source through an
import adapter. CPU greedy FPS starts at point zero; Euclidean kNN and symmetric
squared Chamfer replace CUDA-only extensions. Neural layers run on Metal. These
adapters are checked against geometric definitions, explicit gradients and the
original neural forward path. CUDA-kernel numerical parity is not claimed.

## Cohorts and training

The existing eligible Allen cohort contains 469 training and 140 validation cells
from 21 and 11 disjoint similarity groups. No test cells are prepared. Mesh SHA
checks bind the inputs to the prior source surfaces. Existing 32,768-point surface
pools are reused; their earlier isotropic normalization is removed by Point-MAE's
sample-mean / maximum-radius normalization.

The released checkpoint is compared with 30 epochs of full-network self-supervised
adaptation: AdamW, initial learning rate 0.0001, weight decay 0.05, batch 8, cosine
schedule, seed 42. Each training epoch draws fresh 1,024-point subsets and proper
SO(3) rotations. Anisotropic scaling is omitted to preserve the cell shape target.
All 140 validation cells use fixed point draws and masks. The lowest validation
masked-patch loss at epoch 0 or every fifth epoch selects the checkpoint.
Neither retrieval scores nor biological labels select weights.

## Embedding comparison

The explicit custom descriptor concatenates mean and max pooling of all 64 final
unmasked tokens, yielding 768 dimensions. No untrained classification token is
used. Readouts are:

1. Native orientation.
2. Area-weighted surface-PCA frame, with features averaged over all four proper
   sign choices. This is external pose handling, not an invariant architecture.
   Near-repeated eigenvalues may make the frame unstable; no fallback is selected
   based on evaluation outcomes.

Both have raw, standardized and nuisance-adjusted versions. Standardization and
degree-two Ridge nuisance fits use the original 96 training cells, matching the
existing reference studies. Nuisances are log volume, elongation and sphericity.

The prespecified primary comparison is adapted pose-averaged, standardized,
nuisance-adjusted Point-MAE against identically adjusted traditional features.
The same 48 validation cells, 46 eligible queries and matched other-group candidate
pools are used as in the Allen study. Existing proper-rotation ICP geometry costs
are reused; new selected pairs use the same 384-point samples and geometry helper.
Raw nine-feature morphometry, spatial PCA, spherical harmonics, CAJAL and Allen
provide frozen baselines. Their training/pretraining budgets differ.

Fresh 1,024-point resampling checks cover 48 cells. Twelve prespecified cells are
rotated and compared against the 48-cell gallery. PCA eigensystems and point
normalization are recomputed. No remeshing/segmentation-perturbation claim follows
from these tests. Stability is measured on raw descriptors.

## Reconstruction interpretation

Point-MAE completes masked local neighborhoods while receiving the true missing
patch centers and visible patches. This is not a global-bottleneck reconstruction
from the pooled embedding. Report patch Chamfer separately from Allen/PCA whole-cell
surface F1. The center-only baseline repeats each known patch center 32 times.
Visuals retain visible input and label it separately from predicted missing points.

## Reproduction

The cached official source, checkpoint and existing Allen/source manifests are
required. From the project root in the configured local Python environment:

```sh
.venv/bin/python -m unittest discover -s tests -p test_point_mae_reference.py -v
.venv/bin/python scripts/run_point_mae_reference.py
```

The runner resumes saved five-epoch checkpoints (including optimizer, NumPy, CPU
and Metal RNG states), then evaluates and generates a report. Derived evaluations
are tied to checkpoint/protocol hashes and fail closed on stale caches. The
report preserves an implementation snapshot. Hardware/kernel scheduling can still
cause floating-point variation.

## Limits

Results concern predicted segmentation geometry, not biological conditions or
phenotypes. Validation also selects the adaptation checkpoint. Bootstrap intervals
are descriptive, group-based and uncorrected for exploring multiple readouts.
This is not an independent confirmatory test or a compute-matched architecture
comparison. A geometric improvement alone does not establish additional biological
information beyond conventional morphometry.
