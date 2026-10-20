# Standalone Allen reference benchmark

This experiment is isolated from Organoid Atlas. It never exports vectors, changes
segmentation, or replaces an app build. The full source experiment is snapshotted
for comparisons; biological condition labels are unavailable.

## Released model, not the earlier small approximation

- Official AllenCell `benchmarking_representations` and its pinned CytoDL commit
  `a4a061d1808e0f94f906933642920142b581ba38`.
- Official NPM1 `Rotation_invariant_pointcloud_SDF.ckpt` (epoch 270,
  global step 578314), downloaded from the public AllenCell S3 dataset.
- Released VN-DGCNN encoder, 512 vector channels, hidden width 342, k=20,
  8,192 surface points, and released `DecoderInner` (width 342, five residual
  blocks). Strict checkpoint loading confirms 14,306,163 parameters.
- Invariant readout is the norm of each vector channel, matching the authors'
  embedding export. Vector codes, not their norms alone, condition the decoder.

Only optional dependency imports are adapted to run the selected path without the
legacy Lightning/CUDA/scatter/transformer stack. Unused optional branches fail
closed. Model computation uses the pinned source classes. Memory-bounded
execution chunks pointwise operators and query rows of the exact all-points kNN,
keeping all points/channels/global means. CPU float64 tests compare outputs,
input gradients, and encoder/decoder rotation identities against the released
forward. A separate full-size CPU/Metal audit records numerical differences.

## Mac limitations and local adaptation

The standard full-graph Metal probe exhausted GPU memory on this 16 GB Mac. Its
command-buffer error invalidates that run even though the process printed a
finite tensor. It is not performance evidence. FP32 chunked execution succeeded;
full end-to-end backward still took roughly 23 seconds per cell in the probe.
Reduced-precision probes were rejected, rather than silently changing precision.

Two **partial adaptation** arms accompany the unchanged pretrained reference:

1. Freeze the entire encoder; adapt only the released decoder.
2. Freeze the backbone; adapt its final 2736→512 VNLinear and the released decoder.

The last encoder operator is linear and commutes with mean pooling, so caching
mean pre-head vectors preserves the full-resolution representation exactly up to
floating-point arithmetic. Proper rotations apply jointly to cached vectors and
SDF queries. The two adaptation arms use identical source features, minibatch
orders and random-query budgets. Decoder-only adaptation cannot change the
embedding and therefore separates improved reconstruction from improved codes.

This is **not** full end-to-end 400–2,000 epoch training from the published recipe.
The local budget is 30 epochs, batch size 8, Adam at 0.0002, 2,048 queries per
training cell-step, and one fixed noised input view per training cell. The
backbone stays fixed. Epoch zero and every fifth epoch are evaluated on fixed
20,000-query SDF observations for all 140 validation cells; the best checkpoint
is selected by mean SDF L1. These deviations are explicit and should not be
interpreted as a definitive reproduction or architecture rejection.

## Inputs and split

Use all 469 eligible training cells from the existing 21 training similarity
groups. All 140 eligible validation cells (11 disjoint groups) select adaptation
checkpoints. The fixed 48-cell validation cohort provides directly comparable
reconstruction, retrieval and resampling results. Twelve prespecified cells,
covering the validation groups, receive fresh full-backbone rotation probes.
No test cells are prepared or evaluated.

Original unsmoothed predicted meshes supply area-weighted surface points and
signed-distance targets. The authors' preprocessing centers each mesh on its
bounding-box center and divides by its largest side. Each cell has 32,768 surface
samples and 32,768 uniform cube SDF observations. Exact libigl pseudonormal SDF
replaces the authors' mesh-to-SDF scanning; float32 replaces float16 storage.
Training features use 8,192 subsampled points with replacement and Gaussian noise
0.005. Validation features use clean input samples, with independent input
subsampling checked separately.

Rotation probes rotate already normalized input clouds. Recomputing axis-aligned
bounding-box normalization after arbitrary rotation can change scale; network
SO(3) invariance does not make that preprocessing operation invariant. Input
resampling is not a remeshing or segmentation-noise test.

Similarity groups are proxies for specimen independence. Geometry is from
predicted masks, not manually verified biological ground truth. Calibration is
inherited from the existing dataset.

## Comparisons

The protocol is saved before training or reading validation results.

- Raw invariant norms, train-standardized norms, and degree-2 Ridge adjustment
  for log volume, elongation and sphericity. Readout scaling/adjustment uses the
  original fixed 96 training cells, matching existing Atlas baselines.
- Existing nine-feature morphometry, equally adjusted morphometry, spatial PCA,
  adjusted spatial summaries, spherical harmonics, adjusted harmonics, and the
  original PointNet++ code are fixed references. Training and pretraining budgets
  differ; this is a practical candidate comparison, not a controlled architecture
  ablation.
- A secondary diagnostic compares full512×3 vector codes after optimal proper-SO(3) alignment in latent space. This retains inter-channel angles discarded by channel norms and is explicitly distinct from the published norm readout. It uses matched candidate pools rather than nuisance regression of pairwise distances.
- The original matched validation pools: other similarity groups, at least five
  candidates within one training SD in the three controls, capped at twenty.
  Compare 46 eligible queries by the existing proper-rotation ICP geometry proxy.
  Two thousand group bootstraps report descriptive paired intervals.
- Reconstruction against fresh 2,048-point source samples in the previous common
  normalization. Extract 56³ SDF meshes over ±0.6 author units, then transform them
  back. Report squared Chamfer, F1 at 5% of the common radius, P95 physical error,
  volume error, failed surfaces, closedness and box crossings. Failures must be
  shown alongside successful-case means.
- Fresh 8,192-point resampling on 48 cells; fresh full-backbone rotation on twelve.
  Algebraic rotation of cached vectors must not be presented as a fresh input test.

Validation chooses checkpoints and supports the exploratory comparison; it is
not an independent confirmatory test. Better reconstruction does not prove
biological utility. The primary contrast is adjusted head/decoder-adapted norms
versus identically adjusted nine-feature morphometry.

## Additional structural diagnostic

Two engineered exactly D2-symmetric ellipsoid point sets probe the restriction of
using only globally pooled SO(3)-equivariant vectors. In exact arithmetic every
such global vector must vanish for a D2-symmetric input. A float64 run checks this
on the actual released model; float32 results can break symmetry through numerical
rounding and nearest-neighbor ties. These engineered inputs do not estimate
natural-cell failure rates, and the diagnostic does not select checkpoints. Raw
results are kept separately from the primary real-cell benchmark.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/prepare_allen_reference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/cache_allen_reference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_allen_runtime.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_allen_reference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_allen_reference.py
```

GPU access may require execution outside the desktop sandbox. Public source,
checkpoint hashes, data manifests, protocol, runtime diagnostics, selected
checkpoints and results are kept under `results/allen_reference`. Training never
requires the full legacy dependency installation. Optional-import adapters are
in `scripts/allen_reference_core.py`; they do not alter the downloaded sources.

## Sources

- [Vasan et al., Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02729-9)
- [Official benchmark source and released-model instructions](https://github.com/AllenCell/benchmarking_representations)
- [Pinned CytoDL model source](https://github.com/AllenCell/cyto-dl/tree/a4a061d1808e0f94f906933642920142b581ba38)
