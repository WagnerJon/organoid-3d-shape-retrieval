# Geometry-only cell shape encoding

This implements the first stage discussed in **3D Shape Encoding**: an original segmented cell surface → 2,048 area-weighted points → PointNet++ → a deterministic 128D vector → 2,048 reconstructed surface points. No microscopy intensities or hand-designed measurements enter the network. The original segmentation and measured surfaces are preserved.

## Preparation and separation of shape from size

The Atlas export already contains physical marching-cubes meshes. We reuse those unsmoothed original triangle surfaces rather than extracting surfaces from downsampled display labels. Sampling probabilities follow triangle area, not vertex density. Input and target are independently sampled from the same surface. Each input is centered and divided by its maximum radius; the target uses that same transform. Original volume, surface area, axis dimensions, XYZ center and radius in micrometers remain in the manifest. Decoded points are returned to physical coordinates for display. Shape-only embeddings therefore do **not** retain absolute volume by design.

All 1,088 cells are prepared so Atlas can inspect predictions for any cell. Only the 744 non-excluded, QC-passing cells are eligible for training or evaluation: 469 train, 140 validation and 135 test. The split inherits the previously audited image-similarity groups, not the older random stack split. Stacks and their near-duplicate groups cannot cross partitions. Grouping is a proxy: biological specimen identities are still unavailable, and inferred physical calibration remains unverified. Excluded/QC-failed surfaces are inference-only and do not enter the training loss, early stopping or the architecture gate.

## Model and training

Two kNN set-abstraction stages use farthest-point centers (512 then 128), 32 neighbors and shared MLP/max aggregation, followed by global pooling and a 128D latent layer. A compact MLP decodes the points. The implementation uses existing PyTorch, NumPy and SciPy, with no compiled CUDA-only point-cloud extension; CPU preparation caches the geometry neighborhoods. Rigid rotation preserves those neighborhoods. Mac Metal acceleration was checked with the local MPS runtime.

The baseline minimizes symmetric mean **squared** Chamfer distance to the independently sampled target. Distance blocks limit memory use. Proper random SO(3) rotations affect input and reconstruction target together. There is no incompatible requirement that an invariant code reconstruct arbitrary input poses: **this baseline is not rotation invariant**. Latents remain unnormalized for reconstruction; their normalized copies use cosine distance for Atlas retrieval.

Training indexes only eligible train and validation rows. Validation selects the best checkpoint and stops after ten non-improving epochs, up to forty epochs. Checkpoints retain configuration, manifest fingerprint, optimizer and RNG state. A short smoke run is kept separately and cannot be published. No condition labels, perturbation labels or biological cell classes are used.

## Reproduce

From the repository root, with the existing environment and Atlas export:

```bash
.venv/bin/python scripts/run_pointnet_shape.py --device mps --publish
```

The runner prepares missing geometry, trains, evaluates validation, runs the synthetic stress test, writes a report and optionally attaches predictions to Atlas. For interrupted/completed runs use `--resume`; use `--device cpu` without Metal. Preparation refuses to overwrite an existing prepared dataset. Change `output_dir` in a copied configuration for a different experiment, and prepare again if membership, mesh sources or model geometry change.

Individual steps:

```bash
.venv/bin/python scripts/prepare_pointnet_shape.py
.venv/bin/python scripts/train_pointnet_shape.py --device mps
.venv/bin/python scripts/evaluate_pointnet_shape.py --device mps
.venv/bin/python scripts/benchmark_pointnet_shape.py --device mps
.venv/bin/python scripts/report_pointnet_shape.py
.venv/bin/python scripts/export_pointnet_shape.py --device mps
```

A separate, explicit `evaluate_pointnet_shape.py --split test` can assess the frozen model; it never changes the stage-two decision. Publishing includes inference embeddings for test cells so they can be explored, but does not use them for model fitting or model selection. Exploratory inspection is not an untouched external validation study.

## What would justify the SDF stage?

Thresholds are fixed in `config/pointnet_shape.json` before training, and `stage2_decision.json` uses validation only:

- At least ten completed full-training epochs, 30 validation cells and three validation groups.
- At least 15% lower reconstruction error than an untrained network, and 5% lower than a real training-surface template selected using the median radial profile.
- At least 80% same-cell retrieval under independent surface resampling and an effective embedding rank of at least eight.
- Evidence of complementary geometry retrieval after volume, sphericity and elongation matching: the lower group-bootstrap confidence bound for improvement over standardized classical morphometry must exceed zero.

The retrieval diagnostic uses candidates from different similarity groups, at least five matched candidates per query, and an approximate PCA-frame/proper-sign-aligned point Chamfer reference. This is predicted-surface geometry, not biological ground truth; near-symmetric objects can have ambiguous PCA frames. The synthetic suite separately tests six surface families with varying protrusions/elongation, equal physical volume, rotations and resampling. Neither a pretty PCA plot nor good reconstruction alone counts as evidence of meaningful biological information.

The run report records the outcome without promoting an unsuccessful model. If the gate fails, the rotation-invariant/SDF stage remains deferred. If it passes, the next architecture should follow the equivariant-vector/invariant-latent/pose-aware SDF pattern: shape and pose must be separated, SDF queries should mix near-surface and volumetric samples, and the sign convention must be explicit (negative inside). Fine protrusions, resampling and rotation stability still need direct evaluation. The current implementation does not claim an SDF model exists merely because this roadmap is recorded.

## First completed run

The 40-epoch MPS baseline selected epoch 39. On 140 validation cells in 11 similarity groups, squared Chamfer was 0.01637 versus 0.58551 for the untrained network and 0.04389 for the training-surface template (62.7% below the template). However, strict resampling identity retrieval was only 49.3%, rotated identity retrieval was 5.0%, and the effective embedding rank was 2.29 despite its 128 output coordinates. Nearly identical cells can make strict identity retrieval difficult; these results are diagnostics rather than class accuracy.

For 105 descriptor-matched queries, the mean group-level geometry-cost improvement over classical morphometry was 0.00058, with a 95% group-bootstrap interval of −0.00075 to 0.00177. That does not establish complementary shape retrieval. Synthetic rotated/resampled identity retrieval was 20.8%, versus 100% for the exact-source classical descriptor baseline. Visual examples also show a tendency toward rounded common shapes rather than faithful individual detail.

**The validation gate failed**, so no rotation-invariant/SDF model was promoted or trained. The PointNet++ baseline remains available for exploratory comparison and diagnosis. Full numerical outputs, input/prediction examples and the learning curve are in [the local run report](../results/pointnet_shape/baseline/report.md). No test metric selected this decision.

## Atlas

After publication, choose **Surface PointNet++ · 128D** as the learned space, or compare it beside the other methods in **Compare Methods**. Point marking, tables, membership exclusions and nearest neighbors remain linked. **Compare PointNet++ reconstruction** in the cell inspector opens the original surface and decoded point cloud with linked cameras at the same physical scale. Learned predictions do not replace original measurements. Organoid-level vectors average the included QC-passing **cell-shape** vectors; they are not encodings of complete organoid arrangement.

Artifacts live in `results/pointnet_shape/data` and `results/pointnet_shape/baseline`. The run report includes training curves and input/prediction examples. Run `scripts/export_pointnet_shape.py` again after recreating the base Atlas export. Large datasets, weights and reconstructed points remain local under ignored `results/`.

## Sources

- [PointNet++ paper](https://arxiv.org/abs/1706.02413) and the [PyTorch Geometric point-cloud tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.html): hierarchical neighborhood aggregation.
- [AllenCell CytoDL representation-learning branch](https://github.com/AllenCell/cyto-dl/tree/br_release) and [benchmarking representations](https://github.com/AllenCell/benchmarking_representations): references for a later equivariant/invariant reconstruction experiment. The upstream CytoDL branch documents that Mac GPU training is unsupported, so the baseline does not adopt its dependency stack.
