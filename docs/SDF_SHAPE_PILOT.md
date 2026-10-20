# Isolated rotation-equivariant SDF experiment

Requested experiment: test the proposed replacement and report its results **before
adding anything to Organoid Atlas**. There is intentionally no export or app hook.
All results are stored in `results/sdf_shape_pilot`.

## Reference and implementation scope

- Vasan et al. (2025), *Interpretable representation learning for 3D multi-piece
  intracellular structures using point clouds*, DOI: 10.1038/s41592-025-02729-9.
- Authors' code: https://github.com/AllenCell/cyto-dl/tree/br_release and
  https://github.com/AllenCell/benchmarking_representations.
- Vector Neurons: https://arxiv.org/abs/2104.12229.

This is an independently implemented, paper-inspired **reduced-compute pilot**,
not a reproduction or use of pretrained published weights. The authors' VN and
DGCNN sources were inspected as references, not copied/imported. Our VN layers
mix vector channels without spatial biases and use learned-direction projection
nonlinearities. Three dynamic kNN EdgeConv blocks use mean pooling; the first
includes vector cross products. Later graphs use vector-feature distances.

The encoder produces 128 three-dimensional vectors (a 384-scalar training
bottleneck). A fixed scalar gain of 10 conditions their numerical scale without
changing equivariance. Their norms form the 128D
rotation-invariant analysis features. The SDF decoder receives query–vector dot
products, vector norms, and query radius. These are jointly invariant when the
object and queries rotate together. The decoder therefore preserves the input
pose through the vectors rather than discarding pose before reconstruction.
It is a simplified invariant implicit decoder, not an exact CytoDL clone.

The PointNet++ control has a scalar 128D bottleneck and the same-sized SDF MLP
(input dimension necessarily differs). Models are **not parameter matched**.
Both receive 512 area-sampled points and predict the same SDF targets. Three
independent input samples are prepared: two for training, one held back to check
sampling stability. The original PointNet AE comparison had 2,048 input points
and 469 training cells; these differences must accompany all comparisons.

## Data and safeguards

96 training / 48 validation cells are selected from included, QC-passing cells
using group-balanced morphology diversity. The existing TIFF-similarity groups
remain separate. No test cells are selected. These groups are proxies; original
biological specimen identities are unknown. Twelve training cells from distinct
groups are used for an explicit overfit sanity check; its scores are **training
reconstruction scores**, not generalization evidence.

Targets are signed Euclidean distances to the ORIGINAL closed triangle meshes,
computed with trimesh/rtree, negative inside. Nonclosed or inconsistently wound
meshes cause a preparation error rather than an unreliable target. Existing
centering/max-radius normalization and inferred micrometer calibration are
retained. Targets comprise two Gaussian near-surface bands, uniform volume
queries, and exact zero-distance surface samples. Training target distances are
clipped to ±0.15 radius. Predictions are not clipped in the loss, avoiding zero
gradients from incorrect saturation.

The segmentation is a reference, not manually verified biological truth. The
experiment learns shape representations; it does not change segmentation or
improve its physical resolution. In particular, it cannot recover detail absent
from sparse Z measurements.

## Reproduction

Existing environment additions: `rtree==1.4.1` for exact mesh distance queries.

```sh
.venv/bin/python -m unittest discover -s tests -p test_sdf_shape.py
.venv/bin/python scripts/prepare_sdf_shape.py --workers 2
.venv/bin/python scripts/train_sdf_shape.py --kind pointnet --mode overfit --epochs 250 --device mps
.venv/bin/python scripts/train_sdf_shape.py --kind vn --mode overfit --epochs 250 --device mps
.venv/bin/python scripts/evaluate_sdf_shape.py --mode overfit --device mps
.venv/bin/python scripts/diagnose_sdf_shape.py --device mps --steps 1500
.venv/bin/python scripts/run_sdf_shape_pilot.py --stage pilot --device mps
```

Mac GPU execution needs Metal access outside the execution sandbox. CPU is also
supported. Each run refuses to overwrite an existing checkpoint. Preparation and
training record hashes of the manifest, prepared targets, source meshes and
model sources. The Atlas dataset hash is recorded before preparation and checked
again during evaluation. No Atlas/native source files are edited.

The initial 250-epoch (500-update) sanity check produced average rounded shapes.
The diagnostic fits free per-cell codes and the same decoders for 1,500 updates;
these are training fits, not encoders or generalization results. Its much better
shape reconstruction demonstrates useful decoder capacity, but the difference in
optimization and update count prevents attributing failure solely to the encoder.
The subsequent 120-epoch pilot is warm-started from the corresponding training-only
sanity-check checkpoint, with a fresh optimizer and rotations enabled. All twelve
warm-start cells belong to the 96-cell training set. The free diagnostic codes
and diagnostic decoder weights are NOT used by the pilot.

## Evaluation

SDF predictions are meshed with marching cubes on a 56³ grid, in the same input
coordinate frame. No best-fit rotation or scaling is applied to reconstruction
scores. All methods are scored against 2,048 independent original-surface samples
using symmetric squared nearest-point distance, 95th-percentile point distance,
and F1 at 5% of the normalization radius. These point-based metrics include
sampling noise; they are not exact triangle-to-triangle errors. Reconstructed
volume, closedness, and crossing the extraction box are checked for SDF models.
Fixed low/middle/high-sphericity examples and orthogonal midplane bands aid visual
inspection without selecting favorable examples after seeing model errors.

Analysis features are checked for identity retrieval under fresh surface sampling
and rigid rotation, centered effective rank, and matched-morphometry retrieval
using the existing imperfect PCA-aligned geometry proxy. Reconstruction changes
are summarized by similarity-group paired bootstrap intervals. Validation also
selects checkpoints, so this is exploratory model development, not a final test.
No result establishes biological utility without independent phenotype data.

## Completed pilot result (2026-10-18)

**Do not integrate this pilot.** Both SDF encoders still lose individual cell shape.
On the same 48 validation cells, sampled-surface squared Chamfer was 0.01747 for
the existing point AE, 0.02516 for PointNet++/SDF, and 0.03557 for VN-DGCNN/SDF.
All extracted SDF surfaces were closed and remained within the extraction box,
but that did not imply faithful morphology. VN mean absolute relative volume
error was 29.2%. Its rotation identity retrieval was 100%, while independent
resampling identity was only 4.2% (versus the existing AE's 66.7% on this pool).

The free-code decoder diagnostic on twelve training cells achieved 0.00247
Chamfer and 88.0% surface F1 with the vector-conditioned decoder, showing much
better geometry than the initial end-to-end fit. This is an optimistic training
capacity check, not a learned encoder or held-out accuracy. Its different update
budget and direct per-cell optimization prevent a clean causal attribution.

Keep SDF as a candidate reconstruction representation. A useful next experiment
is to learn shape codes with a shared decoder first, then train and validate an
encoder against those codes, alongside a fuller reference VN-DGCNN setup and a
larger input-point budget. These follow-ups have not been implemented here.

Full report, figures, per-cell scores and group intervals:
`results/sdf_shape_pilot/report.md`. The Atlas dataset SHA-256 remained unchanged.
