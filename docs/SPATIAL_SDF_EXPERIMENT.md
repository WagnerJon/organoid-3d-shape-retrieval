# Geometry-preserving spatial encoder experiment

## Research review (18 October 2026)

The most relevant verified sources reviewed are primary papers and author repositories. This is a targeted review, not an exhaustive survey or a claim that one architecture is universally best.

| Approach | How geometry is represented | Implication for Atlas |
|---|---|---|
| [Vasan et al., Nature Methods 2025](https://www.nature.com/articles/s41592-025-02729-9) | Morphology-appropriate point inputs; rotation-equivariant encoder and SDF reconstruction for polymorphic structures; invariant vector norms for analysis | Preserve the actual surface and measure reconstruction separately from embedding usefulness. The published architecture and larger point budgets must not be equated with our reduced implementation. |
| [MorphoMIL, Cell Systems 2025](https://doi.org/10.1016/j.cels.2025.101229), [author software](https://github.com/Sentinal4D/cellshape) | Geometric learning of 3D cell and nuclear shapes, with population-level attention for treatment profiling | Shape learning needs biological evaluation beyond visually separated clusters. Author software offers [graph/folding point-cloud autoencoders](https://github.com/Sentinal4D/cellshape-cloud) and a separate [3D convolutional voxel autoencoder](https://github.com/Sentinal4D/cellshape-voxel). Our experiment is not a MorphoMIL reproduction. |
| [Wiesner et al., Medical Image Analysis 2024](https://doi.org/10.1016/j.media.2023.102991), [author code](https://github.com/MIAGroupUT/IMPLICIT-CELL-SURFACES) | Continuous SDFs with periodic MLPs and separately optimized shape codes and rotation parameters | High-quality code fitting does not establish amortized encoder performance. Preserve this distinction in our results. Spatial coordinates and code conditioning can recover complex cell surfaces without forcing shape into a global vector frame. |
| [Viana et al., Nature 2023](https://www.nature.com/articles/s41586-022-05563-7) | Aligned cell and nuclear surfaces represented by spherical harmonic coefficients, then PCA | A reconstructable linear representation is a valuable control. Their alignment retains a meaningful apical–basal axis; arbitrary rotation removal is not automatically biologically appropriate. Our PCA control uses distance grids, not spherical harmonics. |
| [O2-VAE, Nature Communications 2024](https://www.nature.com/articles/s41467-024-45362-4) | Orientation-invariant image representations with reconstruction adapted to the symmetry requirement | Orientation handling is an explicit modeling decision. Its image-profile results do not by themselves establish a 3D organoid-shape solution. |
| [CAJAL, Nature Communications 2023](https://www.nature.com/articles/s41467-023-39424-2) | Cell morphology distances based on metric geometry | Independent geometric similarity is useful as a reference for embeddings; a visually plausible reconstruction alone is insufficient. |

The author `cellshape-cloud` example uses a 128-feature DGCNN with a folding decoder. It does not imply that our much smaller global-vector model should perform equivalently. The living-cell SDF paper uses per-object optimization, which is closer to our successful SDF fitting control than our unsuccessful encoder.

## Decision and hypothesis

Our global vector bottleneck can become zero on certain symmetric shapes, and the encoder still loses shape even after removing fitted-vector regression. The new hypothesis is that retaining spatial scalar features before compression improves geometry preservation. This is an inference from our diagnostics, not a claim that the cited papers endorse this particular design.

Implement a dense signed-distance input, three spatial convolution stages, a 128-scalar bottleneck, and a coordinate-conditioned SDF decoder. The decoder uses sinusoidal coordinate features and re-injects the code/coordinate input after two hidden layers. There are no encoder-to-decoder skip connections and no per-validation-cell fitting. This is an independent hybrid design, not a paper reproduction.

Input is a 32³ grid of signed distances clipped at ±0.15 in the original normalized mesh frame. Preparation began with trimesh distances and a conservative narrow-band filter, then switched to [libigl's compiled pseudonormal signed-distance implementation](https://libigl.github.io/libigl-python-bindings/api/igl/) for speed. Completed fields were retained; the backend is recorded for each cell. The audit checks every cached grid value and all existing continuous mesh SDF targets against the compiled implementation before the validation experiment. Dense-grid preparation is cached and hashed. Training targets remain the existing exact, continuous mesh SDF query pool, not a smoothed reconstruction of the input grid.

## Protocol

1. Fit the encoder and decoder to the same 12 training-only cells used by the earlier overfit checks. This tests representational capacity before interpreting generalization.
2. Warm-start from that training-only checkpoint and train on the existing 96 training cells. Select checkpoints using the existing 48 validation cells from disjoint TIFF similarity groups. No test cells are loaded.
3. Compare original meshes, uncompressed input grids, a 95-component linear distance-grid PCA fitted on training cells only, and the nonlinear 128D encoder. With 96 training observations, centered PCA has at most 95 nonzero components.
4. Extract surfaces on the same 56³ grid and score independent 2048-point surface samples: squared Chamfer, F1 at 5% of normalization radius, physical P95 surface error, volume error, closedness and extraction-box crossings. Bootstrap paired geometry differences by validation similarity group.

**This changes the representation as well as the encoder.** Dense mesh SDFs provide more information than sampled surface points. Better reconstruction cannot be attributed solely to architecture, nor does it establish biological information beyond morphometry.

**Rotation invariance is not claimed.** Laboratory orientation is retained deliberately to isolate geometry preservation. Rotation invariance, orientation-sensitive biological axes, fresh-input perturbations and useful nearest neighbors must be evaluated separately before Atlas integration. Re-encoding the identical cached grid is not a meaningful sampling-repeatability test.

All results stay under `results/spatial_sdf`. Atlas source and its dataset remain untouched.

## Commands

```sh
.venv/bin/pip install libigl==2.6.3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/spatial_sdf.py --stage prepare
.venv/bin/python scripts/audit_spatial_sdf.py
.venv/bin/python scripts/spatial_sdf.py --stage overfit --epochs 400 --device mps
.venv/bin/python scripts/evaluate_spatial_sdf.py --mode overfit --device mps
.venv/bin/python scripts/spatial_sdf.py --stage pilot --epochs 200 --device mps
.venv/bin/python scripts/evaluate_spatial_sdf.py --mode pilot --device mps
.venv/bin/python scripts/boundary_spatial_sdf.py
.venv/bin/python scripts/evaluate_spatial_sdf.py --mode boundary --device mps
.venv/bin/python scripts/report_spatial_sdf.py
```

The initial 12-cell capacity check achieved approximately 79% surface F1, but three reconstructions crossed the extraction box. This training-only finding motivated an additional 60-epoch fine-tuning stage with exact mesh-distance targets on the six box faces. It retains the 128D bottleneck and uses no per-validation-cell fitting. The phase selects by validation SDF L1 + 0.25 × boundary L1, including epoch zero. It is a follow-up stage, not part of the original 200-epoch budget; both models' results are retained.

## Completed validation result

On the 48 development-validation cells, the final spatial CNN achieves Chamfer 0.00691 and surface F1 57.1%, versus the previous point-cloud encoder's 0.02913 and 26.5%. This is approximately 76% lower reconstruction error. The group-bootstrap interval for mean paired improvement is positive: [0.01565, 0.03244]. All final surfaces are closed and inside the extraction box; mean volume error is 4.2%.

The 95D linear spatial PCA encoder is stronger: Chamfer 0.00389, F1 76.0%, volume error 3.0% (approximately 87% lower Chamfer than the preceding encoder). It is retained as the best tested compact geometry baseline, with a saved training-only encoder. This finding supports using spatial distance-field information but does not show a nonlinear advantage or a biological advantage over morphometry.

The input grid itself scores F1 88.7%, showing the remaining finite-resolution and compression gaps. Both encoders retain laboratory orientation and are not yet suitable substitutes for a validated rotation-invariant similarity embedding. Atlas remains unchanged.
