# Full-data frozen shape representations

The full-data export extends the 16 Allen, Point-MAE and ShapeEmbedLite3D raw /
adjusted spaces from 609 benchmark cells to all **1,088 cells**. All **36 overview
spaces** then share every included QC-passing cell: **744** with the current
membership/QC decisions, or 765 included cells with the QC-only filter disabled.
The 323 excluded cells have descriptors for review, but remain excluded from
analysis. No source meshes, masks, microscopy, measurements, QC decisions or
membership flags are changed.

**Compare Methods → All methods** and **Discover Shape Groups → All methods**
include full-coverage methods. The exact, unquantized CAJAL reference retains its
original 48-cell scope in the separate validation view. Full-dataset CAJAL qGW
already covers all cells. Quantized and exact GW are different comparisons;
the 48-cell reference is not filled with quantized values or invented embeddings.

## What was computed

- Original 609-cell codes remain exactly unchanged.
- Additional 479 cells receive frozen inference with the same checkpoints and
  encoder/readout architectures. This includes 135 previously unencoded eligible
  benchmark-test cells and 344 excluded/QC-ineligible cells.
- Allen uses the complete 8,192-point encoder, original channel counts and exact
  chunked FP32 scheduling. One backbone pass supplies the released and adapted
  final-head norm descriptors. CPU and Metal workers may differ slightly in
  floating-point rounding; individual cache entries record the backend.
- Point-MAE supplies released/adapted native-orientation and four-sign-frame
  pose-averaged 768D descriptors from 1,024 points.
- Experimental ShapeEmbedLite3D supplies invariant-graph and literal CNN-reference
  128D latent means from 128 FPS surface samples. Reconstruction limitations remain.
- Saved descriptor scaling and degree-two morphology adjustment use the original
  fixed 96 training cells. They are applied to the new rows without refitting.

New geometry pools are area-sampled from the original physical triangle meshes,
using bounding-box center / maximum-side normalization. Seeds use a fixed offset
from the original full-manifest row, independently of cell measurements or output
quality. New inference adds no jitter or random rotations. Original codes retain
their historical input views, including the fixed noisy training views originally
used by Allen. These choices are recorded in the full-data protocol.

This is exploration with frozen models, not retraining or an additional held-out
performance claim. Original training/validation/test group assignments and offline
benchmark artifacts remain unchanged. Atlas retains its original sidebar split
assignments, which differ from the shape benchmark's grouped split. New methods
remain cell-only; incomplete cell subsets are not averaged into organoid codes.

## Artifacts and checks

`results/atlas_full_shape` contains the full manifest, additional-cell surface
pools, raw output caches, merged vectors, frozen-source hashes and completion
record. `spot_checks.json` independently recomputes new Lite3D/Point-MAE outputs on
CPU and checks saved Allen features through independently loaded frozen heads.
The expected CPU/Metal relative differences are checked below 0.001; observed
spot-check errors are below one millionth. Original benchmark-vector equality is
checked exactly before publishing, and every new adjusted vector is checked
against its raw source and saved adjustment fit.

The exporter validates source mesh hashes and nuisance measurements, rebuilds
descriptive PCA/groups using current membership, then publishes atomically with a
hash-named backup in `results/atlas_shape_benchmarks`. All earlier methods remain.
Interactive projection/discovery caches invalidate automatically when the dataset
is replaced. Native checks verify coverage of excluded review cells and that every
eligible cell is shared across the overview methods.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/extend_atlas_shapes.py --stage all --device mps --cpu-workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_full_shape_inference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_atlas_shape_benchmarks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_explorer.py
zsh native/OrganoidExplorer/run_checks.sh --shapes
zsh native/OrganoidExplorer/run_checks.sh
zsh native/OrganoidExplorer/build_app.sh
```

Inference is resumable. The original benchmark directories are never inference
output targets. Full-data vectors are published only when every required code is
finite, identities match and all frozen-source hashes still agree.
