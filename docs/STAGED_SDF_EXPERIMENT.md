# Staged SDF follow-up

The preceding small PointNet++/SDF and VN/SDF pilot failed to preserve individual
geometry. A decoder-only twelve-cell diagnostic did fit much more detailed
surfaces. This follow-up separates representation capacity, code inference and
encoder learning. Nothing is exported to Organoid Atlas.

## Fixed data and comparison

Reuse the same 96 training cells (21 TIFF-similarity groups) and 48 validation
cells (11 other groups). Test cells are never selected. Similarity groups are
proxies; biological specimen identities are unknown. Only included, QC-passing
cells enter the pilot. Original triangle meshes and their existing inferred
physical calibration remain the reference, not manually verified ground truth.

New encoder inputs have 2,048 area-sampled surface points and cached geometric
k=20 neighbors. Two independent views are available for training; a third is
reserved for the sampling-stability check. The preceding pilot used 512 points.
Changing architecture, input density and training objective together means this
is a candidate pipeline comparison, not a clean single-factor ablation.

## Stage 1: shared decoder and free shape codes

Train one vector-conditioned SDF decoder and one 128×3 vector code per training
cell for 12,000 updates, batch size eight. The decoder is the same architecture as
the preceding VN/SDF pilot. SDF targets use the already audited mesh distances,
negative inside, clipped to ±0.15 normalization radius in the loss. Prediction is
not clipped. A 0.001 mean-squared-code prior discourages unconstrained codes.

Decoder weights use AdamW; codes use vector-wise Adam, with one second moment
shared across XYZ. Ordinary coordinatewise Adam would not commute with rotation.
The decoder is selected using training reconstruction only. Validation geometry
does not update its shared weights. Random initial codes are not interpreted as
biological features by themselves.

## Stage 2: can this decoder represent new cells?

Freeze all shared weights. Optimize a code initialized at zero for each validation
cell using 1,536 signed-distance observations from its known segmentation: 768
near-surface, 512 uniform-volume and 256 exact surface points. Run 500 updates.
This is a **shape-fitting method**, not encoder-only inference. It receives more
information and needs more computation than a surface encoder. The other half of
the 3,072 prepared observations is held out from fitting and used to evaluate SDF
prediction. Reconstructed surfaces are evaluated against fresh mesh samples.

The alternate observation set is disjoint and interleaves both near-surface noise
bands, so the sampling check does not accidentally change the noise distribution.
Rotation tests rotate observations jointly. Codes start at zero, and the adaptive
optimizer is rotation-equivariant up to numerical arithmetic. Code norms give
128 scalar features. This guarantees neither unique codes nor biological utility.

## Stage 3: predict codes using an encoder

The reference-style VN encoder uses a local geometric graph, vector-neuron layers,
global-context broadcasts, collated intermediate features and mean pooling. Its
code is independently implemented after inspection of the authors' released
CytoDL VN/DGCNN code. It is **not an exact reproduction**: it uses 2,048 rather than
8,192 points, smaller channels, and per-cloud scalar RMS conditioning. That
conditioning preserves vector equivariance and relative local scales; it avoids
vanishing activations observed at initialization in this deeper small model.

Train for 80 epochs on training cells only, using randomized input views and
joint SO(3) rotation of points, teacher vectors and query coordinates. The shared
decoder remains frozen. The loss is SDF reconstruction plus 0.05 normalized vector
code regression. The reference code scale is an invariant per-channel RMS over
the training codes. Validation SDF loss selects the encoder checkpoint; there is
no final independent test estimate in this exploratory development run.

Evaluate encoder-only reconstructions, followed by optional code refinement from
its prediction. Code refinement observes the known segmentation's SDF, so it is
reported separately. The decoder-only zero-start control is cached and reused
without refitting when the shared checkpoint and inference budget match.

## Metrics and interpretation

- Fresh 2,048-point equal-area samples: symmetric squared Chamfer, P95 point error,
  surface F1 at 5% of normalization radius. Sampling error is included.
- Relative volume error, closedness and crossing the 56³ extraction grid box.
- Identity retrieval under a fresh input sample; fitted methods additionally use
  disjoint SDF observations. Rotation stability is checked, not presumed.
- Centered effective rank and the previous matched-morphometry geometry proxy.
- Paired group bootstrap intervals for changes against the original point AE.

The baseline AE has more training cells and a different architecture. Its metrics
are comparisons of practical candidates, not claims of parameter-matched model
superiority. Geometry retrieval proxies cannot establish biological utility.

An additional diagnostic compares the direct-fit codes using the original norms,
norms after a fixed random 128-channel mixing, and the full upper-triangular Gram
matrix. The latter includes inter-vector angles that norms alone discard. These
readouts were checked after observing poor code repeatability, so they are
exploratory diagnostics, not preregistered model-selection results. A separate
common-query field comparison checks whether latent instability accompanies
large function changes or relatively similar reconstructed fields.

A further mathematical diagnostic uses a D2-symmetric ellipsoidal point cloud.
Permutation invariance and SO(3) equivariance force its global vector code to zero
(the code must be invariant under 180-degree rotations about two distinct axes).
The reference-style encoder passes this test. A free auto-decoder vector code is
not constrained by the input stabilizer, so a nonzero arbitrary code that decodes
such a symmetric shape can be an impossible target for an equivariant encoder.
This identifies a risk of naive vector-code regression, not a demonstrated cause
for every real cell or a criticism of the published jointly trained framework.

## Reproduce

```sh
.venv/bin/python -m unittest discover -s tests -p test_staged_sdf.py
.venv/bin/python scripts/prepare_staged_sdf.py
.venv/bin/python scripts/train_staged_sdf.py --stage decoder --steps 12000 --device mps
.venv/bin/python scripts/run_staged_sdf.py --device mps --steps 500 --epochs 80
.venv/bin/python scripts/diagnose_staged_codes.py
.venv/bin/python scripts/report_staged_sdf.py
```

The Mac GPU requires Metal access outside the sandbox. Existing completed outputs
are not overwritten. Files are written only under `results/staged_sdf` and logs
under `.cache`. Manifest, point-array, decoder and Atlas fingerprints are checked.

## Completed result (2026-10-18)

The shared decoder completed 12,000 updates on 96 training cells, with training
SDF L1 0.00475. The reference-style encoder completed 80 epochs; validation
selected epoch 40. On 48 other-group validation cells:

| Method | Squared surface Chamfer | Surface F1 | Independent-observation identity |
|---|---:|---:|---:|
| Existing point AE | 0.01743 | 31.7% | See prior baseline stability check |
| Staged encoder only | 0.03356 | 24.6% | 10.4% |
| Encoder plus SDF code fitting | 0.00286 | 84.2% | 35.4% |
| SDF fitting from zero, no encoder | 0.00369 | 78.6% | 14.6% |

The refined method's mean absolute relative mesh-volume error was 0.49%. All
surfaces extracted successfully, remained closed and stayed inside the extraction
box. The reconstruction improvement over the old AE was positive in a paired
11-group bootstrap interval. Fitted methods used additional known-segmentation
distance observations, so this does not establish superior encoder-only inference.

All new methods retrieved the same cell under rotation in this 48-cell pool, but
sampling stability and the matched-morphometry geometry proxy remained poor. The
refined codes' geometry-proxy advantage over classical features was negative,
with 95% group interval [-0.00587, -0.00146]. This does not establish biological
usefulness beyond volume, sphericity and elongation.

**Decision:** promising SDF reconstruction, not reliable analysis embeddings.
No Atlas integration. Test symmetry-compatible targets or decoder-field training
plus independent-sampling consistency rather than continuing naive free-vector
code regression. The Atlas dataset fingerprint remained unchanged.

Full report: `results/staged_sdf/report.md`.

Reference: [Vasan et al., Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02729-9),
[authors' code](https://github.com/AllenCell/cyto-dl/tree/br_release).
