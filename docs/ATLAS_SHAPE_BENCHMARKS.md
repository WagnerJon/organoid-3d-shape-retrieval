# Learned shape methods in Organoid Atlas

The [full-data extension](ATLAS_FULL_DATA_SHAPES.md) adds frozen inference for the
remaining cells. The 609-cell protocol described below remains the preserved
benchmark; the full-data overview uses all 1,088 cell codes.

Open **Compare methods → Learned shape benchmarks** at cell level. This view puts
adapted Allen VN-DGCNN norms, pose-averaged adapted Point-MAE, and the invariant
ShapeEmbedLite3D pilot beside all-nine-feature morphometry, spatial PCA and
spherical harmonics. Each appears as a raw and adjusted space.

The dedicated Allen, Point-MAE and ShapeEmbedLite3D filters expose their controls:

| Family | Variants | Dimensions |
| --- | --- | --- |
| Allen VN-DGCNN | Released weights; adapted final encoder head + decoder | 512 vector norms |
| Point-MAE | Released/adapted; pose averaged/native orientation | 768 token descriptor |
| ShapeEmbedLite3D | Invariant EDM graph; literal CNN reference | 128 latent means |

There are **16 new methods**, including raw and adjusted variants. Decoder-only
Allen adaptation is omitted because its encoder codes duplicate the released
encoder. Standardized-only readouts are not separate Atlas choices. Native
Point-MAE and the Lite3D CNN are controls, not recommended invariant encoders.

Keep **Same objects across plots** enabled for a common comparison cohort. Click
or brush points to mark matching objects across plots and tables. Pick a reference
for shared group colors, ARI and nearest-neighbor overlap. **Mark nearest 10**
marks neighbors using full descriptors. **Explore this space in 3D** selects that
method in the reconstruction explorer, where similarity matches can be compared
against original segmented surfaces. The **Validation benchmarks** filter also
includes the 48-cell unquantized CAJAL reference.

## Coverage and interpretation

All new methods contain the same **609 genuine cached cell codes**: 469 benchmark
training cells and 140 benchmark validation cells. No benchmark test codes are
generated, and no values are filled in for missing cells. Atlas's existing split
assignments are preserved; its sidebar split filter can differ from the shape
benchmark split. The exact benchmark assignments are recorded under
`shapeBenchmarks.benchmarkSplits` in the dataset.

These methods are cell-only. They are hidden at organoid level rather than
averaged over an incomplete subset of an organoid. Membership edits continue to
exclude detached cells from projection/group fitting and analysis. Ordinary view
filters retain the fitted descriptive display PCA and exploratory groups.

Raw readouts use the saved Euclidean benchmark distance. Adjusted readouts reuse
the saved descriptor standardization and degree-two nuisance fit for log volume,
elongation and sphericity, fitted on the original 96 development training cells.
They are not refitted on the new 609-cell display cohort. Exact agreement with
all saved 48-cell benchmark raw/adjusted readouts is checked before publishing.

Display PCA/groups are descriptive included-cohort fits. Their apparent
separation, ARI and neighbor overlap do not establish biological discrimination.
Allen uses invariant vector norms rather than flattened equivariant codes.
Point-MAE pose handling is external and has ambiguous-frame limitations.
Lite3D is an unofficial experimental adaptation; its EDM invariance also removes
chirality. Its sparse reconstructions still underfit the actual shapes.

This integration compares embeddings and original surfaces. It does not present
Point-MAE masked patch completion as a global reconstruction or turn sparse
Lite3D point reconstructions into cell meshes.

## Rebuild and verification

From the project root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_atlas_shape_benchmarks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_explorer.py
zsh native/OrganoidExplorer/run_checks.sh
zsh native/OrganoidExplorer/build_app.sh
```

The exporter checks cell identities, source mesh hashes, nuisance measurements
and saved benchmark parity; preserves existing representations; respects current
membership; and publishes atomically with a hash-named dataset backup under
`results/atlas_shape_benchmarks`. The validator checks exact vectors, partial
coverage, cell-only scope and preservation of prior measurements and embeddings.
