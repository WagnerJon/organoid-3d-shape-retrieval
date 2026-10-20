# Spherical harmonics and CAJAL in Organoid Atlas

The **Compare Methods** tab adds five spaces:

- **Spherical harmonics · shell spectrum**: rotation-invariant powers over 32 concentric radial shells and degrees 0–12 (416 values). This is an independent implementation of the spherical-shell harmonic construction, not SPHARM-PDM or Dynamic SPHARM. Occupancy is sampled directly from the original mesh's signed distance with a soft boundary width of 0.035 normalized units. It retains concavities rather than using only the outermost radial intersection. Position and scale are removed with the exact area-weighted surface centroid and RMS surface radius. Finite angular/radial sampling produces numerical rotation error. Power spectra discard phase and chirality; they cannot uniquely characterize every shape.
- **Spherical harmonics · adjusted**: identical training-only degree-2 Ridge adjustment for log volume, elongation and sphericity to the existing shape and traditional comparisons.
- **CAJAL · quantized GW**: official CAJAL 2.0.3, 128 stratified area-sampled surface points, uniform sample weights, Euclidean within-cell distances and 32 quantization clusters. Full pairwise qGW comparisons are cached. This is the supported quantized approximation, not a certified globally optimal GW solution. It removes rigid orientation (including reflections), position and scale. It is shown as an MDS distance map with average-linkage exploratory groups, not PCA of an embedding.
- **CAJAL · adjusted landmark profile**: a vector of qGW distances to the original 96 training cells, adjusted using the same training-only nuisance procedure. Euclidean distances between these residual profiles are a separate approximation, not direct GW distances.

Spherical harmonic and landmark-profile organoid views average included QC cell vectors. Direct CAJAL distances are cell-level only: averaging packed pair-distance rows would not produce an organoid shape. Excluded cells remain available for inspection but cannot enter analyses; membership edits refit only descriptive display projections and groups. The direct CAJAL matrix covers all source cells so manual re-inclusion has valid distances.

A separate **Validation benchmarks** comparison mode includes **CAJAL · unquantized GW · validation** for all 48 validation cells. It uses a complete 48×48 distance matrix from the official unquantized solver. Shared-object filtering applies to every plot, neighbor marking and agreement calculation. The unquantized method is omitted from the default full-dataset view. Choosing All methods with shared-object filtering also restricts the displayed cohort to shared cells.

## Benchmark

Development evaluation uses the same 48 validation cells, 46 eligible queries, 11 groups, candidate pools and source-surface geometry proxy as the previous Atlas comparison. Candidates come from other similarity groups, within one training SD in standardized log volume / elongation / sphericity, with at least five candidates and at most twenty. Matching in SD units is not exact size matching. The unquantized official CAJAL solver is additionally computed for all 1,128 validation pairs and evaluated on the 403 unique matched candidate pairs at the same 128-point resolution. Solver local minima mean that this reference is not a certified global optimum either.

The previous test benchmark is unchanged. New comparisons are exploratory validation analyses; there is no new held-out claim. Geometry agreement with existing predicted surfaces is not biological ground truth. Independent condition or phenotype labels and external specimens are still needed to evaluate biological meaning.

## Sources

- Kazhdan, Funkhouser and Rusinkiewicz (2003), [Rotation Invariant Spherical Harmonic Representation of 3D Shape Descriptors](https://pixl.cs.princeton.edu/gfx/pubs/Kazhdan_2003_RIS/index.php).
- Govek et al. (2023), CAJAL, [official implementation and paper](https://github.com/CamaraLab/CAJAL), [documentation](https://cajal.readthedocs.io/en/latest/).

## Reproduce

CAJAL is installed in `.cache/cajal-env`, leaving the original project environment unchanged. Its environment can read the original `.venv` dependencies through a project-local `.pth` file; keep both environments together. A package-version list and source hashes are saved with the results. Run preparation and the official CAJAL computation before exporting:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/prepare_harmonics_cajal.py
MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .cache/cajal-env/bin/python scripts/run_cajal_comparison.py
MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .cache/cajal-env/bin/python scripts/check_cajal_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/check_harmonic_rotation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_harmonics_cajal.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_harmonics_cajal.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_explorer.py
native/OrganoidExplorer/build_app.sh
native/OrganoidExplorer/run_checks.sh
```

Source meshes are hash-checked. The exporter saves a backup and atomically replaces the analysis export. Original microscopy, labels, meshes, physical measurements, membership and existing vectors are preserved. Pairwise rows keep global identities, so filtering never changes the meaning of a distance lookup. Cached qGW results are used for native browsing without rerunning the solver.
