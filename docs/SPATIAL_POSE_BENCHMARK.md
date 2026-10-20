# Spatial encoder rotation, retrieval and held-out benchmark

This experiment remains outside Atlas. It follows the geometry-preserving spatial SDF pilot and separates three questions: pose robustness, geometry beyond basic morphometry, and held-out generalization.

## Development and test discipline

Development retains the same 96 training cells and 48 validation cells from disjoint TIFF-similarity groups. Canonical PCA uses 95 components fitted on the training cells' four proper sign-frame fields. The aligned CNN uses the existing training-only CNN as initialization, then 120 epochs on the 96 training cells with random proper sign frames, exact SDF targets and box-face supervision. Its checkpoint is selected by validation SDF L1 plus 0.25 boundary L1. No per-validation-cell code fitting occurs.

Before test geometry is opened, `freeze_spatial_benchmark.py` selects the aligned encoder with lowest validation reconstruction Chamfer among those with at least 95% exact-rotation self-identification. Retrieval does not select the encoder. The freeze stores source hashes, checkpoint/nuisance-fit hashes, test-cell identities, package versions and the selected primary comparison. The test stage refuses to run without this lock or when frozen code/artifacts have changed. It records its first opening and refuses silent restarts.

The 135 eligible test cells belong to 11 groups separate from development. Earlier unrelated project experiments evaluated these test groups. They are held out from the spatial encoders' training and selection, but must not be described as globally untouched project data. A new independent dataset is needed for fully fresh confirmation, especially biological validation.

## Pose treatment

Use the original closed cell meshes and physical normalization from the existing manifest. Canonical alignment subtracts the exact triangle-area-weighted surface centroid and uses the eigenvectors of the continuous surface covariance. This avoids bias from mesh tessellation density. Retain the original scalar normalization radius.

Principal eigenvector signs are arbitrary. Represent each cell using the four proper sign frames (determinant +1), and compare representations by the minimum squared Euclidean code distance over frame pairs. This is an orbit comparison, **not a claim of a single 95D/128D invariant embedding**: four PCA codes store 380 values and four CNN codes store 512. Residualization uses permutation-invariant mean and standard deviation across the four frames, giving 190/256 features.

Each validation and test cell receives eight random proper rotations. The benchmark regenerates distance fields directly from rotated meshes rather than rotating a cached grid, avoiding interpolation as a confound. Compare raw laboratory-frame encoders, naive single-frame canonical alignment, and the proper sign-frame orbit. Report same-cell top-1 retrieval against the complete fixed gallery and top-five neighbor Jaccard overlap with candidates from other similarity groups. Reconstruction is evaluated both at the original pose and after the first predetermined rotation.

Fresh-frame stress uses two independent 2048-point area-weighted surface samples to estimate the same surface covariance. This deliberately tests sensitivity to noisy axis estimation; production mesh-derived frames do not sample points. It is not evidence of robustness to segmentation errors. Report cells with normalized minimum eigenvalue gap below 0.05 separately, and cells with sphericity at least 0.9 when present. Symmetric objects can have no unique principal frame; four sign choices do not resolve arbitrary rotations within exactly degenerate eigenspaces.

## Similarity beyond volume, sphericity and elongation

The matching variables are log volume, sphericity and elongation, standardized using training cells only. Candidate neighbors must come from a different similarity group and lie within Euclidean distance 1 in those three standardized controls. Queries need at least five candidates; retain the closest 20 by control distance.

Compare the learned orbit distances with three-control and nine-measure classical distances. The latter adds area, flatness, solidity and axis lengths, with logarithms for dimensional size features and training-only standardization.

For a second control, fit degree-two polynomial Ridge regression (alpha 1, with intercept) from the three controls to pooled orbit features on training cells only. Apply the frozen fit to validation/test cells and retrieve using Euclidean residual-feature distance. Report remaining correlations with the controls: residualization is a finite parametric adjustment, not proof of complete statistical independence.

The independent geometry proxy uses fresh 384-point mesh samples, 24 proper axis-permutation/sign starts and eight centroid-centered rigid ICP iterations. It permits neither scaling nor reflection. It is separate from the encoder's voxel representation, but remains a geometric proxy derived from predicted segmentations, not biological or manually annotated ground truth.

Positive improvement means the encoder retrieves a lower-error geometry neighbor than the baseline. Confidence intervals resample similarity-group mean paired differences 2000 times. Development comparisons and nonprimary test comparisons are exploratory. The prespecified primary test comparison is the development-selected encoder's residual features versus the nine-measure baseline. There is one primary test contrast; no model is chosen by test scores.

## Visual checks and practical limits

Contact sheets use three queries selected by sphericity quantiles among eligible matched queries, without selecting by retrieval success. They show original source surfaces for the query and its encoder/residual/classical neighbors. They are qualitative checks, not blinded biological annotations or a quantitative human preference study.

This is one dataset without condition labels, one training seed, inferred physical calibration and uncertain biological specimen identities. TIFF-similarity groups reduce known duplication but are not guaranteed independent biological specimens. Successful geometry/rotation results do not establish biological utility, and no representation is added to Atlas by this experiment.

## Supplementary checks after the primary test

The real test cohort contains no cells with sphericity ≥0.9 (the validation cohort has one), so a separate adversarial symmetry diagnostic uses eight engineered near-spherical six-lobed meshes with isotropic covariance. It uses the frozen encoder without fitting or selection and is reported separately; its eight-shape gallery is not directly comparable to the real-cell gallery.

Visual inspection also exposed size differences permitted by the population-standard-deviation matching rule. A secondary sensitivity check residualizes the nine classical features with the same training-only polynomial Ridge procedure, then compares these adjusted classical features with the frozen PCA residual features. This was added after the primary test, is not confirmatory, and cannot change the selected model or primary result. No new model or hyperparameter is selected by this analysis.

## Run order

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/prepare_spatial_benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/train_spatial_benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/evaluate_spatial_benchmark.py --stage development
.venv/bin/python scripts/freeze_spatial_benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/evaluate_spatial_benchmark.py --stage test
.venv/bin/python scripts/report_spatial_benchmark.py
```
