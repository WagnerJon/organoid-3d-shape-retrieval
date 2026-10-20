# Spatial shape comparisons in Atlas

The **Compare Methods** tab offers **Shape and traditional**, **After size / shape adjustment**, and **All methods**. Reference colors follow the same cell identities. Click or brush to inspect those cells in the reconstruction explorer and tables. Neighbors use full comparison distances, not 2D projection distances.

New spaces:

- Frozen 95D spatial SDF PCA with four proper axis-sign frames. Axis gaps below 0.05 trigger a separate orientation-independent surface signature for comparisons involving either ambiguous cell. MDS displays these mixed distances; average-linkage produces exploratory groups. This hybrid is not a single rotation-invariant neural code, and need not obey the triangle inequality.
- Surface radial/pair-distance quantiles, compressed by a new 95D PCA fitted only on the original 96 training cells. This descriptor is invariant under rigid transforms of the same triangulated mesh, but loses chirality and some spatial arrangement information, and is sensitive to remeshing. It cannot uniquely identify every shape.
- Adjusted spatial orbit summaries and adjusted surface signatures. The spatial summary retains the near-symmetry alignment limitation; only the full hybrid comparison has the ambiguity fallback.
- Traditional control (volume, elongation, sphericity), size, shape, all-nine, and adjusted all-nine spaces. Positive size features are log transformed; scaling uses the same original training cells.

Degree-2 Ridge adjustment for log volume, elongation and sphericity is fitted only on training cells and applied identically to traditional and shape features. It does not prove biological independence. Display PCA and clusters are descriptive fits on included cells, and membership edits refit them. The learned reconstruction PCA checkpoint is not refit. Excluded cells remain inspectable but cannot contribute to analyses. Organoid plots average included QC cell descriptors; hybrid cell-pair distances cannot be meaningfully averaged, so that method is absent at organoid level.

`results/atlas_spatial/report.md` and `feature_comparison.png` report the development-only geometry comparison. The earlier frozen test is not rerun or reused for tuning the fallback. The engineered near-symmetry check tests identical meshes under 64 rigid rotations; it does not establish universal shape discrimination or remeshing stability. Biological usefulness still requires independent phenotype/condition labels.

Reproduce from the project environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/export_atlas_spatial.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_atlas_spatial.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/report_atlas_spatial.py
native/OrganoidExplorer/build_app.sh
native/OrganoidExplorer/run_checks.sh
```

Export verifies source geometry hashes and the frozen PCA checkpoint, saves a backup of `dataset.json`, and atomically replaces only the analysis export. Source surfaces, raw stacks, segmentation labels, physical measurements, membership decisions and existing representation vectors are preserved. The per-cell cache avoids repeating surface-distance queries on unchanged meshes. Use the original Atlas export and model files; the frozen experiment scripts and checkpoints remain unchanged.
