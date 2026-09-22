# Reproducing the experiments

Run commands from the repository root. The recorded environment was **Python 3.12 on macOS ARM64** with Apple MPS for neural-network training. Other platforms may need different wheels and device settings. Training outputs live in ignored `results/`; monitor a run with `tail -f results/<run>/training.log`. The `run_*.py` launchers use macOS `caffeinate`; elsewhere, invoke the corresponding prepare/train/evaluate scripts directly.

## Environment and data

```bash
python3.12 -m venv .venv
git clone https://github.com/DanuserLab/u-segment3D.git vendor/u-segment3D
git -C vendor/u-segment3D checkout 33974282e436d456f4941a2c2b9a8c2d1d9ed2fb
git -C vendor/u-segment3D apply ../../config/cellpose4-dependency.patch
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install ./vendor/u-segment3D
.venv/bin/python -m pip check
```

The patch updates the upstream package's Cellpose constraint for the recorded Cellpose 4.2.1.1 setup. Upstream code and Cellpose model weights are not distributed here. Obtain the [Mouse-Organoid-Cells-CBG archive](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0) from EmbedSeg and extract `train/images/X###.tif` and `train/masks/Y###.tif` under `data/Mouse-Organoid-Cells-CBG/`, or set an environment override:

```bash
export ORGANOID_DATA_DIR=/absolute/path/to/Mouse-Organoid-Cells-CBG/train
.venv/bin/python scripts/reconstruct.py inventory
```

The recorded Z/Y/X spacing is **1.0/0.1733/0.1733 µm**, inferred from literature rather than TIFF metadata. Confirm it for your acquisition before interpreting physical size.

## Reconstruction and conventional features

```bash
.venv/bin/python scripts/reconstruct.py predict --config config/first5-cpsam-v2.json --limit 5
.venv/bin/python scripts/run_remaining.py
.venv/bin/python scripts/analyze_all.py
.venv/bin/python scripts/report_all_analysis.py
.venv/bin/python scripts/smooth_all_cells.py
```

The first command runs a five-stack pilot. The remaining-stacks runner resumes completed work. For a CPU-configured all-stack path use `.venv/bin/python scripts/reconstruct.py predict --config config/organoids.json`. Prediction uses raw images; annotations are loaded afterward for evaluation. Smoothing creates separate mesh exports and does not alter labels or CNN inputs.

## Learned representations and exploratory groups

Run after reconstruction and morphology analysis:

```bash
.venv/bin/python scripts/prepare_simclr3d.py
.venv/bin/python scripts/train_simclr3d.py
.venv/bin/python scripts/report_simclr3d.py
.venv/bin/python scripts/prepare_raw_simclr3d.py
.venv/bin/python scripts/run_raw_simclr3d.py
.venv/bin/python scripts/prepare_isolated_raw_simclr3d.py
.venv/bin/python scripts/run_isolated_raw_simclr3d.py
.venv/bin/python scripts/run_icone3d.py
.venv/bin/python scripts/run_context_icone3d.py
.venv/bin/python scripts/compare_organoid_representations.py
.venv/bin/python scripts/compare_organoid_cell_distributions.py
.venv/bin/python scripts/explore_organoid_phenotype_groups.py
.venv/bin/python scripts/overlay_cell_count_phenospace.py
```

Raw context-rich crops contain unmasked neighboring cells; isolated raw crops zero voxels outside the predicted mask. The morphology-probe validation set also selected checkpoints and is therefore diagnostic, not untouched test data.

## Shape-specialized retrieval and audit

```bash
.venv/bin/python scripts/run_shape_retrieval.py
.venv/bin/python scripts/prepare_multicell_shape_retrieval.py
.venv/bin/python scripts/run_multicell_shape_retrieval.py
.venv/bin/python scripts/reevaluate_multicell_retrieval.py
.venv/bin/python scripts/audit_shape_retrieval_split.py
.venv/bin/python scripts/report_matched_3d_retrieval.py
```

The raw student has no mask input at inference but is trained with a predicted-mask teacher and auxiliary shape targets. The matched audit fixes cell count, restricts total foreground volume to 80–125% of each query, and evaluates normalized pairwise centroid-distance arrangement among different-stack candidates. It does not directly assess cell surfaces or contact topology. The image-similarity audit flags likely related train/test fields but cannot prove specimen identity.

## Validation and public figures

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_publication_assets.py
```

The asset builder copies only analytical plots and aggregate tables into `docs/`. The committed results are a snapshot of a completed run; retraining can change them. A future independent benchmark requires specimen/acquisition identifiers and group-disjoint splits before training.
