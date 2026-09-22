# Validation — 2026-09-19

## Completed SAM v2 batch

The first five stacks (X001, X003, X005, X007, X009) were reconstructed with cpsam_v2 on Apple MPS, using `config/first5-cpsam-v2.json`. Predictions consumed images only; masks were loaded after reconstruction for comparison. Total processing time was 68.4 minutes. All 20 reference cells matched at IoU >= 0.5; each stack also contained two unmatched predicted objects. Mean foreground Dice was 0.801; instance F1 was 0.80 for every stack. All native label dimensions, instance counts, and corresponding mesh counts were checked. See `results/cpsam_v2_first5/REPORT.md` and `summary.csv` for per-stack details and artifact links.

## Earlier setup checks

The pilot metrics below refer to the original **Cellpose 3.1.1.3 cyto3** setup. The active configuration has since switched to **Cellpose 4.2.1.1 cpsam_v2**; these accuracy scores do not describe the new model.

SAM v2 migration verification: dependency check and all four unit tests pass. Two X001 middle slices, resized to 64×64, ran through `uSegment3D.Cellpose2D_model_auto` with `cpsam_v2`. Returned finite probabilities of shape (2,64,64) and flows of shape (2,2,64,64). CPU inference took about 100 seconds for these two slices. This verifies the model interface, not full-volume SAM v2 reconstruction accuracy; no SAM v2 batch was launched.

- Environment dependency check: passed; u-Segment3D and scikit-fmm import successfully.
- Data inventory: 108 matching image/mask volumes, 750 annotated cell instances.
- Existing annotations: all 750 cells exported as calibrated PLY meshes under `results/reference`; combined measurements in `results/reference/all_cells.csv`.
- Four unit tests passed: physical mesh coordinates/winding and instance matching for perfect, merged, and empty predictions.
- Completed-run resume behavior verified: identical configuration skips completed inference.
- X001 smoke test (scale 0.15, diameter 30): completed all three views, aggregation and exports; foreground Dice 0.485. This config is only an installation check.
- X001 pilot (default config: scale 0.5, automatic diameter, background correction): completed in 485.6 seconds on CPU.

Pilot outputs: `results/run_9faa7c24d3fa/X001/`.

At IoU >= 0.5, the pilot recovered all four reference objects, with two additional objects: precision 0.667, recall 1.0, F1 0.80, mean IoU of matched cells 0.673. Foreground Dice was 0.701. The comparison PNG was visually inspected: the four main cell boundaries are resolved, but the prediction includes excess foreground and extra objects.

This establishes a working setup, not final segmentation accuracy across the collection. Review/tune filtering and boundaries on several representative volumes before processing all 108 predictions. No full prediction batch was launched; the full reference-mask mesh export is complete. Physical measurements depend on the literature-derived voxel calibration in the config.
