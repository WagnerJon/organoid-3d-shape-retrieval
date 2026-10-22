# Nearby Mac workers

Atlas can prepare a multi-stack dataset using paired Apple Silicon Macs. Dask
coordinates whole-stack or plane-batch tasks inside the preparation process; workers receive
fixed processing requests over pinned HTTPS. No Dask port or arbitrary Python
execution endpoint is exposed on the LAN.

## First setup

Build the portable worker package from the repository root after setting up the
project's dependencies and vendor checkouts (see the [pipeline setup](REPRODUCIBILITY.md)
and [Cellpose DINO setup script](../scripts/setup_cellpose_dino.sh)):

```bash
native/AtlasWorker/build_worker.sh
```

This creates `results/Atlas Worker for Mac.zip`. The package contains the app,
processing scripts and required vendor sources; images, model weights,
credentials and installed runtimes are excluded.

1. Copy and unzip `Atlas Worker for Mac.zip` on the mini. Keep the entire folder
   together; open `Atlas Worker.app` inside it.
2. Click **Set up worker runtime…** once. It installs a private Python 3.12 and
   the pinned runtime using uv's official installer. Internet and several GB of
   disk space are needed. System Python and shell profiles are unchanged.
3. Click **Accept jobs**, and allow local-network access if macOS requests it.
4. On the main Mac, open **New Dataset → Processing Macs → Nearby Macs…**.
   Click **Find nearby workers**. Compare the fingerprint with the mini's
   display, enter its six-digit code, and pair.
5. Select **This Mac**, the mini, or both. Close the worker panel and prepare
   the dataset normally. Each available Mac takes the next queued stack.

Pairing codes expire after ten minutes or one pairing. Restart sharing for a
fresh code. Keep both apps open. Sharing and distributed preparation prevent
idle system sleep while active; closing a laptop or explicitly sleeping still
interrupts it. Bonjour needs a local network that permits device-to-device
traffic; guest Wi-Fi and firewall rules can prevent discovery.

## Processing and results

Each worker admits one heavy stack job at a time. Cellpose uses its local
MPS GPU when enabled and available; u-Segment3D uses the existing sequential,
checkpointed pipeline and memory guard. Supplied-label stacks are also supported.
This scheduler does not dispatch the separate PanSeg/FOCUS comparison tabs or
slice previews, and it does not pool RAM.
Final joint feature plots and descriptors are computed by the coordinating Mac.

### Split one large stack

In **New Dataset → Processing Macs**, enable **Split large stacks into distributed
batches**. Select the paired Macs and choose 8–64 planes per batch (default 16).
Each free Mac takes the next batch from the XY, XZ or YZ inference queue.

The coordinating Mac preprocesses the entire image once, preserving the existing
normalization and physical resampling. Only the required preprocessed float32
planes are sent to each worker. Cellpose uses the same supplied diameter and
per-plane settings as the unsplit pipeline. Singleton final batches are supported.
The coordinator checks coverage and assembles all probabilities and directional
flows on disk before running **one global u-Segment3D linking step** and
export. There is no independent 3D slab segmentation or label stitching, so
batch boundaries do not truncate cells.

Image preparation and final 3D linking still require sufficient RAM on the
coordinating Mac, even when This Mac is deselected for inference. This option
speeds shared inference. Select the separate memory-saving linking option to
reduce the coordinator’s linking memory use.
Transfer and model-loading overhead can make small stacks slower; prefer larger
batches on fast networks. Temporary batch files and returned predictions need
additional disk space. Transfer-only plane files are deleted after successful
assembly. Batch results, logs, hashes and checkpoints remain under
`stacks/Stack_*/inference_batches/`; completed views are reused on resume.
Imported label stacks bypass inference as before.

NumPy image and label arrays are first streamed into calibrated TIFF caches on
the coordinating Mac. Missing calibration starts at 1 × 1 × 1 µm and remains
editable. Original NumPy paths and checksums are retained in dataset provenance.

Raw/label TIFFs are streamed to a worker cache and SHA-256 verified. Identical
inputs are sent only once. Results return as a checksum-verified archive;
archive paths are validated and local source paths restored before Atlas export.
The final dataset stays in the chosen results folder on the main Mac. Raw files
remain in their original locations. Per-stack worker provenance is recorded in
`stacks/Stack_*/worker_provenance.json`.

The worker package keeps its cache under `.cache/atlas-worker`: `blobs/` holds
input copies; `jobs/` holds checkpoints, logs and result archives. These files
remain on that worker until you clear them while sharing is stopped. The
shareable ZIP excludes all caches, credentials, runtime installs and image data.

Cancellation stops connected remote jobs and retains completed stages. A lost
connection triggers reconnection attempts; if it remains unavailable, preparation
stops with a resumable error. Resume after reconnecting to reattach to the same
job or restart from its checkpoints. Changing only the processing Macs does not
invalidate the analysis settings. Jobs on an unreachable worker may continue;
stop sharing on that Mac to cancel them locally.

Workers compare protocol, pipeline/vendor source hashes and numerical-library
versions before accepting jobs. Update both the app and worker package together.
A worker runtime installed inside its folder should be set up again if that
folder is moved, because Python environment paths can depend on its location.

## Validation

`tests/test_atlas_workers.py` runs two real TLS worker services on loopback with
Dask scheduling, pairing/pinning, input caching, one-job admission, cancellation,
retry, three-stack result collection and final Atlas export using supplied labels.
`native/OrganoidExplorer/run_checks.sh --workers` starts a worker through the
native launcher, publishes/discovers it through Bonjour, pairs it, removes its
credentials and shuts it down. These checks simulate the two Macs on one host;
A fresh packaged Python installation also completed a real MPS Cellpose/u-Segment3D run on the X001 example (scale 0.15), returned its results over HTTPS and exported two detected objects to Atlas. These are engineering checks, not segmentation-accuracy validation. Physical MacBook-to-mini Wi-Fi behavior still needs verification on your network.

`tests/test_atlas_inference_batches.py` checks noncubic XYZ probability/flow
assembly, singleton tails, complete coverage, overlap/nonfinite rejection and
single-stack dispatch. `scripts/check_atlas_inference_batches.py` compares real
MPS batch predictions routed through local Dask and a packaged TLS worker against
an existing unsplit reference. This is a loopback check, not a network speed benchmark.

A fresh X001 end-to-end run (scale 0.15, six batches of up to 32 planes) also produced exactly identical probability/flow arrays and final 3D labels to the unsplit reference and successfully exported the Atlas dataset. Reports: `results/inference-batch-validation/report.json` and `results/inference-batch-full-stack/batch_validation_report.json`.

## Large-stack memory planning and preprocessing

Before inference, Atlas estimates the physically resampled grid, the size of one
float32 image and a conservative whole-volume linking allowance (256 bytes per
processed voxel for Standard, or 48 bytes for Memory-saving fine grid, plus
1.25 GiB runtime overhead, capped by the existing process
safeguard and 65% of installed RAM). This is a planning heuristic, not a measured
peak or a guarantee; foreground occupancy and other apps affect actual memory.
The app explicitly offers a lower processing scale and keeps prior settings in
the original results folder. No scale is changed automatically. A lower scale
reduces spatial detail; test the diameter again in processed pixels.

For uint8/uint16 TIFFs, preprocessing now streams individual raw planes, computes
an exact global positive-pixel median from an integer histogram, and resamples
on one global trilinear grid into a disk-backed image. It retains global
background correction and percentile normalization; a one-plane halo preserves
the 3D median filter across processing blocks. Small floating-point differences
from the original full-volume Torch interpolation are covered by numerical
parity tests. Other input dtypes retain the original preprocessing path.

The reported 578 × 1920 × 1920 stack at scale 0.85 expands to
1611 × 1632 × 1632: one float32 volume alone is 15.98 GiB. On the user's 16 GiB
Mac, the planner suggests 0.17 (322 × 326 × 326). Streaming preprocessing of that
actual TIFF at 0.17 completed in 20.4 seconds with reported peak process RSS
1.31 GiB. This check did not run Cellpose or final 3D linking. Report:
`results/big-organoid-streamed-preprocess-validation/validation_report.json`.

`tests/test_atlas_preprocessing.py` verifies early rejection before pixel reads,
compressed TIFF streaming, exact positive medians, trilinear interpolation,
global background/normalization parity and median-filter boundaries.

## Memory-saving fine-grid linking

Choose **New Dataset → Segmentation and comparisons → 3D linking → Memory-saving
fine grid**. It retains the chosen grid and existing u-Segment3D parameters:
probabilities/flows are fused with filtering halos, foreground trajectories run
in bounded batches, and connected components are reconciled globally. This
avoids independent slab labels and coarse-grid merging. It supports the current
fixed Atlas settings; unsupported parameter changes fail explicitly.

Local Cellpose views stream one plane at a time and save progress every eight
planes. Linking saves completed fusion and eight-plane trajectory chunks. Cancel
stops the processing group; resume with unchanged inputs, settings and pipeline
versions reuses these checkpoints. Fusion and global component stages restart
if interrupted. Final labels are verified before temporary linking arrays are
removed. Cached predictions and the small linking manifest remain.

The results drive supplies scratch space (planning allowance about 24 bytes per
processed voxel, plus retained predictions/exports). Atlas checks free space
before processing and linking. Remote whole-stack jobs use the worker’s drive;
split-plane jobs link on the coordinator. The same mode is sent to workers, and
pipeline hashes prevent mixing old and new worker packages. Update and reopen
both Atlas and Atlas Worker after replacing the package.

The new implementation preserved every label voxel up to instance ID numbering
on the cached MV5 stack (186 objects) and the real large-stack crop (14 objects).
The crop also passed preparation through meshes, physical measurements, QC,
feature descriptors and Atlas export. See `results/streamed-linking-integration/`.
These checks establish numerical parity, not biological accuracy. The full
large raw stack has not been processed at high resolution. Preprocessing,
large image planes, exports and foreground coverage still constrain RAM.
