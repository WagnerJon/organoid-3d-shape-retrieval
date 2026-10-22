# Organoid Atlas — native macOS experiment explorer

## Prepare raw images

Choose **New Dataset…** from the opening screen, toolbar or File menu (⌘N).
Add one or more single-channel Z/Y/X TIFF or NumPy (`.npy`) stacks, confirm each stack's voxel
spacing in µm, choose Cellpose `cpsam_v2`, `cpdino` or `cpdino-vitb`, and select
a new results folder. Existing integer label TIFFs can replace segmentation for
individual stacks. OME and ImageJ calibration are suggested when available;
missing calibration defaults to 1 × 1 × 1 µm and is labeled as a default; edit it if known.

NumPy imports accept numeric 3D arrays in Z/Y/X order, including the BlastoSPIM2
image stacks. They contain no embedded voxel spacing: Atlas starts with 1 × 1 × 1 µm; edit
the Z/Y/X values if known and confirm the axis interpretation. Slice previews read only the selected plane
through a read-only memory map. Preparation streams a calibrated TIFF cache into
the results folder for local and multi-Mac processing, retaining the original
array path and source hashes. Existing integer labels can also be NumPy arrays.

In **New Dataset → Segmentation and comparisons → Try settings on one slice**,
choose a stack, XY/XZ/YZ plane and slice. **Load raw slice** shows the image;
**Test segmentation on this slice** runs the current Cellpose settings and shows
raw microscopy alongside a cell-label overlay with opacity controls. **Show
diameter guide** adds a translucent purple disc to both previews, sized from the
Cellpose diameter in processed pixels, the processing scale and voxel spacing.
It follows the fitted image in XY/XZ/YZ and updates immediately when diameter or
scale changes. Changing calibration clears the preview so it can be reloaded
with the correct aspect ratio. **Flow threshold (slice test)** controls Cellpose's
2D mask consistency filter (default `0.4`; `0` disables it). This threshold does
not affect full-volume u-Segment3D linking, which consumes raw flows rather than
2D masks. Other Cellpose settings remain shared with full dataset preparation,
and changing settings marks the
previous test as stale. Only one test/preparation worker runs at a time; Cancel
also stops a slice test. No output folder or completed dataset is needed.

The test reads one XY plane, or streams XY pages to extract an XZ/YZ plane;
it does not load or segment the whole volume. It uses slice-local preprocessing
with calibrated in-plane scaling and standard Cellpose 2D mask formation.
It is a settings preview, not a validation of 3D linking: full-volume intensity
normalization, 3D neighborhood context and u-Segment3D aggregation may differ.
Preview labels stay in temporary storage and never replace dataset labels.

Preparation runs stacks and segmentation stages sequentially with checkpoints,
a memory safeguard, progress and cancellation. **Resume saved preparation…**
restores an interrupted job's settings; completed steps are reused. Choose a new
folder when changing inputs or settings. Raw TIFFs remain referenced at their
original paths. Results include calibrated surfaces, review flags, combined
traditional/size/shape feature plots and optional raw multi-shell harmonic
features. Detached-cell membership uses the processed segmentation grid, and
can be reviewed in **Organoid Membership**. This workflow does not train learned
encoders or automatically run every experimental descriptor.

For multiple Macs, open **Processing Macs → Nearby Macs…** to discover and pair
Atlas Workers through Bonjour. Choose this Mac, paired workers, or both; Dask
assigns whole stacks or optional plane batches with one heavy job per device, caches transferred inputs,
and collects results into one dataset. The separate **Atlas Worker for Mac**
package includes its own first-run runtime setup. See [worker setup and limits](../../docs/ATLAS_WORKERS.md).

Use **Import OME-Zarr…** in New Dataset for local OME-NGFF 0.4/0.5 directories,
including 5D TCZYX movies. Choose raw/deconvolved image, channel, pyramid level
and a zero-based time-point range/stride. Each selected frame becomes an
independent sample. Optional cell/nucleus instance labels are imported on the
same grid; lumen semantic masks and GEFF tracks are not treated as cells.
Slice tests read the selected Zarr plane directly. Preparation copies only the
selected 3D volumes into calibrated TIFFs inside the results folder, retaining
selection, original transforms and timestamps. [OME-Zarr details](../../docs/OME_ZARR_IMPORT.md).

The preparation window shows processing-grid and memory estimates and offers a
lower scale explicitly when whole-volume linking would exceed the planning
budget. Raw uint8/uint16 TIFF preprocessing streams planes; global intensity
normalization and physical spacing are retained. Changing scale uses a fresh
results folder when earlier checkpoints exist; retest the cell diameter.

Choose **3D linking → Memory-saving fine grid** to retain the selected resolution
while streaming Cellpose planes and linking through disk-backed arrays. The
standard mode remains the default. This option works with local, whole-stack
worker and distributed-plane preparation. Flow chunks and Cellpose views have
checkpoints; use **Resume saved preparation…** with the same settings. Disk
estimates and free-space checks cover cached predictions and temporary linking
files. Temporary linking arrays are removed after final labels are verified;
Cellpose predictions, final labels and provenance remain cached. Preprocessing
and exports still require RAM; this does not make an arbitrarily large grid fit.
New runs record the actual interpolation grid and its offset from raw voxel
centers so floor-rounded dimensions do not change physical calibration.

Select **Open dataset in Atlas** when preparation finishes. Processing artifacts
and logs remain beside `atlas_dataset/`; only completed exports are opened.
Requires the existing project Python environment and u-Segment3D installation.

Validation: `tests/test_dataset_preparation.py` exercises two compressed OME
stacks with existing labels through real surface/QC/descriptor export, physical
calibration, joint aggregates, detached-cell exclusion, checkpoint recovery,
single-cell export and protection against changed inputs or unrelated output
folders. Cellpose inference itself uses the existing checkpointed processing
pipeline; the workflow test does not run a new biological segmentation benchmark.


A SwiftUI/AppKit application with SceneKit rendering. Processing stays local unless you select paired LAN workers. Worker transfers use authenticated HTTPS; no cloud inference or browser shell is used. Segmentation downloads pretrained weights when needed.

## Open

Build output: `results/Organoid Atlas.app`. Double-click in Finder. The local build opens `results/explorer_dataset` automatically; use **File → Open Experiment** (⌘O) to choose a different exported dataset folder.

Requires Apple Silicon macOS 14 or later for the supplied local build. The Swift package source can also be built for Intel using a compatible toolchain. The app is ad-hoc signed for local use, not notarized for distribution.

## Explore

- Switch between **Cells** and **Organoids** in the sidebar. Organoid means an image stack in this dataset.
- Click either PCA plot or a table row. The corresponding cell turns orange in its parent reconstruction, and the raw slice moves to its centroid.
- Drag a rectangle in either plot to mark objects across both plots and the table. **Show marked only** filters to that subset. Clear marking restores the cohort.
- Filter by stack, QC, split, learned-space group, or ID. Filters also restrict similarity candidates. Click table headers to sort.
- Choose among mask SimCLR, isolated/context raw SimCLR, and isolated/context raw IConE.
- In 3D, drag to orbit, scroll to zoom, and click a surface to select its cell. **Isolate cell** removes neighboring cells from view.
- Choose XY/XZ/YZ and move the slice slider to inspect raw data and predicted labels. Enable **Slice 3D** to clip the surface at the same physical depth. Clipped surfaces have open cut faces, not filled caps.
- Click a labeled cell in a raw slice to select it. Raw previews use independent display contrast per stack, so brightness should not be treated as a quantitative cross-stack measurement.
- Compare the top traditional and learned neighbors below the table. Click a match to open two interactive 3D views, with centered cells, equal physical scale, and linked rotation/zoom. Switch off isolation to see parent-stack context.
- Export visible or marked rows as CSV with ⇧⌘E.
- Click the **camera** on any plot to open a rendering preview. Hide axes, grid, title, legend or selection highlights; edit axis labels and colors; choose light, dark, custom or transparent backgrounds. Copy the image or save a PNG at 1200–4800 pixels wide. The export uses that plot’s current cohort, palette and loaded cell surfaces. Full-screen 3D exports preserve the current camera angle. These rendering choices leave the interactive plot and analysis unchanged.


## Compare learning methods

The **Compare Methods** tab places traditional morphometry and all five self-supervised representations in a 3×2 PCA grid for either cells or stacks.

- Turn on **Unify plot colors** to choose one feature and palette for every method. The shared range uses all filtered cells or organoids, including objects missing from some methods. Expanded windows and PNG exports use the same colors. Turn it off to restore each plot’s earlier settings. Cell and organoid choices remain separate.
- Each plot has independent **Color by** and **Palette** controls for all numeric measurements present in the dataset, including custom fields and organoid cell counts. Available palettes include Viridis, Magma, Inferno, Cividis, Turbo, blue–white–red and grayscale; Atlas group colors remain available in group mode. Feature legends share the same filtered-cohort range across plots. Missing values are gray and constant features use the palette midpoint. Settings follow the same method into full-screen/3D and reconstructed-cell views; changing a 3D palette preserves the camera. Coloring changes display only, with cell and organoid settings kept separately for the session.
- Select a **Reference** method. **Reference groups** applies that method’s cluster colors to the same objects in every panel; **Each method’s groups** shows independent cluster assignments.
- Click any plot to link the selected object across all six spaces. Drag a rectangle to mark a set everywhere. The sidebar filters, QC toggle and marked-only view apply to every panel.
- **Mark nearest 10** highlights the selected object and one method’s full-vector nearest neighbors in every plot. **Inspect in 3D** returns to the reconstruction explorer with the selection intact.
- Each panel reports **adjusted Rand index (ARI)** against the reference on the visible objects shared by both methods. ARI is invariant to arbitrary group numbers: 1 means identical partitions, 0 is chance-level agreement, and negative values are possible. Fewer than two shared objects returns unavailable.
- **Query neighbors** reports overlap of the nearest ten full-vector neighbors with the reference, using the same candidate pool. The denominator decreases when fewer candidates exist. The **Other stacks only** toggle affects neighbors, not ARI. Empty candidate pools report unavailable.

PCA and cluster fits remain fixed when filtering. ARI and neighborhood scores update for the filtered cohort and selection. Agreement indicates consistency across representations, not biological truth. PCA axes can differ in orientation, scale and retained variance between methods.

## Scientific interpretation

Traditional vectors contain nine globally standardized morphometry features: volume, surface area, sphericity, elongation, flatness, solidity, and three principal-axis lengths. Similarity uses Euclidean distance on the full standardized vector. Learned similarity uses cosine distance on the full embeddings. Distances in different methods are not calibrated to one another.

PCA and six-group k-means are descriptive fits to the full experiment cohort, including held-out objects. They are not evaluation models, biological labels, or evidence of generalization. PCA axes, variance and groups are separate for each representation and level. Brushing links object identities across these independently fitted views.

Stack representations average QC-passing cell vectors, then normalize learned means. This provides a cell-distribution summary, **not** a direct whole-organoid spatial arrangement encoder. Stack volume is the sum of all predicted cell volumes; displayed shape measures are QC-passing cell means. Raw shape-retrieval/boundary/shape-metric teacher–student experiments are not silently substituted for self-supervised embeddings.

The physical spacing is inferred from dataset literature rather than verified TIFF metadata. Biological specimen identities are unknown; adjacent stacks can be related observations. Quality flags concern predicted reconstructions. These limitations are also available under **Methods & provenance** in the app.

## Rebuild

From the repository root:

```sh
.venv/bin/python scripts/export_explorer.py
native/OrganoidExplorer/build_app.sh
open "results/Organoid Atlas.app"
```

The exporter requires the existing Python environment and experiment outputs. It reads the reconstruction manifest, measured cells, embedding CSVs joined by `cell_id`, original PLY surfaces, predicted label TIFFs, and source microscopy paths in `complete.json`. Missing raw microscopy is explicitly shown as unavailable; missing required surfaces/labels fail the export. Data with only partial embedding coverage is shown only where that representation exists.

The bundle build uses `swiftc` directly because the local Command Line Tools installation has incompatible SwiftPM manifest modules and duplicate SwiftBridging definitions. `toolchain.sh` selects the installed macOS 15.5 SDK when available and applies a build-local virtual filesystem overlay to hide the obsolete duplicate module map. No system files are changed. Override the SDK with `EXPLORER_SDK`. The app requires no third-party Swift dependencies. `Package.swift` is included for standard SwiftPM/Xcode environments:

```sh
swift build --package-path native/OrganoidExplorer
EXPLORER_DATASET="$PWD/results/explorer_dataset" swift test --package-path native/OrganoidExplorer
```

`Tests/OrganoidExplorerTests` checks vector metrics, axis/index mapping, truncated meshes, and optional integration integrity for all 1,088 real meshes and 108 raw/label previews. `run_checks.sh` runs equivalent checks with the direct compiler on this Mac.

## Portable export format (v1)

`dataset.json` stores cell/stack IDs, QC, scalar measures, parent associations, PCA points, groups, full representation vectors and source provenance. Each stack folder contains:

- `{label}.mesh`: little-endian UInt32 vertex and triangle counts; packed float32 XYZ vertices; packed float32 XYZ normals; packed UInt32 triangle indices. Geometry stays at original physical XYZ positions with no decimation or smoothing.
- `labels.u16`: little-endian UInt16 label volume in C-order ZYX.
- `raw.u8`: display-normalized raw preview on the same ZYX grid, when available.

Preview arrays are downsampled to at most 144 voxels per axis. Metadata records XYZ physical extent in ZYX order. Full-resolution source paths are retained in the manifest; the application currently displays the prepared previews, not lossless full-resolution volume streaming.

The generated dataset is about 1.3 GB and lives under the ignored `results/` tree alongside the generated app. It is not copied into the source repository or published.

## Validation on this Mac

- Release bundle compiled and ad-hoc signed successfully.
- `scripts/validate_explorer.py`: all 1,088 original meshes, 108 previews, source morphology values, and all five embedding families validated by cell ID.
- `run_checks.sh`: full-vector distances, orthogonal slice indexing and image generation, all 1,088 native mesh loads, all six representation associations, and corrupt-file rejection passed.
- Live app check: cell/stack views rendered; changing a cell table row updated selection, metrics and raw slice depth; nearest-neighbor comparison opened both real meshes at a common scale.
- Raw previews are display-resolution data. 3D clip faces remain uncapped. Stack similarity is based on aggregate cell features, not a learned spatial arrangement model.

Comparison validation: Swift ARI and nearest-neighbor overlap agree with scikit-learn/NumPy for all six methods on the 985 QC-passing cells. Native edge-case tests cover relabeled clusters, negative agreement, identical degenerate partitions, mismatched/insufficient coverage, reference self-agreement and empty candidate pools.

Live comparison checks passed: switching the reference updated colors and agreement; marking ten neighbors marked eleven linked objects including the query; marked-only filtering reduced all panels to the same eleven objects and recomputed scores. The sidebar scrolls on smaller windows to keep the top controls visible.

## Segmentation lab

**Cellpose models and A/B comparison:** each Cellpose panel offers `cpsam_v2` (the previous default), `cpdino` and `cpdino-vitb`. Select **Cellpose model** in **Compare Cellpose with** to show independent **Cellpose A** and **Cellpose B** controls. Choose a model and scale/diameter/background/GPU settings for each, then run both or run only one side. Both methods use the same XY/XZ/YZ inference and u-Segment3D aggregation, restore labels to the original raw grid, and share slice controls, instance matching, disagreement overlays and agreement metrics. Existing runs can be selected separately on either side; the selected A/B runs are remembered per source. Switching to PanSeg or FOCUS-3D preserves A and restores the relevant B run when switching back.

For CellposeDINO, install the official backbone with `zsh scripts/setup_cellpose_dino.sh`. It pins DINOv3 commit `6876159a11b4df116f30f667f8c9888617df0751` and installs it without replacing existing Torch/Cellpose dependencies or adding unrelated training requirements. Cellpose downloads the selected model's weights on first use into `.cache/cellpose`. DINO uses its default 384-pixel tiles; SAM uses 256-pixel tiles. Run manifests store the chosen model, side, parameters, dependency versions and checkpoint SHA-256. Earlier manifests without explicit model/side fields remain usable as the original `cpsam_v2` runs.

Validation: both official DINO models completed real MPS inference on the 378 × 401 middle plane of X001, producing finite probability/flow fields of the expected shape. Backend checks cover dispatch of all three model choices through the reconstruction configuration, model/side provenance, invalid-model rejection and same-grid comparisons. Native checks cover independent A/B settings and selected runs, run-title model labels, switching away and restoring B while preserving A, and imported reconstruction settings. These are inference/integration checks, not a 3D segmentation accuracy benchmark.

The **Segmentation** tab opens the original raw TIFF for an experiment stack, or a new single-channel 3D TIFF in ZYX order. Verify the Z/Y/X voxel calibration before running. Applying a new calibration reloads the source and selects only runs with that calibration.

- **Cellpose-SAM + u-Segment3D** uses the same image-only preprocessing, three orthogonal Cellpose passes and aggregation as `scripts/reconstruct.py`. Scale and diameter refer to its processed isotropic volume. A small scale is useful for a first trial; full runs can take minutes. **Load existing** imports the experiment’s `labels_native.tif` without rerunning inference, retaining the original settings.
- **PanSeg** runs the official pretrained 3D U-Net, distance-transform watershed and optional multicut. The confocal and light-sheet models are plant-trained, so results on organoids are exploratory. The worker checks 3D convolution support before choosing Apple GPU acceleration and falls back to CPU when unavailable. The setup script installs PyPI PyTorch 2.14, which supports MPS Conv3d on this Mac; conda’s 2.13 build did not. An optional normalized raw-intensity foreground cutoff excludes background after segmentation; 0 disables it. This is image-derived and never uses another method’s masks or annotations.
- Running jobs show a native progress bar with the current stage and percentage. Cellpose inference advances by completed slices in each orthogonal pass; PanSeg advances by completed patches. Preprocessing, aggregation, watershed, multicut and final output stages are labeled. This is stage-based progress, not a time estimate.
- **Run both methods** runs sequentially. Each run has its own immutable request, native TIFF labels, uint32 preview, instance volumes, timestamps and provenance under `results/segmentation_explorer/runs/`. First use downloads PanSeg weights; image data remains local. **Cancel** stops the worker process group; incomplete runs are excluded from history.
- The raw and both overlays share XY/XZ/YZ, slice and brightness controls. Click an instance or its volume-table row to highlight its overlap partner; table clicks center the slice. Dashed squares flag cells touching the stack border. Run selectors let you compare different parameter trials.
- Agreement is computed on the full native voxel grid. Foreground Dice measures occupied-volume overlap; cell matches use one-to-one IoU matching at 0.5, independent of instance label numbers. Scores describe agreement between methods, not accuracy against annotations. The disagreement overlay distinguishes foreground differences and overlapping instances without valid matches.
- Display previews have at most 144 voxels along the longest axis. They do not enter inference or agreement calculations. Comparisons require the same raw SHA-256, native dimensions and calibration; changed raw or label files are rejected.

The development bundle requires this project’s `.venv` and PanSeg environment. Install PanSeg with `scripts/setup_panseg.sh` (Miniforge/conda required). It pins the upstream code commit and installs dependencies in `.cache/panseg-env`, separately from Cellpose/u-Segment3D. Full dependency versions can be exported with `conda list -p .cache/panseg-env --explicit`.

Backend checks: `.venv/bin/python -m unittest discover -s tests -p test_segmentation_explorer.py` covers sparse/permuted labels, splits, empty/disjoint masks, color/time/channel and 2D input rejection, native-grid calibration, file integrity, physical volumes and preview output.

Validation of the segmentation extension: both image-only pipelines ran on the original X001 stack (70 × 378 × 401). The original full Cellpose result and PanSeg each contained six retained cells; native-grid matching found four pairs at IoU ≥ 0.5. PanSeg was tested with the confocal model, multicut, minimum size 100 and raw foreground cutoff 0.05. Fresh Cellpose inference was also tested at scale 0.15. These are engineering checks, not an accuracy benchmark. Python comparison/integrity checks and native all-plane rendering/selection checks pass. Live visual inspection of this extension could not be performed because the computer-use connection was unavailable.

## PanSeg Try Out

The **PanSeg Try Out** tab separates pipeline tuning from the current segmentation comparison. Its controls sit beside linked raw, boundary-probability and instance-label viewers. The stack and calibration are shared with **Segmentation**.

Prediction controls include all ten bundled official 3D boundary models (including mouse embryo cells), compatible BioImage.IO U-Net IDs and custom U-Net YAML/weights; device, output channel, automatic/manual patch dimensions, model-derived/manual/tiled halos and single-patch batching. Raw preprocessing exposes independent Z/Y/X resampling, Gaussian smoothing and median filtering. **Match training voxel size** derives factors from the selected zoo model; verify that the resulting working volume is suitable for your stack and available memory.

Watershed controls expose threshold, seed/weight smoothing, alpha, working-grid minimum size, slice-by-slice segmentation, calibrated anisotropy, non-maximum seed suppression, blockwise execution and thread count. Clustering offers watershed only, multicut, GASP and mutex watershed, with beta, post minimum size and GASP linkage. Background cutoff uses raw normalized intensity and never annotation masks. This workbench targets single-channel 3D boundary-based cell segmentation; it does not reproduce napari’s unrelated multichannel, training or nuclei-assisted lifted-multicut workflows.

**Predict + segment** creates a saved trial with working-grid boundaries, a boundary preview and labels mapped back to the original raw grid. **Re-segment saved boundaries** skips neural-network inference when the source, calibration and inference controls match. Changes to model/preprocessing/patch/device/output-channel settings require a fresh prediction. The backend also verifies boundary file integrity and working-grid geometry. Anisotropy is derived from effective working-grid spacing, normalized to its smallest axis. Minimum voxel sizes refer to that working grid.

Trial history retains the displayed run’s original settings. Edited controls do not alter it or the comparison run. **Transfer displayed run + settings to comparison** selects the exact saved labels and their executable settings in **Segmentation**, preserving the existing Cellpose run and recomputing native-grid agreement. It does not run inference again. Older imported runs without recorded prediction settings remain inspectable but require a new trial before this transfer workflow. Presets can be named, saved locally and imported/exported as JSON; **Restore this run’s settings** fills the controls from the displayed trial.

Validation: a real X001 trial with resampling, Gaussian smoothing and multicut completed, followed by GASP and slice-wise mutex-watershed trials reusing the saved boundaries without neural-network inference. The slice-wise trial also exercised non-maximum seed suppression and calibrated 2D spacing. Python checks cover cache compatibility/integrity and all clustering dispatches. Native checks cover trial isolation, exact-run transfer despite unsaved edits, preset round trips, reuse eligibility and boundary/label rendering.

## Optional membrane-guided refinement

Open the dedicated **Post Processing** tab (also linked from **Segmentation**). It keeps its source, settings, before/after selections and processing state separate from the model comparison. **Use segmentation stack** adopts the currently prepared raw stack and exact calibration; alternatively choose an experiment stack or open a raw TIFF. Calibration can be inspected/edited here; applying a different calibration requires matching saved runs or re-importing labels. **Import original labels…** labels otherwise unidentified external TIFFs as imported, rather than assigning a model name.

Choose **Before** from saved original Cellpose, PanSeg, FOCUS-3D or imported native-grid results, then select an existing **After** trial or click **Run post processing**. This runs the secondary optimizer without rerunning the original model or overwriting its outputs. The after selector shows only children of the selected original. **Restore trial settings** fills the controls from the displayed saved trial; edited controls affect the next run and do not alter either displayed result. Switching tabs retains workbench state without replacing the segmentation comparison. Refining an already refined result is deliberately blocked; select its original parent instead.

Large **Before/After** panels share calibrated XY/XZ/YZ planes, slice, brightness and overlay/contour controls. Cell selections are linked by preserved instance ID. Raw membrane and **Changed voxels** panels show added foreground (green), removed foreground (magenta), and reassigned identity (amber), independent of overlap-matching thresholds. A clickable per-cell table shows original/refined physical volumes and percentage changes; shield icons identify protected cells. Summary metrics, exact saved settings, output-folder access and a run log support review. The left settings pane and right inspection pane scroll independently; run/cancel and history controls stay visible.

**Shape-preserving refinement v2** replaces the original scalar distance-field offsets, which could carve holes and distort otherwise natural cells. A coarse physical vector field warps each original signed-distance shape. Displacement magnitude and its spatial derivatives are bounded to prevent folding the continuous warp; discrete topology is checked separately. Soft membrane shells are rendered with an anisotropic Gaussian PSF and fitted to acquired image planes. The objective includes initial-shape and volume consistency, smooth displacement, surface area and cell exclusion. An optional ellipsoid prior is disabled by default; no learned normal-cell prior is used. It is acceptable for an already good segmentation to remain unchanged.

**Conservative defaults** resets to 30 iterations, 0.5 µm maximum movement, 2 µm controls, 0.5 shape/smoothness weights, no ellipsoid prior, learning rate 0.03, 10% maximum volume change, 10% changed voxels relative to the original cell, and at most 5% increase in volume-normalized surface area. New cavities, handles and components are rejected by default. Cells failing these guards revert to their original region; diagnostics store original/final topology and surface roughness. These are preservation limits, not evidence that accepted changes are biologically correct. Old trials are labeled **Legacy refinement** and kept intact. Rerun from the original with Conservative defaults; restoring legacy settings does not reproduce the removed v1 algorithm.

Controls expose iteration count, membrane/shape/smoothness/ellipsoid weights, maximum movement, control-lattice spacing, Gaussian PSF σ in Z/XY, learning rate, maximum relative volume change and Auto/MPS/CPU execution. PSF defaults are estimates, not a measurement of your microscope. This step assumes membrane-stained raw data; it is not appropriate for an arbitrary nuclear/cytoplasmic channel. No unacquired Z planes are treated as observations. Instance boundaries move locally in physical units; labels remain on the original voxel grid. **This does not create missing cells, merge/split identities, or solve incorrect inter-slice linking.**

The default connectivity guard keeps the portions connected to each original component's interior anchor. Spurious detached portions containing at most 1% of the original cell's voxels are pruned; substantial new fragmentation restores that cell's original region. Protection is repeated because restoring one cell can affect a neighbor. Diagnostics record protected cells, reasons and pruned voxels. An interactive 32M cell-voxel limit avoids unbounded allocation. Constant-intensity images and empty/mismatched/modified parent labels are rejected. Cancel uses the existing worker-process-group cancellation.

Each completed run saves immutable parent/request JSON, original-grid TIFF labels, previews, recomputed physical volumes, optimization history, refinement diagnostics and a cropped **continuous fitted membrane** TIFF with ROI origin/calibration. That TIFF is a modeled diagnostic before discretization and safety guards, not acquired microscopy. Source/parent label hashes are checked before and after processing; the new run is published only after completion. Optimization loss can fall even when segmentation accuracy worsens, and safety guards may roll back fitted cells; the displayed loss refers to the continuous optimization before those guards. Original/refined IoU/Dice measures agreement, not accuracy.

Validation includes explicit cavity/tunnel/roughness regression tests, an irregular anisotropic cell with misleading internal intensity, and a known overgrown-sphere correction using a deliberately wider movement budget. Tests also cover sparse IDs, bounded motion, immutable inputs, volume protection, parent integrity and saved provenance. On the original X009 segmentation underlying the reported poor trial, v2 conservative defaults completed on MPS with 55 changed voxels and no new topology defects or material roughness changes. The earlier trial had changed 27,180 voxels and introduced topology defects in several cells. This establishes a much more conservative behavior, not improved biological accuracy. Native checks cover exact pairing, independent tab state, identity selection and change overlays. `native/OrganoidExplorer/run_checks.sh --worker` exercises a real two-iteration X001 job through the native launcher.

## FOCUS-3D comparison

In **Segmentation**, select **FOCUS-3D** beside **PanSeg** to compare it with the existing Cellpose-SAM + u-Segment3D pipeline. Raw microscopy, both label overlays, disagreement, cell tables and native-grid IoU matching use the same source stack and calibration. Switching methods restores each method's saved comparison run. Transferring a PanSeg trial switches back to PanSeg without replacing the Cellpose result.

Install the headless backend with `zsh scripts/setup_focus3d.sh`. It creates `.cache/focus3d-env` independently of Cellpose and PanSeg, with PyTorch 2.14.0, and pins upstream source commit `033017f20234f33722baca7b94679283e8fa6d47`. It uses upstream's cross-platform no-Detectron2 inference implementation; no napari or Detectron2 installation is needed. Source and weights are not redistributed in the app.

The official pretrained checkpoints are access-controlled at [Qinghua-thu/FOCUS-3D](https://huggingface.co/Qinghua-thu/FOCUS-3D). Request access there, download the checkpoint appropriate for your imaging target (general, membrane or nuclei), and use **Choose checkpoint…**. That path is remembered locally. No Hugging Face token is stored by Atlas. **Import labels…** also accepts externally generated FOCUS-3D TIFF instance labels on the same original grid.

Controls expose approximate cell radius in original XY pixels, automatic/raw-intensity background cutoff, minimum/maximum instance size, normalization percentiles, CPU/auto/CUDA device, batch size, overlapping patch strides, score/mask thresholds, minimum edge area and stitching thresholds. The official model patch stays fixed at 32 × 96 × 96 (ZYX); strides must be smaller than the corresponding patch dimension. An optional YAML picker supports compatible custom checkpoint/config pairs. Missing learned checkpoint parameters are rejected rather than silently using partial initialization.

Z-to-XY anisotropy comes from the shared physical calibration. Upstream assumes square XY pixels; unequal X/Y spacing is rejected rather than silently miscalibrated. Cell-radius normalization changes the inference grid, and upstream restores labels to the original raw shape using nearest-neighbor interpolation before publication or comparison. Size filters refer to the resampled inference grid.

**Apple GPU:** under **Advanced inference & stitching → Device**, choose **Apple GPU (MPS)** or **Auto (GPU preferred)**. Auto selects CUDA, then Apple MPS, then CPU. The Atlas adapter replaces upstream's CUDA/CPU-only device selector without changing the model architecture or inference mathematics. Explicit unavailable GPU requests fail clearly. Model inference uses float32 on MPS; CPU preprocessing, instance stitching and comparison remain unchanged. Unsupported PyTorch MPS operations may use CPU fallback, enabled by the native worker and recorded in provenance. Batch size 1 limits activation memory. GPU speed depends on workload; the first small-patch benchmark was faster on CPU, so acceleration is not guaranteed. Older saved runs retain their recorded device setting; change Device explicitly to test MPS.

Progress callbacks feed the native progress bar, including completed-patch counts and stitching messages. Cancel uses the existing worker-process-group cancellation. Each run records original controls, resolved options, checkpoint/config SHA-256, upstream commit, adapter/worker hashes, dependency versions and upstream logs. Incomplete or invalid-grid outputs are not published into run history.

Validation: the isolated environment imports the real upstream inference module. Backend tests cover parameter/calibration validation, device selection, incompatible checkpoint rejection, worker dispatch with controlled inference outputs, native-grid publication and Cellpose-versus-FOCUS matching, missing checkpoints, invalid output grids and progress messages. Native tests cover method switching, saved run/settings restoration and progress. The downloaded official membrane checkpoint completed real MPS and CPU inference on a 32 × 96 × 96 crop of X001. At default thresholds both retained no cells; agreement on those empty masks does not establish meaningful segmentation equivalence. A second test with score threshold 0.1 retained two instances on both devices, with foreground Dice 1.0 and both instance IoUs 1.0. End-to-end times were approximately 37.4 s (MPS) and 15.2 s (CPU), including checkpoint loading/hashing, inference and publication. This demonstrates GPU execution and parity on one crop, not a speedup or whole-stack accuracy benchmark. `scripts/check_focus3d_mps.py` compares CPU/MPS convolution, volumetric sampling/interpolation and GroupNorm outputs; `scripts/benchmark_focus3d_mps.py` runs the pretrained crop benchmark and saves timings/manifests under `.cache/focus3d-mps-validation-*`.


### Main organoid membership

The **Organoid Membership** tab reviews the cells included in exploration analyses. Automatic detection uses the largest 26-connected foreground component of each native-resolution predicted label volume, with a cell included when a strict majority of its voxels belongs to that component. This identifies detached tissue without changing the segmentation, filling gaps, smoothing meshes or imposing shape priors. Connectivity is a geometric heuristic: separated genuine tissue or two similarly sized organoids need review. Each stack is treated as one main organoid, not as verified independent biological samples.

Excluded cells remain inspectable and are red in the 3D reconstruction and raw-label overlay. Select a cell in this tab or in the reconstruction and choose **Include cell**, **Exclude cell** or **Use automatic decision**. Decisions persist in `dataset.json`. The included cohort is used for traditional feature normalization, cell PCA and k-means, method agreement, nearest neighbors, marked measurements, CSV row exports and organoid aggregates. Changing membership refits all descriptive projections and clusters; ordinary display/QC filters retain those fits. Learned embeddings are retained without retraining their models. Organoid volumes and counts sum included cells; shape means and representation averages use included QC-passing cells. Stacks without any included cells are omitted from organoid analyses but retained for review and restoration.

Exporting an experiment now detects membership automatically. An existing export can be updated with `.venv/bin/python scripts/organoid_membership.py --dataset results/explorer_dataset`. Initial classification requires its native `sourceLabels` files; saved decisions and previews remain usable without those files. The native app requires the project Python environment to refit membership edits. Failed updates leave the last complete dataset intact.

Segmentation, PanSeg and Post Processing retain complete prediction masks and mark detached cells red. Their measurement tables and comparison statistics use main-organoid cells, detected independently for each run; saved runs are classified when opening their raw stack. Comparison notes identify this scope, and original native-grid masks and hashes remain unchanged. Workbench membership is automatic per prediction; manual Atlas decisions curate the exported representation dataset. Method agreement remains agreement, not accuracy against ground truth.

Use **Hide excluded cells** below the reconstruction to hide their 3D surfaces and label overlays in both cell and organoid views. The same option is available in shape comparison. Raw microscopy remains visible and inclusion decisions are unchanged; turn the option off when reviewing excluded cells.


### Slice navigation performance

The explorer keeps up to three recently selected preview volumes in a memory-limited cache, including their label and microscopy arrays. After selecting a stack, cell, plane or display setting, a background task warms the displayed plane stack around the current slice. Cached frames include the exact brightness, selected label, overlay and exclusion settings; changing these cannot display old highlights. Frames are bounded to 32 MiB per volume, and the volume cache reserves at most 128 MiB; memory pressure can evict frames and they will be rendered again on demand. Larger custom previews warm a nearby range that fits the cache. The cache contains the existing display previews, not additional full-resolution image resampling.

Slice navigation has its own observable state so scrubbing does not rebuild the PCA plots, tables or similarity searches. SceneKit retains materials while moving through slices; with 3D slicing enabled, it updates clipping uniforms only. Reopening an experiment clears its volume cache, while switching back to a recent stack reuses its existing volume and prepared frames.

## Additional learned shape comparisons

The **Compare methods** view includes Allen VN-DGCNN, Point-MAE and the experimental
ShapeEmbedLite3D pilot, with raw/adjusted variants and dedicated control views.
The original benchmark covers 609 training/validation cells and remains cell-only.
The full-data frozen-inference extension fills the remaining 479 cells while
preserving those benchmark codes. See the [full-data guide](../../docs/ATLAS_FULL_DATA_SHAPES.md).
See [comparison details](../../docs/ATLAS_SHAPE_BENCHMARKS.md) for coverage,
interpretation and reproducible export/validation instructions.

## Full-screen plots and shape discovery

Use a plot's expansion icon or **3 PCs** button for a native full-screen plot with
linked selection and a 2D/3D switch. In cell view, **Cell shapes** replaces points
with previews of the actual reconstructed surfaces. Each plot loads only its own
visible cells, with a progress indicator and size control; **Points** releases
that plot's previews. Shapes use a normalized display size and retain linked
selection. The expanded 3D view renders rotatable cell surfaces.

**Discover Shape Groups** offers closest-shape
galleries across methods and automatic full-distance HDBSCAN groups, representative
shapes, linked cameras and group marking. See [interactive shape guide](../../docs/ATLAS_INTERACTIVE_SHAPES.md).
