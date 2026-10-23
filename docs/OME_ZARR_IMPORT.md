# OME-Zarr import

Open **New Dataset → Import OME-Zarr…**. Select a local, extracted `.ome.zarr`
directory or its immediate parent (for example `001-small`). Choose:

- Image: raw or deconvolved.
- Channel: names come from OME metadata, for example `mem9` and `H2B`.
- Resolution: dimensions and level-specific physical voxel spacing are shown.
- Time points: first/last indices and stride, using zero-based indices.
- Segmentation: run Cellpose or reuse the image's cell/nucleus instance labels.

Each selected T/C/level becomes one independent Atlas sample, with a distinct
name in the stack selector. Confirm calibration and test a slice before running
Cellpose; retune diameter after changing channel or resolution. The usual
memory planning, checkpoints and optional Mac workers apply.

The importer supports numeric Zarr v2/OME-NGFF 0.4 and Zarr v3/OME-NGFF 0.5
image groups with explicitly named T/C/Z/Y/X axes. It applies both level and
multiscale scale/translation transforms and converts supported spatial units to
micrometers. Slice previews read only the selected spatial plane. Preparation
streams bounded Z slabs of the selected frame/channel into calibrated OME-TIFFs
under `inputs/Stack_*/`, with at most 256 MiB decoded source chunks admitted.
Select a coarser level or rechunk larger chunks first. These cached TIFFs are
used by the existing preprocessing, worker dispatch and viewer.

When usable physical calibration is absent, the selected level defaults to an
estimated cubic-stack depth: X = Y = 1 and Z = max(width, height) / slice count.
Only Z changes; the XY field of view retains its aspect ratio. Values are
nominal µm with unknown absolute physical scale and remain editable. Known
level-specific calibration and saved preparation requests are preserved.

`source.json` retains original OME metadata, axis definitions, level, T/C,
spatial origin, and available DCA acquisition timestamps. The final dataset
also records each sample's original OME-Zarr selection. Cell coordinates remain
local to the selected volume; original physical origins are kept as provenance.
Source metadata/chunk size/mtime fingerprints detect changes; cached imports
are verified by SHA-256. Raw Zarr stores are opened read-only. Cancelled imports
leave no completion marker and can be retried with the same request.

Instance labels must match shape, voxel scale and spatial origin at the chosen
level, have matching time points or a singleton time axis, and contain integer
IDs 0–65535. Background is zero. Known lumen/epithelium semantic layers are not
offered as cell instances. Frames do not imply biological replicates, tracked
identity or lineage; GEFF graph browsing and cell tracking are outside this
import workflow. ZIP archives must first be extracted; remote URL/object stores
are not handled by this local importer.

The reader is pinned to Zarr 3.0.10 and numcodecs 0.15.1 to retain the existing
NumPy 1.26 segmentation runtime. New worker packages include the reader source;
workers receive the selected calibrated TIFFs, not the complete 5D movie.

Validation: `tests/test_atlas_ome_zarr.py` covers v2/v3, nanometer calibration,
axis ordering, T/C/level slicing, exact streamed materialization, source/cache
changes, frame identity, resume and final Atlas export. The worker test checks
that conversion precedes remote dispatch. `run_checks.sh --ome-zarr` checks real
`001-small` metadata, native selection/preview/preparation, frame labels in the
actual Atlas loader and the import window. Two real frames at level 2 with
existing cell labels exported 10 cells; this is an import/engineering check,
not a segmentation-accuracy or tracking benchmark.

References: [example dataset](https://zenodo.org/records/22078388),
[OME-NGFF 0.5](https://ngff.openmicroscopy.org/0.5/).
