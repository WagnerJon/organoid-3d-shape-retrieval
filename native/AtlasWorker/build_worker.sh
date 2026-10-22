#!/bin/zsh
set -eu
PROJECT_ROOT="${0:A:h:h:h}"
source "$PROJECT_ROOT/native/OrganoidExplorer/toolchain.sh"
WORKER_PACKAGE="$PROJECT_ROOT/results/Atlas Worker for Mac"
mkdir -p "$WORKER_PACKAGE/Atlas Worker.app/Contents/MacOS" "$WORKER_PACKAGE/Atlas Worker.app/Contents/Resources"
swiftc -O -whole-module-optimization "${COMPILER_FLAGS[@]}" -module-name AtlasWorker "$PROJECT_ROOT/native/OrganoidExplorer/Sources/OrganoidExplorer/AtlasWorkers.swift" "$PROJECT_ROOT/native/AtlasWorker/WorkerApp.swift" -o "$WORKER_PACKAGE/Atlas Worker.app/Contents/MacOS/AtlasWorker"
/usr/bin/python3 - "$PROJECT_ROOT" "$WORKER_PACKAGE" <<'PY'
from pathlib import Path
import plistlib,shutil,sys
root,out=map(Path,sys.argv[1:])
scripts=['atlas_ome_zarr.py','atlas_numpy.py','atlas_preprocessing.py','atlas_inference_batches.py','atlas_worker.py','atlas_worker_client.py','prepare_atlas_dataset.py','run_standalone_cellpose.py','qc_standalone_cellpose.py','export_standalone_atlas.py','atlas_streamed_linking.py','atlas_streaming_core.py','setup_atlas_worker.sh']
(out/'scripts').mkdir(exist_ok=True)
for name in scripts:shutil.copy2(root/'scripts'/name,out/'scripts'/name)
shutil.copy2(root/'requirements-lock.txt',out/'requirements-lock.txt')
for relative in ['vendor/u-segment3D/segment3D','vendor/dinov3']:
    target=out/relative
    if target.exists():shutil.rmtree(target)
    shutil.copytree(root/relative,target,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
for name in ['LICENSE','LICENSE.txt']:
    path=root/'vendor/u-segment3D'/name
    if path.exists():shutil.copy2(path,out/name)
app=out/'Atlas Worker.app'
info=dict(CFBundleExecutable='AtlasWorker',CFBundleIdentifier='org.local.OrganoidAtlas.Worker',CFBundleName='Atlas Worker',CFBundlePackageType='APPL',CFBundleShortVersionString='0.1.3',CFBundleVersion='4',LSMinimumSystemVersion='14.0',NSPrincipalClass='NSApplication',NSLocalNetworkUsageDescription='Advertise an Atlas Worker and process microscopy stacks sent by your paired Macs.',NSBonjourServices=['_atlas-worker._tcp'])
(app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
(out/'README.md').write_text('''# Atlas Worker for Apple Silicon Macs

Copy this entire folder to the Mac mini and open **Atlas Worker.app**.
Keep the app together with its scripts and vendor folders.

1. Click **Set up worker runtime…** once. Setup downloads uv/Python 3.12 and the pinned processing packages. Internet and several GB of free disk space are required. It does not change your system Python or shell settings.
2. Click **Accept jobs**. Allow local-network access if macOS asks.
3. On your MacBook open Atlas → New Dataset → Nearby Macs → Find nearby workers.
4. Compare the displayed fingerprint on both Macs and enter the worker's six-digit code. The code expires after ten minutes or one successful pairing; restart sharing for another code.
5. Select This Mac, the paired mini, or both, then prepare your dataset normally.

One stack or inference batch runs at a time on each Mac. In New Dataset, enable **Split large stacks into distributed batches** to share Cellpose XY/XZ/YZ planes. Image preparation and one global 3D linking step remain on the coordinating Mac; RAM is not pooled. Select **3D linking → Memory-saving fine grid** to reduce linking RAM using temporary files on the results drive. It retains the chosen resolution, saves progress for resume, and clears temporary linking arrays after verified completion. Preprocessing and exports still need RAM. Update this package and the main Atlas app together. Dask coordinates locally; only fixed, authenticated HTTPS jobs cross the network. Raw files are checksum-verified and cached in .cache/atlas-worker/blobs; results and checkpoints stay in .cache/atlas-worker/jobs. Returned results are collected into the preparation folder on the coordinating Mac. RAM is not pooled. NumPy image/label arrays are cached as calibrated TIFFs on the coordinating Mac before transfer; missing voxel spacing starts at editable 1 × 1 × 1 µm. The worker accepts Cellpose/u-Segment3D or supplied-label preparation, not PanSeg/FOCUS jobs.

Keep the app open and both Macs awake. Stopping sharing cancels its current processing group and retains checkpoints. If a connection is lost, resume the preparation after reconnecting; the same job reuses its checkpoints. Relocating the package after installing its runtime may require setting up the runtime again. Update both the Atlas app and this package together when pipeline versions change.

NumPy arrays and OME-Zarr selections are converted to calibrated TIFF caches on the coordinating Mac before transfer. NumPy inputs must be 3D Z/Y/X arrays; confirm their voxel spacing in New Dataset.

The worker uses pinned TLS certificates and pairing credentials. Never send your pairing code to an unverified device. The first use of a Cellpose model may download its weights.

Installer source: https://docs.astral.sh/uv/reference/installer/
''')
PY
codesign --force --deep --sign - "$WORKER_PACKAGE/Atlas Worker.app"
/usr/bin/python3 - "$WORKER_PACKAGE" "$PROJECT_ROOT/results/Atlas Worker for Mac.zip" <<'PYZIP'
from pathlib import Path
import sys,zipfile
root,out=map(Path,sys.argv[1:])
# Never distribute caches, certificates, credentials, installed runtimes or microscopy.
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['Atlas Worker.app','scripts','vendor','requirements-lock.txt','README.md','LICENSE','LICENSE.txt']:
        base=root/name
        if not base.exists():continue
        for path in ([base] if base.is_file() else sorted(base.rglob('*'))):
            if path.is_file() and not any(part in ('.git','__pycache__','.cache','.venv') for part in path.relative_to(root).parts):
                archive.write(path,Path(root.name)/path.relative_to(root))
PYZIP
print "Built $WORKER_PACKAGE"
