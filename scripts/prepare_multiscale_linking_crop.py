"""Prepare one fixed membrane-rich crop for the offline multiscale experiment."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import tifffile
import run_standalone_cellpose as pipeline
from atlas_preprocessing import preprocess


def main():
    import dask
    from segment3D import parameters
    dask.config.set(num_workers=1, scheduler='threads')
    root = pipeline.ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guide-run', type=Path, required=True, help='Coarse standalone guide run used to select the fixed crop')
    parser.add_argument('--reference-run', type=Path, required=True, help='Standalone run supplying aggregation parameters')
    parser.add_argument('--output', type=Path, default=root/'results/multiscale-linking-benchmark/big-stack-crop-source')
    args = parser.parse_args()
    guide = args.guide_run
    cfg = json.loads((guide/'run.json').read_text())
    info = json.loads((guide/'processing.json').read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Fixed before seeing any segmentation: highest local intensity contrast
    # among 48^3 guide windows with 15-85% intensity foreground (see candidates).
    guide_center = np.array([200, 80, 280])
    spacing = np.array(cfg['spacingZYXum'])
    center = np.rint((guide_center+.5)*np.array(info['effectiveSpacingZYXum'])/spacing-.5).astype(int)
    lengths = np.rint(60./spacing).astype(int)
    starts = np.maximum(0, center-lengths//2)
    stops = np.minimum(cfg['nativeShape'], starts+lengths)
    crop = np.empty(tuple(stops-starts), dtype='uint16')
    with tifffile.TiffFile(cfg['source']) as tif:
        for i,z in enumerate(range(starts[0],stops[0])):
            crop[i] = tif.pages[z].asarray()[starts[1]:stops[1],starts[2]:stops[2]]
    raw = output/'raw-crop.ome.tif'
    tifffile.imwrite(raw,crop,ome=True,photometric='minisblack',metadata=dict(axes='ZYX',PhysicalSizeZ=spacing[0],PhysicalSizeY=spacing[1],PhysicalSizeX=spacing[2],PhysicalSizeZUnit='µm',PhysicalSizeYUnit='µm',PhysicalSizeXUnit='µm'))
    pre = parameters.get_preprocess_params()
    pre.update(factor=float(spacing.min()/.5),voxel_res=(spacing/spacing.min()).tolist(),do_bg_correction=True,bg_ds=8)
    details = preprocess(raw,output,pre)
    shape = np.array(details['processedShape'])
    # Torch's explicit scale_factor uses this actual sampling step, including
    # when floor-rounded output dimensions truncate the far edge by <1 voxel.
    effective_spacing = spacing/(pre['factor']*np.array(pre['voxel_res']))
    pipeline.save(output/'processing.json',dict(**details,effectiveSpacingZYXum=effective_spacing,parameters=pre))
    pipeline.save(output/'run.json',dict(source=str(raw),originalSource=cfg['source'],originalCropStartZYX=starts,originalCropStopZYX=stops,
        physicalOriginZYXum=starts*spacing+(effective_spacing-spacing)/2,rawCropOriginZYXum=starts*spacing,
        sourceStat=pipeline.source_stat(raw),sourceSHA256=pipeline.digest(raw),spacingZYXum=spacing,
        nativeShape=crop.shape,model='cpsam_v2',diameterProcessedPx=60.,
        cropSelection='Fixed intensity-contrast criterion, not chosen from segmentation outcomes. Cropped-context results are not a full-stack accuracy benchmark.'))
    reference = args.reference_run/'aggregation.json'
    pipeline.save(output/'aggregation.json',dict(parameters=json.loads(reference.read_text())['parameters']))
    print(json.dumps(dict(output=str(output),nativeCropShape=crop.shape,fineShape=shape.tolist(),fineSpacing=effective_spacing.tolist()),default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':main()
