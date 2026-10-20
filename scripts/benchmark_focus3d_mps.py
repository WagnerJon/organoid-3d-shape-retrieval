"""Validate pretrained FOCUS-3D on one real microscopy patch using MPS/CPU."""
import argparse
import json
from pathlib import Path
import time

import tifffile
import torch

from segmentation_explorer import prepare, run, compare, save

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--score-threshold',type=float,default=.5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path('.cache') / ('focus3d-mps-validation-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    original = tifffile.imread(args.raw)
    starts = [(n-p)//2 for n,p in zip(original.shape,(32,96,96))]
    crop = original[starts[0]:starts[0]+32,starts[1]:starts[1]+96,starts[2]:starts[2]+96]
    assert crop.shape == (32,96,96)
    image = folder/'raw_crop.tif'
    tifffile.imwrite(image,crop,metadata={'axes':'ZYX'},photometric='minisblack')
    # Unit calibration and reference cell radius keep this operation benchmark
    # to one patch. This is not a calibrated whole-organoid biological analysis.
    source = prepare(image,[1,1,1],folder/'source')
    report = dict(checkpoint=str(args.checkpoint),raw=str(args.raw),cropStartZYX=starts,shape=list(crop.shape),scoreThreshold=args.score_threshold,torch=torch.__version__,runs={})
    results = {}
    for device in (['mps','cpu'] if args.cpu else ['mps']):
        print(f'\nBENCHMARK {device}',flush=True)
        started = time.perf_counter()
        result = run(source,'focus3d',dict(checkpoint=str(args.checkpoint),device=device,score_threshold=args.score_threshold),folder/device)
        elapsed = time.perf_counter()-started
        report['runs'][device] = dict(seconds=elapsed,cells=len(result['cells']),resolvedDevice=result['settings']['resolvedOptions']['device'],folder=result['folder'])
        results[device] = result
        if device == 'mps':
            torch.mps.synchronize(); torch.mps.empty_cache()
        save(folder/'benchmark.json',report)
    if args.cpu:
        report['agreement'] = compare(results['cpu'],results['mps'],source)
        report['endToEndSpeedup'] = report['runs']['cpu']['seconds']/report['runs']['mps']['seconds']
        save(folder/'benchmark.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
