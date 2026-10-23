"""Real pretrained Cellpose inference smoke test on one microscopy plane."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLPOSE_LOCAL_MODELS_PATH',str(ROOT/'.cache/cellpose'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import tifffile
from atlas_cellpose_models import MODEL_NAMES, make_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--model',choices=MODEL_NAMES,required=True)
    args = parser.parse_args()
    raw = tifffile.imread(args.raw)
    image = raw[len(raw)//2]
    start = time.perf_counter()
    model = make_model(pretrained_model=args.model,gpu=True)
    masks, flows, styles = model.eval(image,diameter=30,compute_masks=False)
    assert flows[1].shape == (2,*image.shape), flows[1].shape
    assert np.isfinite(flows[1]).all() and np.isfinite(flows[2]).all()
    report = dict(model=args.model,device=str(model.device),shape=list(image.shape),seconds=time.perf_counter()-start,
                  flowShape=list(flows[1].shape),probabilityShape=list(flows[2].shape))
    (ROOT/'.cache'/f'{args.model}-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
