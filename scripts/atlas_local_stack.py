"""Isolated local stack process used by Atlas's Dask dispatcher."""
import argparse,json
from pathlib import Path
from prepare_atlas_dataset import prepare_stack

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--logs',type=Path,required=True);a=p.parse_args()
    request=json.loads(a.request.read_text())
    if request['input'].get('result'):
        from atlas_inference_batches import infer
        infer(request['input']['path'], request['settings'], a.output)
    else: prepare_stack(request['input'],request['settings'],a.output,a.logs,lambda line:None,export_atlas=False)
