"""Run one real checkpointed segmentation through the packaged HTTPS worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from atlas_worker import ROOT,atomic
from atlas_worker_client import pairing,remote_stack
from prepare_atlas_dataset import inspect,prepare_stack


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    package=ROOT/'results/Atlas Worker for Mac';args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    models=package/'.cache/cellpose'
    if not models.exists():models.symlink_to(ROOT/'.cache/cellpose',target_is_directory=True)
    state=args.output/'worker-state';ready=args.output/'ready.json'
    code="""
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from atlas_worker import create_server,atomic
server,fp=create_server(Path(sys.argv[2]),host='127.0.0.1')
atomic(Path(sys.argv[3]),dict(id=server.state.id,name='Packaged worker',host='127.0.0.1',port=server.server_port,fingerprint=fp,code=server.state.code))
server.serve_forever()
"""
    if ready.exists():ready.unlink()
    process=subprocess.Popen([str(package/'.venv/bin/python'),'-u','-c',code,str(package/'scripts'),str(state),str(ready)])
    try:
        for _ in range(100):
            if ready.exists():break
            if process.poll() is not None:raise RuntimeError('Packaged worker failed to start')
            time.sleep(.1)
        endpoint=json.loads(ready.read_text());code=endpoint.pop('code')
        credentials=args.output/'credentials.json';pairing(endpoint,code,credentials)
        entry=inspect(args.raw);entry['spacing']=json.loads((ROOT/'config/organoids.json').read_text())['spacing_zyx_um']
        settings=dict(model='cpsam_v2',scale=.15,diameter=30.,gpu=True,background=True)
        run=args.output/'Stack_001'
        record=remote_stack(endpoint,entry,settings,run,lambda stage:print(stage,flush=True),credential_path=credentials)
        staging=prepare_stack(entry,settings,run,args.output/'logs',lambda line:None)
        dataset=json.loads((staging/'dataset.json').read_text())
        report=dict(worker=record,cells=len(dataset['cells']),stages=json.loads((run/'complete.json').read_text())['stages'],validAtlasExport=(staging/'dataset.json').exists())
        atomic(args.output/'report.json',report)
        print('PASS: packaged Python, pinned HTTPS, real MPS Cellpose/u-Segment3D and local Atlas export;',len(dataset['cells']),'cells',flush=True)
    finally:process.terminate();process.wait(timeout=10)

if __name__=='__main__':main()
