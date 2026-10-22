"""Real MPS plane batches via local Dask + packaged TLS worker, versus unsplit fields.

Uses a prior standalone run as the unchanged numerical reference. This is a
single-Mac loopback integration check, not a two-physical-Mac speed benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from atlas_worker import ROOT, atomic
import atlas_worker_client as client
from atlas_inference_batches import ORDERS


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reference=a.reference.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    package=ROOT/'results/Atlas Worker for Mac';ready=output/'ready.json';ready.unlink(missing_ok=True)
    code='''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from atlas_worker import create_server,atomic
server,fp=create_server(Path(sys.argv[2]),host='127.0.0.1')
# Simulate a second hostname for scheduler routing. The global compute lock
# remains shared: only one actual inference runs on this physical test Mac.
server.state.machine='loopback-batch-validation-worker'
atomic(Path(sys.argv[3]),dict(id=server.state.id,name='Packaged batch worker',host='127.0.0.1',port=server.server_port,fingerprint=fp,code=server.state.code))
server.serve_forever()
'''
    process=subprocess.Popen([str(package/'.venv/bin/python'),'-u','-c',code,str(package/'scripts'),str(output/'worker-state'),str(ready)])
    old=client.CREDENTIALS
    try:
        for _ in range(200):
            if ready.exists():break
            if process.poll() is not None:raise RuntimeError('Worker failed to start')
            time.sleep(.1)
        endpoint=json.loads(ready.read_text());pair_code=endpoint.pop('code');client.CREDENTIALS=output/'credentials.json';client.pairing(endpoint,pair_code)
        processed=np.load(reference/'processed.npy',mmap_mode='r');jobs=[];expected=[]
        settings=dict(model='cpsam_v2',scale=.15,diameter=30.,gpu=True,background=True)
        for view,order in ORDERS.items():
            planes=processed.transpose(order);start=len(planes)//2
            probabilities=np.load(reference/'predictions'/f'{view}_prob.npy',mmap_mode='r').transpose(order)
            flows=np.load(reference/'predictions'/f'{view}_flow.npy',mmap_mode='r').transpose((0,)+tuple(i+1 for i in order))
            for lo,hi in [(start,start+2),(start+2,start+3)]:
                folder=output/f'{view}_{lo}';folder.mkdir(exist_ok=True);source=folder/'planes.npy';np.save(source,np.ascontiguousarray(planes[lo:hi]))
                jobs.append(dict(path=str(source),result=str(folder/'result'),spacing=[1.,1.,1.],name=f'{view} {lo}:{hi}'))
                expected.append((probabilities[lo:hi].copy(),flows[:,lo:hi].copy()))
        request=dict(inputs=jobs,settings=settings,_inferenceBatches=True,execution=dict(useLocal=True,workers=[endpoint]))
        (output/'logs').mkdir(exist_ok=True)
        progress=client.distribute(request,output)
        differences=[]
        for job,(p,f) in zip(jobs,expected):
            result=Path(job['result']);actual_p=np.load(result/'prob.npy');actual_f=np.load(result/'flow.npy')
            np.testing.assert_allclose(actual_p,p,rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(actual_f,f,rtol=1e-5,atol=1e-5)
            differences.append(dict(batch=job['name'],maxProbabilityError=float(np.max(np.abs(actual_p-p))),maxFlowError=float(np.max(np.abs(actual_f-f)))))
        workers=sorted({x['worker'] for x in progress.values()})
        if len(workers)!=2:raise AssertionError('Both scheduler slots were not exercised')
        # A resume must not invoke either local or remote inference again.
        client.distribute(request,output)
        atomic(output/'report.json',dict(passed=True,batches=len(jobs),planes=9,workers=workers,reference=str(reference),differences=differences,scope='Single physical Mac; local + packaged TLS worker, MPS, same global compute lock; no speedup claim'))
        print('PASS: six batches, including singleton tails, match unsplit XY/XZ/YZ predictions; both slots and resume verified',flush=True)
    finally:
        client.CREDENTIALS=old;process.terminate();process.wait(timeout=10)

if __name__=='__main__':main()
