"""Pinned HTTPS client and whole-stack dispatcher for Atlas workers."""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import ssl
import sys
import tarfile
import tempfile
import threading
import time

from atlas_worker import ROOT,ARTIFACTS,atomic,engine,sha
CREDENTIALS=Path(os.environ.get('ATLAS_WORKER_CREDENTIALS',str(ROOT/'.cache/atlas-workers/credentials.json')))
_CANCEL=threading.Event()
_ACTIVE={}
_LOCAL={}
_LOCK=threading.RLock()
_CONTEXTS={}


class PinnedConnection(http.client.HTTPSConnection):
    def __init__(self,endpoint,timeout=120):
        self.fingerprint=endpoint['fingerprint']
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT);context.check_hostname=False;context.verify_mode=ssl.CERT_NONE
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        super().__init__(endpoint['host'],int(endpoint['port']),context=context,timeout=timeout)
    def connect(self):
        super().connect()
        if not secrets.compare_digest(hashlib.sha256(self.sock.getpeercert(binary_form=True)).hexdigest(),self.fingerprint):
            self.close();raise ssl.SSLError('Worker certificate changed. Pair with this worker again after verifying its fingerprint.')


def credentials(path=None):
    path=Path(path or CREDENTIALS)
    return json.loads(path.read_text()) if path.exists() else {}


def token_for(endpoint,path=None):
    value=credentials(path).get(endpoint['id'])
    if not value or value['fingerprint']!=endpoint['fingerprint']:raise PermissionError('Worker is not paired. Pair it in Nearby Macs.')
    return value['token']


def api(endpoint,method,path,value=None,token=None,timeout=120):
    body=json.dumps(value).encode() if value is not None else None
    headers={'Content-Type':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    connection=PinnedConnection(endpoint,timeout)
    try:
        connection.request(method,path,body,headers);response=connection.getresponse();data=response.read()
        result=json.loads(data)
        if response.status>=400:
            error=RuntimeError(result.get('error','Worker request failed'));error.status=response.status;raise error
        return result
    finally:connection.close()


def pairing(endpoint,code,path=None):
    result=api(endpoint,'POST','/pair',dict(code=code))
    if result['id']!=endpoint['id']:raise ValueError('Worker identity changed during pairing.')
    path=Path(path or CREDENTIALS);values=credentials(path)
    values[endpoint['id']]=dict(fingerprint=endpoint['fingerprint'],token=result['token'])
    atomic(path,values)
    return dict(paired=True,id=result['id'])


def upload(endpoint,path,token,report):
    path=Path(path);before=path.stat();digest=sha(path)
    if api(endpoint,'GET','/blobs/'+digest,token=token)['exists']:
        report('Using cached '+path.name);return digest
    report('Sending '+path.name)
    connection=PinnedConnection(endpoint)
    try:
        with path.open('rb') as source:
            connection.request('PUT','/blobs/'+digest,body=source,headers={'Content-Length':str(before.st_size),'Authorization':'Bearer '+token})
            response=connection.getresponse();reply=json.loads(response.read())
            if response.status!=200:raise RuntimeError(reply.get('error','Input transfer failed'))
        after=path.stat()
        if [before.st_size,before.st_mtime_ns]!=[after.st_size,after.st_mtime_ns]:raise ValueError('Raw file changed during transfer.')
        return digest
    finally:connection.close()


def receive(endpoint,job,token,run,entry):
    run=Path(run);run.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='atlas-receive-',dir=run.parent))
    try:
        connection=PinnedConnection(endpoint)
        archive=temporary/'result.tar.gz'
        try:
            connection.request('GET',f'/jobs/{job}/result',headers={'Authorization':'Bearer '+token})
            response=connection.getresponse()
            if response.status!=200:raise RuntimeError('Worker result is not available.')
            length=int(response.getheader('Content-Length','0'))
            if not 0<length<shutil.disk_usage(temporary).free-512*1024**2:raise ValueError('Insufficient disk space for worker results.')
            h=hashlib.sha256();remaining=length
            with archive.open('wb') as out:
                while remaining:
                    if _CANCEL.is_set():raise InterruptedError('Cancelled')
                    chunk=response.read(min(1024*1024,remaining))
                    if not chunk:raise ValueError('Incomplete worker result transfer')
                    out.write(chunk);h.update(chunk);remaining-=len(chunk)
            if h.hexdigest()!=response.getheader('X-Atlas-SHA256'):raise ValueError('Worker result checksum mismatch')
        finally:connection.close()
        extracted=temporary/'run';extracted.mkdir()
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers();size=0
            for member in members:
                p=Path(member.name)
                if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in ARTIFACTS or not (member.isfile() or member.isdir()):raise ValueError('Invalid worker artifact path/type')
                size+=member.size
            if size>shutil.disk_usage(temporary).free-512*1024**2:raise ValueError('Insufficient disk space to extract results')
            for member in members:
                target=extracted/member.name
                if member.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out,1024*1024)
        required = ['prob.npy','flow.npy','batch.completed.json'] if entry.get('result') else ['run.json','processing.json','complete.json','labels_processed.npy','labels_native.ome.tif','cell_measurements.csv','qc_summary.json']
        for name in required:
            if not (extracted/name).is_file():raise ValueError('Worker result is incomplete: '+name)
        if not entry.get('result'):
            cfg=json.loads((extracted/'run.json').read_text());cfg.update(source=entry['path'],sourceStat=entry['sourceStat'])
            if entry.get('labels'):cfg.update(importedLabels=entry['labels'])
            atomic(extracted/'run.json',cfg)
        if run.exists():run.rename(run.with_name(run.name+'.previous-'+secrets.token_hex(4)))
        extracted.rename(run)
    finally:shutil.rmtree(temporary,ignore_errors=True)


def remote_stack(endpoint,entry,settings,run,report,credential_path=None,task_progress=None):
    token=token_for(endpoint,credential_path)
    info=api(endpoint,'GET','/info')
    if info['id']!=endpoint['id'] or {k:info[k] for k in engine()}!=engine():raise ValueError('Worker processing versions differ; update its package.')
    raw=upload(endpoint,entry['path'],token,report)
    label=upload(endpoint,entry['labels'],token,report) if entry.get('labels') else None
    settings=dict(settings)
    model_hash=None
    if settings.get('model')=='local':
        from atlas_cellpose_models import local_fields
        settings.update(local_fields(settings,freeze=True))
        model_hash=upload(endpoint,settings['localModelPath'],token,report)
        if model_hash!=settings['localModelSHA256']:raise ValueError('Local model changed during worker upload')
    spec=dict(rawHash=raw,labelHash=label,spacing=entry['spacing'],settings=settings)
    if model_hash:spec['modelHash']=model_hash
    if entry.get('result'):spec['kind']='inference-batch'
    payload=dict(spec=spec,compatibility=engine())
    while True:
        if _CANCEL.is_set():raise InterruptedError('Cancelled')
        try:
            submitted=api(endpoint,'POST','/jobs',payload,token);break
        except RuntimeError as e:
            if getattr(e,'status',0)!=409:raise
            report('Worker is busy; waiting');_CANCEL.wait(3)
    job=submitted['id']
    with _LOCK:_ACTIVE[job]=(endpoint,token)
    try:
        failures=0
        while True:
            if _CANCEL.is_set():raise InterruptedError('Cancelled')
            try:saved=api(endpoint,'GET',f'/jobs/{job}',token=token,timeout=10);failures=0
            except (OSError,http.client.HTTPException):
                failures+=1;report('Reconnecting to worker…')
                if failures>=12:raise RuntimeError('Worker disconnected. Completed checkpoints remain on the worker; resume when it reconnects.')
                _CANCEL.wait(min(10,failures));continue
            report(saved['stage'])
            if task_progress:task_progress(float(saved.get('fraction',0)))
            if saved['status']=='complete':break
            if saved['status'] in ('failed','cancelled','interrupted'):raise RuntimeError(saved['stage']+' · Resume preparation to retry cached stages.')
            _CANCEL.wait(1)
        report('Receiving results');receive(endpoint,job,token,run,entry)
        if model_hash and not entry.get('result'):
            cfg=json.loads((Path(run)/'run.json').read_text());cfg.update(local_fields(settings))
            atomic(Path(run)/'run.json',cfg)
        record=dict(worker=endpoint['name'],id=job,sourceSHA256=raw)
        if model_hash:record['checkpointSHA256']=model_hash
        return record
    finally:
        if _CANCEL.is_set():
            with contextlib.suppress(Exception):api(endpoint,'DELETE',f'/jobs/{job}',token=token,timeout=5)
        with _LOCK:_ACTIVE.pop(job,None)


def cancel_all(*unused):
    _CANCEL.set()
    with _LOCK:local=list(_LOCAL.values())
    for process in local:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGTERM)
            def force(p=process):
                import subprocess
                try:p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
            threading.Thread(target=force,daemon=True).start()
    with _LOCK:jobs=list(_ACTIVE.items())
    def stop(item):
        job,(endpoint,token)=item
        with contextlib.suppress(Exception):api(endpoint,'DELETE',f'/jobs/{job}',token=token,timeout=1)
    threads=[threading.Thread(target=stop,args=(item,),daemon=True) for item in jobs]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=1.2)


def local_stack(entry,settings,run,logs,report,task_progress=None):
    import subprocess
    run,logs=Path(run),Path(logs);logs.mkdir(parents=True,exist_ok=True)
    dispatch_name=run.parent.name if entry.get('result') else run.name
    req=logs/(dispatch_name+'-dispatch.json');atomic(req,dict(input=entry,settings=settings))
    with (logs/(dispatch_name+'-dispatch.log')).open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/atlas_local_stack.py'),'--request',str(req),'--output',str(run),'--logs',str(logs)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        with _LOCK:_LOCAL[process.pid]=process
        try:
            if _CANCEL.is_set():cancel_all()
            for line in process.stdout:
                log.write(line);log.flush()
                if line.startswith(('Starting ','Using completed stage:')):report(line.strip())
                if line.startswith('@segmentation-progress '):
                    from atlas_cellpose_3d import stage_sequence
                    STAGES=stage_sequence(settings)
                    value=json.loads(line.split(' ',1)[1]);phase=value['stage']
                    report(value['label'])
                    if task_progress and phase in STAGES:task_progress((STAGES.index(phase)+float(value['fraction']))/len(STAGES))
            if process.wait()!=0:raise RuntimeError('Local stack stopped; checkpoints retained. See '+str(logs/(dispatch_name+'-dispatch.log')))
        finally:
            process.stdout.close()
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGTERM)
            with _LOCK:_LOCAL.pop(process.pid,None)


def dispatch_task(context_id,index,entry,target):
    context=_CONTEXTS[context_id]
    request,output,report,progress,lock=(context[k] for k in ['request','output','report','progress','lock'])
    from prepare_atlas_dataset import prepare_stack
    if _CANCEL.is_set():raise InterruptedError('Cancelled')
    name=target['name'] if target else 'This Mac';run=Path(entry['result']) if entry.get('result') else Path(output)/'stacks'/f'Stack_{index+1:03d}'
    checkpoint='batch.completed.json' if entry.get('result') else 'complete.json'
    if not (run/checkpoint).exists():
        report(index,name,'Starting '+entry.get('name','stack'))
        def task_progress(fraction):
            with lock:
                state=progress.setdefault(index,{});state['fraction']=max(state.get('fraction',0),min(1,max(0,fraction)))
        if target:
            record=remote_stack(target,entry,request['settings'],run,lambda s:report(index,name,s),task_progress=task_progress)
            atomic(run/'worker_provenance.json',record)
        else:local_stack(entry,request['settings'],run,Path(output)/'logs',lambda s:report(index,name,s),task_progress=task_progress)
    with lock:progress[index]=dict(worker=name,stage='Batch complete' if entry.get('result') else 'Stack complete',done=True)
    report(index,name,'Batch complete' if entry.get('result') else 'Stack complete')
    return index


def distribute(request,output):
    from distributed import Client,LocalCluster
    from prepare_atlas_dataset import prepare_stack,event
    _CANCEL.clear();execution=request.get('execution') or {};targets=list(execution.get('workers',[]))
    if execution.get('useLocal',True):targets.insert(0,None)
    if not targets:raise ValueError('Select this Mac or at least one paired worker.')
    import socket
    machines={socket.gethostname()} if execution.get('useLocal',True) else set()
    for endpoint in targets:
        if endpoint:
            token_for(endpoint);info=api(endpoint,'GET','/info')
            if {k:info[k] for k in engine()}!=engine():raise ValueError('Worker versions differ. Update the worker package before starting.')
            machine=info.get('machine',info['name'])
            if machine in machines:raise ValueError('The same Mac was selected more than once. Use one processing slot per Mac.')
            machines.add(machine)
    awake=None
    if sys.platform=='darwin':
        import subprocess
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    original=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *a:cancel_all())
    progress={};lock=threading.Lock()
    def report(index,name,stage):
        with lock:
            previous=progress.get(index,{})
            done=previous.get('done',False)
            progress[index]=dict(worker=name,stage=stage,done=done,fraction=previous.get('fraction',0))
            print('@worker-progress '+json.dumps(dict(index=index,worker=name,stage=stage,unit='batch' if request.get('_inferenceBatches') else 'stack')),flush=True)
            event(f'{name} · {'batch' if request.get('_inferenceBatches') else 'stack'} {index+1}/{len(request["inputs"])} · {stage}', .7*sum(1 if v.get('done',False) else v.get('fraction',0) for v in progress.values())/len(request['inputs']))
    context_id=secrets.token_hex(12)
    _CONTEXTS[context_id]=dict(request=request,output=output,report=report,progress=progress,lock=lock)
    # Each local Dask worker controls one device; only fixed HTTPS jobs cross the LAN.
    try:
        with LocalCluster(n_workers=len(targets),threads_per_worker=1,processes=False,protocol='inproc',dashboard_address=None,resources={'segmentation':1},silence_logs=50) as cluster,Client(cluster) as client:
            addresses=list(client.scheduler_info()['workers'])
            queue=list(enumerate(request['inputs']))
            # Submit just one task per device, replenishing its queue as it finishes.
            pending={}
            def submit(slot):
                if queue:
                    index,entry=queue.pop(0)
                    future=client.submit(dispatch_task,context_id,index,entry,targets[slot],workers=[addresses[slot]],resources={'segmentation':1},pure=False)
                    pending[future]=slot;return future
            for slot in range(len(targets)):submit(slot)
            try:
                while pending:
                    if _CANCEL.is_set():raise InterruptedError('Preparation cancelled')
                    ready=[f for f in pending if f.done()]
                    if not ready:time.sleep(.2);continue
                    for future in ready:
                        slot=pending.pop(future);future.result();submit(slot)
            except BaseException:
                cancel_all();raise
    finally:
        if sys.exc_info()[0]:cancel_all()
        signal.signal(signal.SIGTERM,original)
        _CONTEXTS.pop(context_id,None)
        if awake:awake.terminate()
    return progress


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['info','pair','paired','forget']);p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();request=json.loads(a.request.read_text());endpoint=request['endpoint']
    if a.operation=='pair':result=pairing(endpoint,request['code'])
    elif a.operation=='forget':
        values=credentials();value=values.pop(endpoint['id'],None)
        if value:
            with contextlib.suppress(Exception):api(endpoint,'POST','/unpair',{},value['token'],timeout=2)
        atomic(CREDENTIALS,values);result=dict(forgotten=True)
    elif a.operation=='paired':result=dict(paired=endpoint['id'] in credentials() and credentials()[endpoint['id']]['fingerprint']==endpoint['fingerprint'])
    else:result=api(endpoint,'GET','/info')
    atomic(a.output,result)

if __name__=='__main__':main()
