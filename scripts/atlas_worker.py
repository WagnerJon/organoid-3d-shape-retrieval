"""Atlas LAN worker: pinned HTTPS, explicit pairing, content-addressed inputs.

Only fixed stack-processing jobs are accepted. Dask is never exposed on the LAN.
"""
import argparse
import contextlib
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import threading
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 1
HEX = re.compile(r'^[a-f0-9]{64}$')
ARTIFACTS = ['run.json','processing.json','aggregation.json','summary.json','complete.json','qc_summary.json','prob.npy','flow.npy','batch.completed.json',
             'cell_measurements.csv','cell_measurements_pipeline.csv','labels_processed.npy','labels_processed.ome.tif',
             'labels_native.ome.tif','preview.png','QUALITY_CONTROL.md','README.md','meshes','logs','streamed_linking']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,allow_nan=False));os.chmod(temp,0o600);temp.replace(path)


@contextlib.contextmanager
def compute_slot():
    directory=Path(os.environ.get('ATLAS_WORKER_COMPUTE_DIR',tempfile.gettempdir()))
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'organoid-atlas-compute.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def engine():
    from importlib.metadata import version,PackageNotFoundError
    files=['run_standalone_cellpose.py','qc_standalone_cellpose.py','prepare_atlas_dataset.py','atlas_worker.py','atlas_inference_batches.py','atlas_preprocessing.py','atlas_ome_zarr.py','atlas_numpy.py','atlas_streamed_linking.py','atlas_streaming_core.py']
    h=hashlib.sha256()
    for name in files:h.update((name+sha(ROOT/'scripts'/name)).encode())
    for path in sorted((ROOT/'vendor/u-segment3D/segment3D').rglob('*.py')):
        h.update((str(path.relative_to(ROOT/'vendor/u-segment3D'))+sha(path)).encode())
    for path in sorted((ROOT/'vendor/dinov3/dinov3').rglob('*.py')):
        h.update((str(path.relative_to(ROOT/'vendor/dinov3'))+sha(path)).encode())
    versions={}
    for name in ['cellpose','torch','numpy','scipy','scikit-image','tifffile','trimesh','numba','edt','connected-components-3d','opencv-python-headless','dask']:
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]='missing'
    return dict(protocol=PROTOCOL,engine=h.hexdigest(),versions=versions,python=list(sys.version_info[:2]))


def certificate(state):
    cert,key=state/'certificate.pem',state/'private.pem'
    if not cert.exists() or not key.exists():
        subprocess.run(['/usr/bin/openssl','req','-x509','-newkey','rsa:3072','-sha256','-nodes','-days','3650',
                        '-subj','/CN=Atlas Worker','-keyout',str(key),'-out',str(cert)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        os.chmod(key,0o600)
    return cert,key,hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()


class WorkerState:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.folder.mkdir(parents=True,exist_ok=True);os.chmod(self.folder,0o700)
        self.lock=threading.RLock();self.upload_lock=threading.Lock();self.active=None;self.process=None
        self.identity=self.folder/'identity.json'
        if not self.identity.exists():atomic(self.identity,dict(id=secrets.token_hex(16)))
        self.id=json.loads(self.identity.read_text())['id'];self.code=f'{secrets.randbelow(1000000):06d}'
        self.pair_until=time.time()+600;self.pair_attempts=0
        self.tokens_path=self.folder/'tokens.json'
        self.tokens=json.loads(self.tokens_path.read_text()) if self.tokens_path.exists() else {}
        self.blob_owners_path=self.folder/'blob-owners.json'
        self.blob_owners=json.loads(self.blob_owners_path.read_text()) if self.blob_owners_path.exists() else {}
        self.compatibility=engine();self.computeDirectory=None;self.machine=socket.gethostname();self.stopping=False
        (self.folder/'blobs').mkdir(exist_ok=True);(self.folder/'jobs').mkdir(exist_ok=True)
        for path in (self.folder/'jobs').glob('*/state.json'):
            state=json.loads(path.read_text())
            if state['status'] in ('running','packaging'):state.update(status='interrupted',stage='Worker restarted; resume this job');atomic(path,state)

    def info(self):
        with self.lock:return dict(id=self.id,name=socket.gethostname(),machine=self.machine,slots=1,busy=self.active is not None,**self.compatibility)

    def pair(self,code):
        with self.lock:
            self.pair_attempts+=1
            if self.pair_attempts>10 or time.time()>self.pair_until or not secrets.compare_digest(str(code),self.code):
                raise PermissionError('Pairing code is invalid or expired. Restart sharing to obtain a new code.')
            token=secrets.token_urlsafe(32);owner=hashlib.sha256(token.encode()).hexdigest()
            self.tokens[owner]=dict(created=time.time());atomic(self.tokens_path,self.tokens)
            self.pair_until=0
            return dict(token=token,id=self.id)

    def owner(self,token):
        key=hashlib.sha256(token.encode()).hexdigest()
        if key not in self.tokens:raise PermissionError('Pair this Atlas with the worker first.')
        return key

    def job(self,job_id,owner):
        if not HEX.fullmatch(job_id):raise ValueError('Invalid job ID')
        folder=self.folder/'jobs'/job_id
        state=json.loads((folder/'state.json').read_text())
        if state['owner']!=owner:raise PermissionError('Job belongs to a different paired Atlas.')
        return folder,state

    def submit(self,payload,owner):
        if payload.get('compatibility')!=self.compatibility:raise ValueError('Atlas and worker processing versions differ. Update the worker package.')
        spec=payload['spec'];raw=spec['rawHash'];labels=spec.get('labelHash')
        if not HEX.fullmatch(raw) or (labels and not HEX.fullmatch(labels)):raise ValueError('Invalid input identity')
        for value in [raw]+([labels] if labels else []):
            path=self.folder/'blobs'/value
            if owner not in self.blob_owners.get(value,[]) or not path.is_file() or sha(path)!=value:raise ValueError('Upload the verified input files first.')
        if spec.get('kind','stack') not in ('stack','inference-batch'):raise ValueError('Unsupported job kind')
        if spec.get('kind')=='inference-batch' and labels:raise ValueError('Batch jobs do not accept labels')
        spacing=spec['spacing'];settings=spec['settings']
        from math import isfinite
        if len(spacing)!=3 or not all(isinstance(x,(int,float)) and isfinite(x) and x>0 for x in spacing):raise ValueError('Invalid spacing')
        if settings.get('model') not in ('cpsam_v2','cpdino','cpdino-vitb') or not .05<=float(settings['scale'])<=1 or not 2<=float(settings['diameter'])<=500:raise ValueError('Invalid segmentation settings')
        if type(settings.get('gpu')) is not bool or type(settings.get('background')) is not bool:raise ValueError('Invalid device/background settings')
        from atlas_streamed_linking import validate_mode
        validate_mode(settings.get('linkingMode'))
        identity=dict(spec=spec,compatibility=self.compatibility,owner=owner)
        job_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        with self.lock:
            folder=self.folder/'jobs'/job_id
            if (folder/'state.json').exists():
                _,saved=self.job(job_id,owner)
                if saved['status'] in ('complete','running','packaging'):return dict(id=job_id,**saved)
            if self.active is not None:raise BlockingIOError('Worker is processing another stack. Wait for it to finish.')
            if self.stopping:raise RuntimeError('Worker is stopping')
            folder.mkdir(exist_ok=True)
            atomic(folder/'request.json',identity)
            saved=dict(owner=owner,status='running',stage='Starting stack',updated=time.time())
            atomic(folder/'state.json',saved);self.active=job_id
            threading.Thread(target=self.execute,args=(job_id,identity),daemon=True).start()
            return dict(id=job_id,**saved)

    def execute(self,job_id,identity):
        folder=self.folder/'jobs'/job_id
        try:
            with (folder/'worker.log').open('a') as log:
                with self.lock:
                    if json.loads((folder/'state.json').read_text())['status']=='cancelled':return
                    environment=dict(os.environ)
                    if self.computeDirectory:environment['ATLAS_WORKER_COMPUTE_DIR']=str(self.computeDirectory)
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run-job','--state',str(self.folder),'--job',job_id],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
                    self.process=p
                    if self.stopping:self.stop_process(p)
                code=p.wait()
            with self.lock:
                state=json.loads((folder/'state.json').read_text())
                if state['status']=='cancelled':return
                if code!=0:raise RuntimeError('Stack processing stopped; '+(folder/'worker.log').read_text()[-1800:])
                state.update(status='packaging',stage='Packaging results');atomic(folder/'state.json',state)
            archive=folder/'result.tar.gz';temp=folder/'result.tmp'
            with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
                for name in ARTIFACTS:
                    path=folder/'run'/name
                    if path.exists():tar.add(path,arcname=name)
            temp.replace(archive)
            with self.lock:
                current=json.loads((folder/'state.json').read_text())
                if current['status']=='cancelled':return
                current.update(status='complete',stage='Ready',archiveSHA256=sha(archive),bytes=archive.stat().st_size);atomic(folder/'state.json',current)
        except Exception as error:
            with self.lock:
                current=json.loads((folder/'state.json').read_text())
                if current['status']!='cancelled':current.update(status='failed',stage=str(error));atomic(folder/'state.json',current)
        finally:
            with self.lock:self.active=None;self.process=None

    @staticmethod
    def stop_process(p):
        if p and p.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
            def force():
                try:p.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
            threading.Thread(target=force,daemon=True).start()

    def cancel(self,job_id,owner):
        with self.lock:
            folder,state=self.job(job_id,owner)
            if state['status'] in ('running','packaging'):
                state.update(status='cancelled',stage='Cancelled; checkpoints retained');atomic(folder/'state.json',state)
                if self.active==job_id:self.stop_process(self.process)
            return dict(status='cancelled')


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def setup(self):super().setup();self.connection.settimeout(120)
    def respond(self,status,value):
        data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def body(self):
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=65536:raise ValueError('Invalid request length')
        return json.loads(self.rfile.read(length))
    def authorized(self):return self.server.state.owner(self.headers.get('Authorization','').removeprefix('Bearer '))
    def route(self):
        state=self.server.state;path=self.path.split('?')[0];parts=path.strip('/').split('/')
        if self.command=='GET' and path=='/info':return self.respond(200,state.info())
        if self.command=='POST' and path=='/pair':return self.respond(200,state.pair(self.body().get('code','')))
        owner=self.authorized()
        if len(parts)==2 and parts[0]=='blobs' and HEX.fullmatch(parts[1]):
            target=state.folder/'blobs'/parts[1]
            if self.command=='GET':return self.respond(200,dict(exists=target.is_file() and owner in state.blob_owners.get(parts[1],[])))
            if self.command=='PUT':
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<shutil.disk_usage(state.folder).free-512*1024**2:raise ValueError('Insufficient cache disk space or invalid upload size')
                if not state.upload_lock.acquire(blocking=False):raise BlockingIOError('Another upload is in progress')
                temp=target.with_suffix('.'+secrets.token_hex(8)+'.upload')
                try:
                    remaining=length;h=hashlib.sha256()
                    with temp.open('wb') as out:
                        while remaining:
                            chunk=self.rfile.read(min(1024*1024,remaining))
                            if not chunk:raise ValueError('Incomplete upload')
                            out.write(chunk);h.update(chunk);remaining-=len(chunk)
                    if h.hexdigest()!=parts[1]:raise ValueError('Input checksum mismatch')
                    temp.replace(target)
                    with state.lock:
                        owners=state.blob_owners.setdefault(parts[1],[])
                        if owner not in owners:owners.append(owner)
                        atomic(state.blob_owners_path,state.blob_owners)
                    return self.respond(200,dict(verified=True))
                finally:
                    with contextlib.suppress(FileNotFoundError):temp.unlink()
                    state.upload_lock.release()
        if self.command=='POST' and path=='/unpair':
            with state.lock:
                if state.active:
                    active=json.loads((state.folder/'jobs'/state.active/'state.json').read_text())
                    if active['owner']==owner:state.cancel(state.active,owner)
                state.tokens.pop(owner,None);atomic(state.tokens_path,state.tokens)
            return self.respond(200,dict(forgotten=True))
        if self.command=='POST' and path=='/jobs':return self.respond(200,state.submit(self.body(),owner))
        if len(parts) in (2,3) and parts[0]=='jobs':
            folder,saved=state.job(parts[1],owner)
            if len(parts)==2:
                if self.command=='DELETE':return self.respond(200,state.cancel(parts[1],owner))
                if self.command=='GET':
                    status=folder/'run/status.json'
                    if saved['status']=='running' and status.exists():
                        with contextlib.suppress(ValueError):
                            detail=json.loads(status.read_text());saved.update(stage=detail.get('stage',saved['stage']),fraction=detail.get('taskFraction',0))
                    saved.pop('owner',None);return self.respond(200,saved)
            if len(parts)==3 and parts[2]=='result' and self.command=='GET' and saved['status']=='complete':
                path=folder/'result.tar.gz';self.send_response(200);self.send_header('Content-Length',str(path.stat().st_size));self.send_header('X-Atlas-SHA256',saved['archiveSHA256']);self.end_headers()
                with path.open('rb') as stream:shutil.copyfileobj(stream,self.wfile,1024*1024)
                return
        raise FileNotFoundError('Unknown worker operation')
    def dispatch(self):
        try:self.route()
        except PermissionError as e:self.respond(403,dict(error=str(e)))
        except BlockingIOError as e:self.respond(409,dict(error=str(e)))
        except FileNotFoundError:self.respond(404,dict(error='Not found'))
        except (ValueError,KeyError,TypeError) as e:self.respond(400,dict(error=str(e)))
        except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        except Exception:self.respond(500,dict(error='Worker operation failed; inspect the worker log.'))
    do_GET=dispatch;do_POST=dispatch;do_PUT=dispatch;do_DELETE=dispatch


def create_server(folder,host='0.0.0.0',port=0):
    state=WorkerState(folder);cert,key,fingerprint=certificate(state.folder)
    server=ThreadingHTTPServer((host,port),Handler);server.daemon_threads=True;server.state=state
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert,key);server.socket=context.wrap_socket(server.socket,server_side=True,do_handshake_on_connect=False)
    return server,fingerprint


def run_job(state,job_id):
    from prepare_atlas_dataset import inspect,prepare_stack
    folder=state/'jobs'/job_id;request=json.loads((folder/'request.json').read_text());spec=request['spec']
    raw=state/'blobs'/spec['rawHash']
    if spec.get('kind')=='inference-batch':
        from atlas_inference_batches import infer
        return infer(raw,spec['settings'],folder/'run')
    entry=inspect(raw);entry.update(spacing=spec['spacing'])
    if spec.get('labelHash'):
        label=state/'blobs'/spec['labelHash'];s=label.stat();entry.update(labels=str(label),labelStat=[s.st_size,s.st_mtime_ns])
    prepare_stack(entry,spec['settings'],folder/'run',folder,lambda line:None,export_atlas=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['serve','run-job']);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--port',type=int,default=0);parser.add_argument('--job');args=parser.parse_args()
    if args.operation=='run-job':return run_job(args.state,args.job)
    args.state.mkdir(parents=True,exist_ok=True)
    lock=(args.state/'instance.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    singleton=Path(os.environ.get('ATLAS_WORKER_INSTANCE_DIR',str(Path.home()/'Library/Application Support/Organoid Atlas Worker')))
    singleton.mkdir(parents=True,exist_ok=True)
    machine_lock=(singleton/'instance.lock').open('w');fcntl.flock(machine_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    server,fp=create_server(args.state,port=args.port)
    print('@worker-ready '+json.dumps(dict(port=server.server_port,fingerprint=fp,code=server.state.code,**server.state.info())),flush=True)
    def stop(*unused):
        server.state.stopping=True;server.state.stop_process(server.state.process)
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parent=os.getppid()
    def watch_parent():
        while not server.state.stopping:
            time.sleep(2)
            if os.getppid()!=parent:stop();return
    threading.Thread(target=watch_parent,daemon=True).start()
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
    try:server.serve_forever(poll_interval=.3)
    finally:
        server.server_close()
        if awake:awake.terminate()

if __name__=='__main__':main()
