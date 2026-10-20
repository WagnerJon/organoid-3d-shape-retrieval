"""Continue the authorized offline benchmark after full-resolution caching."""
import json,os,subprocess,sys,time
from pathlib import Path
from allen_reference_core import OUT
from pointnet_shape_core import ROOT,sha256


def main():
    started=time.monotonic()
    while not (OUT/'cache.completed.json').exists():time.sleep(5)
    cache=json.loads((OUT/'cache.completed.json').read_text())
    if cache.get('status')!='complete' or cache.get('jobs')!=669:raise ValueError('Incomplete backbone cache')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),XDG_CACHE_HOME=str(ROOT/'.cache'))
    for phase,script in [('adapting','train_allen_reference.py'),('evaluating','evaluate_allen_reference.py'),('reporting','report_allen_reference.py')]:
        print('BENCHMARK PHASE',phase,flush=True)
        (OUT/'analysis_status.json').write_text(json.dumps(dict(status=phase,seconds=time.monotonic()-started)))
        with (OUT/(phase+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if process.wait()!=0:raise RuntimeError('Benchmark phase failed: '+phase)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-started,report_sha256=sha256(OUT/'report.md'))))
    print('Standalone benchmark and report complete',flush=True)


if __name__=='__main__':main()
