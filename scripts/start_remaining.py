"""Detach the worker and keep the Mac awake for the worker's lifetime."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cpsam_v2_remaining'
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'progress.log').open('a') as log:
    worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_remaining.py')],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(worker.pid)],stdout=log,stderr=log,start_new_session=True)
time.sleep(2)
if worker.poll() is not None:raise SystemExit(f'Worker exited immediately; see {OUT / "progress.log"}')
print(f'Started worker PID {worker.pid}\nLog: {OUT / "progress.log"}',flush=True)
