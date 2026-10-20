"""Detach the long-running boundary-aware training/evaluation worker."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/boundary_shape"
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "pipeline.log").open("a") as log:
    worker = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/run_boundary_shape.py")],
                              cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(2)
if worker.poll() is not None:
    raise SystemExit(f"Worker exited immediately; see {OUT / 'pipeline.log'}")
print(f"Started boundary-shape worker PID {worker.pid}\nLog: {OUT / 'pipeline.log'}", flush=True)
