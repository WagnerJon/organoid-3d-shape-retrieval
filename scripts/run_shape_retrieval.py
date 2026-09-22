"""Prepare, train, then evaluate the shape-specialized raw-image encoder."""
from pathlib import Path
import json, os, subprocess, sys, traceback
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results/shape_retrieval"; OUT.mkdir(parents=True, exist_ok=True)
(OUT / "worker.pid").write_text(str(os.getpid()))
(OUT / "pipeline_status.json").write_text(json.dumps({"status": "running"}, indent=2))
awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
try:
    if not (OUT / "manifest.csv").exists():
        subprocess.run([sys.executable, "-u", "scripts/prepare_shape_retrieval.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", "scripts/train_shape_retrieval.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", "scripts/evaluate_shape_retrieval.py"], cwd=ROOT, check=True)
    (OUT / "pipeline_status.json").write_text(json.dumps({"status": "complete"}, indent=2))
except Exception:
    (OUT / "pipeline_status.json").write_text(json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2)); raise
finally:
    awake.terminate()
