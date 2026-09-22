"""Run/resume isolated-raw training and then produce matched comparisons."""
from pathlib import Path
import json, os, subprocess, sys, traceback
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/simclr3d_raw_isolated"
(OUT / "worker.pid").write_text(str(os.getpid()))
(OUT / "pipeline_status.json").write_text(json.dumps(
    {"status": "running", "training_and_comparison_complete": False}))
awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
try:
    subprocess.run([sys.executable, "-u", "scripts/train_simclr3d.py", "--config",
                    "config/simclr3d_raw_isolated.json"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", "scripts/compare_isolated_raw_simclr3d.py"],
                   cwd=ROOT, check=True)
    (OUT / "pipeline_status.json").write_text(json.dumps(
        {"status": "complete", "training_and_comparison_complete": True}))
except Exception:
    (OUT / "pipeline_status.json").write_text(json.dumps(
        {"status": "failed", "error": traceback.format_exc()}))
    raise
finally:
    awake.terminate()
