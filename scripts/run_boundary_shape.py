"""Train then evaluate the boundary-aware raw-image experiment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/boundary_shape"


def status(value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline_status.json").write_text(json.dumps(value, indent=2))


def main():
    if not (OUT / "prepare_complete.json").exists():
        raise RuntimeError("Dataset preparation must finish before training")
    status({"status": "training", "pid": os.getpid()})
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        subprocess.run([sys.executable, "-u", "scripts/train_boundary_shape.py"], cwd=ROOT, check=True)
        status({"status": "evaluating", "pid": os.getpid()})
        subprocess.run([sys.executable, "-u", "scripts/evaluate_boundary_shape.py"], cwd=ROOT, check=True)
        status({"status": "complete", "pid": os.getpid()})
    except Exception:
        status({"status": "failed", "pid": os.getpid(), "error": traceback.format_exc()})
        raise
    finally:
        awake.terminate()


if __name__ == "__main__":
    main()
