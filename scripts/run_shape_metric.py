"""Train and evaluate the controlled shape-target embedding experiment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/shape_metric"


def status(value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline_status.json").write_text(json.dumps(value, indent=2))


def main():
    if not (OUT / "metric_targets.npy").exists():
        raise RuntimeError("Run prepare_shape_metric.py first")
    cfg = json.loads((ROOT / "config/shape_metric.json").read_text())
    seeds = cfg.get("seeds", [cfg["seed"]])
    status({"status": "training", "pid": os.getpid(), "seeds": seeds, "current_seed": seeds[0]})
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for seed in seeds:
            status({"status": "training", "pid": os.getpid(), "seeds": seeds, "current_seed": seed})
            print(f"SEED {seed}: training", flush=True)
            subprocess.run([sys.executable, "-u", "scripts/train_shape_metric.py", "--seed", str(seed)], cwd=ROOT, check=True)
            status({"status": "evaluating", "pid": os.getpid(), "seeds": seeds, "current_seed": seed})
            subprocess.run([sys.executable, "-u", "scripts/evaluate_shape_metric.py", "--seed", str(seed)], cwd=ROOT, check=True)
        status({"status": "complete", "pid": os.getpid(), "seeds": seeds})
    except Exception:
        status({"status": "failed", "pid": os.getpid(), "error": traceback.format_exc()})
        raise
    finally:
        awake.terminate()


if __name__ == "__main__":
    main()
