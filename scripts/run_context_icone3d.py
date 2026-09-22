"""Run context-rich raw IConE, then refresh organoid-level comparisons."""
from pathlib import Path
import json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/icone3d_raw_context"
OUT.mkdir(exist_ok=True);(OUT/"worker.pid").write_text(str(os.getpid()))
(OUT/"pipeline_status.json").write_text(json.dumps({"status":"running","training_and_comparison_complete":False}))
awake=subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(os.getpid())])
try:
    subprocess.run([sys.executable,"-u","scripts/train_icone3d.py","--config",
                    "config/icone3d_raw_context.json"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u","scripts/compare_organoid_representations.py"],cwd=ROOT,check=True)
    (OUT/"pipeline_status.json").write_text(json.dumps({"status":"complete","training_and_comparison_complete":True}))
except Exception:
    (OUT/"pipeline_status.json").write_text(json.dumps({"status":"failed","error":traceback.format_exc()}));raise
finally:awake.terminate()
