"""Refresh artifacts during the active training run, then exit at completion."""
import json,time,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/simclr3d';last=None
while True:
    try:
        state=json.loads((out/'status.json').read_text());key=(state['status'],state.get('epoch',state.get('epochs')))
        if key!=last:
            subprocess.run([sys.executable,str(root/'scripts/report_simclr3d.py')],cwd=root,check=True)
            last=key
        if state['status']=='complete':break
        if time.time()-(out/'status.json').stat().st_mtime>1800:
            print('Training status has not advanced for 30 minutes; stopping report watcher.',flush=True);break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(60)
