"""Wait for the joint segmentation export, then run frozen inference in fresh processes."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(folder):
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    try:
        print('Waiting for the five-organoid segmentation export', flush=True)
        while not (folder/'segmentation_complete.json').exists():
            status = folder/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') == 'failed':
                raise RuntimeError('Segmentation pipeline failed; inference remains checkpointed')
            time.sleep(5)
        stages = ['prepare', 'spatial', 'image-inputs', 'image-codes', 'pointnet', 'lite', 'point-mae', 'allen', 'harmonics', 'cajal', 'publish']
        for index, stage in enumerate(stages):
            python = ROOT/'.cache/cajal-env/bin/python' if stage == 'cajal' else Path(sys.executable)
            state = dict(status='running', stage='representations', methodStage=stage, stageIndex=index+1, stageCount=len(stages), updated=time.time())
            (folder/'status.json').write_text(json.dumps(state, indent=2))
            print('Starting representation stage:', stage, flush=True)
            with (folder/'logs'/f'representations_{stage}.log').open('a') as log:
                subprocess.run([str(python), str(ROOT/'scripts/infer_joint_organoid_methods.py'), str(folder), '--stage', stage], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('Completed representation stage:', stage, flush=True)
        with (folder/'logs/validation.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/validate_joint_organoids.py'), str(folder)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        (folder/'status.json').write_text(json.dumps(dict(status='complete', validated=True, updated=time.time()), indent=2))
        print('COMPLETE: five organoids, all 36 transferable methods, validated joint Atlas dataset', flush=True)
    except Exception as exc:
        (folder/'inference_failure.json').write_text(json.dumps(dict(error=repr(exc), updated=time.time()), indent=2))
        print('Inference stopped:', repr(exc), flush=True)
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
