"""Resume the five GFP organoids sequentially, then publish one Atlas dataset."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def call(args, log):
    print('Starting', ' '.join(map(str, args)), flush=True)
    with log.open('a') as handle:
        subprocess.run([str(x) for x in args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'logs').mkdir(exist_ok=True)
    (folder/'raw').mkdir(exist_ok=True)
    (folder/'organoids').mkdir(exist_ok=True)
    source = folder.parent
    previous = source/'GFP-24hrs-2ndy-cleared-MV_5_cellpose_uSegment3D_scale035'
    reused = folder/'organoids/MV5'
    if not reused.exists():
        print('Copying completed MV5 outputs into joint folder', flush=True)
        shutil.copytree(previous, reused, ignore=shutil.ignore_patterns('._*', 'atlas_dataset', 'renderings'))
    inventory = []
    for number in range(1, 6):
        raw = source/f'C2-GFP-24hrs-2ndy-cleared-MV. - GFP-24hrs-2ndy-cleared-MV #{number}.tif'
        copy = folder/'raw'/raw.name
        if not copy.exists():
            shutil.copy2(raw, copy)
        assert copy.stat().st_size == raw.stat().st_size
        inventory.append(dict(organoid=f'MV{number}', sample=f'GFP24h_MV{number}', raw=str(copy), original=str(raw)))
    save(folder/'inventory.json', inventory)
    (folder/'README.md').write_text('''# GFP 24 h — five organoids\n\n`raw/` contains copies of the five source stacks. `organoids/MV1`–`MV5` contain calibrated Cellpose cpsam_v2 + uSegment3D outputs, meshes, measurements, intermediate predictions and QC. MV5 reuses the completed previous run. `atlas_dataset/` is the combined Atlas experiment; choose this folder with Command-O in Atlas. `representations/` contains frozen inference outputs and provenance.\n\nStacks are distinct organoids from the same dataset. Physical calibration is verified from each TIFF. Scale 0.35, processed diameter 30 pixels, background correction enabled, whole-volume hole filling disabled. Each segmentation stage runs in a separate process; organoids are processed sequentially. All candidate objects are retained and QC flags remain visible. No biological-condition discrimination claim is made.\n''')
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for number in [4, 3, 2, 1]:
            entry = inventory[number-1]
            output = folder/'organoids'/f'MV{number}'
            save(folder/'status.json', dict(status='running', stage='segmentation', organoid=f'MV{number}', updated=time.time()))
            call([sys.executable, ROOT/'scripts/run_standalone_cellpose.py', '--input', entry['raw'], '--output', output], folder/'logs'/f'MV{number}.log')
            if not (output/'qc_summary.json').exists():
                call([sys.executable, ROOT/'scripts/qc_standalone_cellpose.py', output], folder/'logs'/f'MV{number}_qc.log')
        parts = []
        for number in range(1, 6):
            output = folder/'organoids'/f'MV{number}'
            sample = f'GFP24h_MV{number}'
            staging = folder/'organoids'/f'MV{number}'/'atlas_export'
            if not (staging/'dataset.json').exists():
                call([sys.executable, ROOT/'scripts/export_standalone_atlas.py', output, staging, '--sample-id', sample], folder/'logs'/f'MV{number}_atlas.log')
            parts.append((staging, json.loads((staging/'dataset.json').read_text())))
        atlas = folder/'atlas_dataset'
        atlas.mkdir(exist_ok=True)
        for staging, part in parts:
            sample = part['samples'][0]
            sample['sourceImage'] = inventory[int(sample['id'][-1])-1]['raw']
            destination = atlas/sample['id']
            if not destination.exists():
                shutil.copytree(staging/sample['id'], destination, ignore=shutil.ignore_patterns('._*'))
        data = dict(parts[0][1])
        data.update(title='GFP 24 h — five organoids',
                    cells=[c for _, d in parts for c in d['cells']],
                    organoids=[c for _, d in parts for c in d['organoids']],
                    samples=[s for _, d in parts for s in d['samples']])
        # Use joint included-cohort normalization and plots for the ordinary
        # traditional space; frozen comparison encoders are added separately.
        import numpy as np
        from organoid_membership import project
        features = data['traditionalFeatures']
        fit = np.array([[c['metrics'][k] for k in features] for c in data['cells'] if c['qc']])
        mean, std = fit.mean(0), np.maximum(fit.std(0), 1e-12)
        data['normalization'] = dict(mean=mean.tolist(), std=std.tolist())
        for c in data['cells']:
            c['representations']['traditional']['vector'] = ((np.array([c['metrics'][k] for k in features])-mean)/std).tolist()
        data['methods'][0]['variance'] = project(data['cells'], 'traditional')
        for stack in data['organoids']:
            use = [c for c in data['cells'] if c['sample']==stack['sample'] and c['qc']]
            if not use:
                use = [c for c in data['cells'] if c['sample']==stack['sample']]
                stack['qc'] = False
            stack['representations']['traditional']['vector'] = np.mean([c['representations']['traditional']['vector'] for c in use], axis=0).tolist()
        data['methods'][0]['stackVariance'] = project(data['organoids'], 'traditional')
        data['provenance'] = 'Five distinct GFP 24 h organoids, TIFF-verified calibration. Original segmentation surfaces; all candidate objects retained, QC flags preserved. Organoid membership has not been classified; inclusion is provisional. Raw data copies, segmentation outputs and inference caches are bundled together. No experimental condition labels or biological validation supplied.'
        path = atlas/'dataset.json'
        if not path.exists():
            save(path, data)
        save(folder/'segmentation_complete.json', dict(status='complete', organoids=5, objects=len(data['cells']), qcPassing=sum(c['qc'] for c in data['cells'])))
        save(folder/'status.json', dict(status='segmentation_complete', objects=len(data['cells']), updated=time.time()))
        print('All five segmentations and joint Atlas export complete', flush=True)
    except Exception as exc:
        save(folder/'status.json', dict(status='failed', error=repr(exc), updated=time.time()))
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
