"""Use one OpenMP runtime for PyPI PyTorch and conda's segmentation libraries.

Keeping both libomp copies causes OpenMP initialization to abort. Resolve the
bundled PyTorch library to conda's runtime; do not suppress duplicate errors.
Only the isolated PanSeg environment is modified. The original is retained.
"""
from pathlib import Path
import sys

prefix = Path(sys.prefix)
if prefix.name != 'panseg-env':
    raise SystemExit('Run with .cache/panseg-env/bin/python only.')
bundled = prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/lib/libomp.dylib'
shared = prefix / 'lib/libomp.dylib'
if bundled.exists() and not bundled.is_symlink() and shared.exists():
    backup = bundled.with_name('libomp.pypi-backup.dylib')
    if backup.exists():
        raise SystemExit('OpenMP backup already exists; inspect the environment before reconfiguring.')
    bundled.replace(backup)
    bundled.symlink_to(shared)
print('PanSeg OpenMP runtime:', bundled.resolve())
