#!/bin/zsh
# Install the headless PanSeg backend without changing the reconstruction environment.
set -eu
PROJECT_ROOT="${0:A:h:h}"
PANSEG_COMMIT=a0c7123b0dc9ad59141b5dde4e1b5017c60ae53a
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then CONDA_EXE="$HOME/miniforge3/bin/conda"; fi
if [[ -z "$CONDA_EXE" ]]; then print 'Install Miniforge (conda) first.'; exit 1; fi
mkdir -p "$PROJECT_ROOT/vendor"
if [[ ! -d "$PROJECT_ROOT/vendor/panseg/.git" ]]; then
    git clone https://github.com/kreshuklab/panseg.git "$PROJECT_ROOT/vendor/panseg"
fi
git -C "$PROJECT_ROOT/vendor/panseg" checkout "$PANSEG_COMMIT"
if [[ -d "$PROJECT_ROOT/.cache/panseg-env/conda-meta" ]]; then
    PANSEG_ACTION=install
else
    PANSEG_ACTION=create
fi
"$CONDA_EXE" "$PANSEG_ACTION" -y -p "$PROJECT_ROOT/.cache/panseg-env" -c conda-forge \
    python=3.12 python-elf nifty vigra pytorch torchvision bioimageio.core \
    pydantic pyyaml scikit-image tifffile h5py trimesh requests pandas pooch tensorboard napari pyside6 qt-material qtpy python-graphviz
# PyPI PyTorch 2.14 supports Conv3d on Apple MPS; conda's 2.13 build does not.
"$PROJECT_ROOT/.cache/panseg-env/bin/python" -m pip install --upgrade torch==2.14.0 torchvision==0.29.0
"$PROJECT_ROOT/.cache/panseg-env/bin/python" "$PROJECT_ROOT/scripts/configure_panseg_runtime.py"
print 'PanSeg ready. First inference downloads model weights into .cache/panseg.'
