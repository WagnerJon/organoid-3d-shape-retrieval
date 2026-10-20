#!/bin/zsh
# Install the upstream headless PyTorch backend in a separate environment.
set -eu
PROJECT_ROOT="${0:A:h:h}"
FOCUS_COMMIT=033017f20234f33722baca7b94679283e8fa6d47
FOCUS_ENV="$PROJECT_ROOT/.cache/focus3d-env"
mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/.cache/focus3d/models"
if [[ ! -d "$PROJECT_ROOT/vendor/focus3d/.git" ]]; then
    git clone https://github.com/yu-lab-vt/FOCUS-3D.git "$PROJECT_ROOT/vendor/focus3d"
fi
if [[ "$(git -C "$PROJECT_ROOT/vendor/focus3d" rev-parse HEAD)" != "$FOCUS_COMMIT" ]]; then
    git -C "$PROJECT_ROOT/vendor/focus3d" fetch --depth 1 origin "$FOCUS_COMMIT"
    git -C "$PROJECT_ROOT/vendor/focus3d" checkout --detach "$FOCUS_COMMIT"
fi
if [[ ! -x "$FOCUS_ENV/bin/python" ]]; then
    "$PROJECT_ROOT/.venv/bin/python" -m venv "$FOCUS_ENV"
fi
"$FOCUS_ENV/bin/python" -m pip install 'torch==2.14.0' 'numpy==1.26.4' 'scipy==1.17.1' \
    'scikit-image==0.25.2' 'tifffile==2025.6.11' 'zarr==2.18.7' 'PyYAML==6.0.3' 'tqdm==4.67.1' 'imagecodecs==2024.12.30'
PYTHONPATH="$PROJECT_ROOT/vendor/focus3d/src" "$FOCUS_ENV/bin/python" -c 'from focus3d.segmentation.FOCUS3D import inference_win; print("FOCUS-3D headless inference ready")'
print 'Select a downloaded official FOCUS-3D checkpoint (.pth) in the app. Weights require access at https://huggingface.co/Qinghua-thu/FOCUS-3D'
