#!/bin/zsh
set -eu
WORKER_ROOT="${0:A:h:h}"
if [[ "$(uname -m)" != arm64 ]]; then
    print 'This worker package targets Apple Silicon Macs.'
    exit 1
fi
mkdir -p "$WORKER_ROOT/.cache/worker-tools"
export UV_PYTHON_INSTALL_DIR="$WORKER_ROOT/.cache/python"
export UV_CACHE_DIR="$WORKER_ROOT/.cache/uv"
if [[ ! -x "$WORKER_ROOT/.cache/worker-tools/uv" ]]; then
    /usr/bin/curl --fail --location --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh -o "$WORKER_ROOT/.cache/worker-tools/install-uv.sh"
    UV_UNMANAGED_INSTALL="$WORKER_ROOT/.cache/worker-tools" /bin/sh "$WORKER_ROOT/.cache/worker-tools/install-uv.sh"
fi
WORKER_UV="$WORKER_ROOT/.cache/worker-tools/uv"
"$WORKER_UV" python install --no-bin 3.12
if [[ ! -x "$WORKER_ROOT/.venv/bin/python" ]]; then
    "$WORKER_UV" venv --managed-python --python 3.12 "$WORKER_ROOT/.venv"
fi
/usr/bin/awk '$0 !~ /^scikit-fmm==/' "$WORKER_ROOT/requirements-lock.txt" > "$WORKER_ROOT/.cache/worker-tools/runtime-requirements.txt"
"$WORKER_UV" pip install --python "$WORKER_ROOT/.venv/bin/python" -r "$WORKER_ROOT/.cache/worker-tools/runtime-requirements.txt"
"$WORKER_UV" pip install --python "$WORKER_ROOT/.venv/bin/python" --no-deps "$WORKER_ROOT/vendor/dinov3"
"$WORKER_ROOT/.venv/bin/python" -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[1])/"scripts")); from atlas_worker import engine; print("Worker runtime ready:",engine())' "$WORKER_ROOT"
