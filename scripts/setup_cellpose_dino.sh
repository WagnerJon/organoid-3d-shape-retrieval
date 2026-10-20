#!/bin/zsh
set -eu
PROJECT_ROOT="${0:A:h:h}"
DINO_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
if [[ ! -d "$PROJECT_ROOT/vendor/dinov3/.git" ]]; then
    git clone https://github.com/facebookresearch/dinov3.git "$PROJECT_ROOT/vendor/dinov3"
fi
if [[ "$(git -C "$PROJECT_ROOT/vendor/dinov3" rev-parse HEAD)" != "$DINO_COMMIT" ]]; then
    git -C "$PROJECT_ROOT/vendor/dinov3" fetch --depth 1 origin "$DINO_COMMIT"
    git -C "$PROJECT_ROOT/vendor/dinov3" checkout --detach "$DINO_COMMIT"
fi
# Cellpose inference only needs DINO's PyTorch backbone. Preserve the existing
# reconstruction environment instead of installing DINO's training extras.
"$PROJECT_ROOT/.venv/bin/python" -m pip install --no-deps "$PROJECT_ROOT/vendor/dinov3"
"$PROJECT_ROOT/.venv/bin/python" -c 'from cellpose.vit import dinov3_vitl16, dinov3_vitb16; print("Both Cellpose DINO backbones ready")'
