#!/bin/zsh
# Original surfaces, fixed UMAPs, one full rigid turn per six-second loop.
set -eu
PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"
source native/OrganoidExplorer/toolchain.sh
if [[ ! -f results/morphology_umap/provenance.json ]]; then
    .venv/bin/python scripts/render_morphology_umap.py
fi
swiftc -O "${COMPILER_FLAGS[@]}" scripts/render_umap_turntable.swift -o .cache/render_umap_turntable
for mode in showcase all_cells; do
    if [[ "$mode" == all_cells ]]; then
        .venv/bin/python scripts/animate_morphology_umap.py --all-cells
        rate=12; count=72; suffix=_all_cells
    else
        .venv/bin/python scripts/animate_morphology_umap.py
        rate=20; count=120; suffix=''
    fi
    folder="results/morphology_umap/animation/$mode"
    .cache/render_umap_turntable "$folder/scene.json"
    ffmpeg -hide_banner -loglevel error -y -framerate "$rate" -i "$folder/frames/%04d.png" -i "$folder/overlay.png" -filter_complex '[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]' -map '[v]' -frames:v "$count" -an -c:v libx264 -preset slow -crf 20 -movflags +faststart "docs/figures/rotating_morphology_umap${suffix}.mp4"
done
ffmpeg -hide_banner -loglevel error -y -i docs/figures/rotating_morphology_umap.mp4 -filter_complex 'fps=12,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' -loop 0 docs/figures/rotating_morphology_umap.gif
