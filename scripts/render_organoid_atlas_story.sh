#!/bin/zsh
set -eu
PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"
source native/OrganoidExplorer/toolchain.sh
.venv/bin/python scripts/prepare_organoid_atlas_story.py
swiftc -O "${COMPILER_FLAGS[@]}" scripts/render_organoid_atlas_story.swift -o .cache/render_organoid_atlas_story
.cache/render_organoid_atlas_story results/morphology_umap/organoid_story/scene.json
ffmpeg -hide_banner -loglevel error -y -framerate 20 -i results/morphology_umap/organoid_story/frames/%04d.png -framerate 20 -i results/morphology_umap/organoid_story/overlays/%04d.png -filter_complex '[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]' -map '[v]' -frames:v 320 -an -c:v libx264 -preset slow -crf 20 -movflags +faststart docs/figures/organoid_to_shape_atlas.mp4
ffmpeg -hide_banner -loglevel error -y -i docs/figures/organoid_to_shape_atlas.mp4 -filter_complex 'fps=12,scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' -loop 0 docs/figures/organoid_to_shape_atlas.gif
