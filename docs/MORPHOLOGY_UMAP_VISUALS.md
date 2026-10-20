# Rotating cell-surface UMAPs

The README loop shows the same 180-cell coverage subset in both maps; both UMAPs
were fitted on all 744 included, QC-passing cells. The companion MP4 shows all
744 in each map. Every original cell surface makes one full rigid 360° turn over
six seconds. Centers, colors and isotropic display scales remain fixed throughout;
there is no morphing, layout movement, frame-dependent resizing or mesh smoothing.
Smooth normals affect lighting only. GIF loops indefinitely; MP4s contain one turn.

Traditional: nine training-standardized morphology features. Spatial: the frozen
95D spatial PCA sign-frame distance with its ambiguity fallback, which had the
lowest observed matched-validation neighbor geometry cost. UMAP uses each full
method distance, 30 neighbors, min_dist 0.30 and seed 42. Sphericity colors use the
same clipped 0.55–0.95 scale. These are exploratory geometry layouts, not cell types.

With the local source dataset, Atlas export and dependencies available:

```sh
.venv/bin/python -m pip install --target .cache/umap-render-deps --no-deps umap-learn==0.5.12 pynndescent==0.6.0
zsh scripts/render_morphology_animation.sh
```

Requires macOS, Metal, the Swift toolchain, ffmpeg and the existing Python analysis
environment. SceneKit renders original triangles offscreen. Intermediate frames,
coordinates, selected identities and source hashes stay under ignored
`results/morphology_umap/`; no Atlas data, frozen fits or benchmark results change.
Source microscopy: Mouse-Organoid-Cells-CBG, distributed with EmbedSeg.
