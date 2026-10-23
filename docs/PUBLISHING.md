# GitHub repository and updates

The project is published at [WagnerJon/organoid-atlas](https://github.com/WagnerJon/organoid-atlas). It contains code, configs, tests, the report, analytical figures and summary tables. It deliberately excludes the large local workspace outputs, source TIFFs/masks, meshes, embeddings, checkpoints, and upstream code. Review the [data policy](DATA.md) and [report limitations](REPORT.md) before sharing additional artifacts.

The former `WagnerJon/organoid-3d-shape-retrieval` repository URL redirects to this repository. Do not reuse that repository name under the same account, as doing so would remove the redirect. GitHub Pages uses https://wagnerjon.github.io/organoid-atlas/; the former Pages URL is not redirected. The existing Zenodo DOI remains unchanged.

The included synthetic demo is an exception to the research-output exclusions: `native/OrganoidExplorer/Resources/DemoShapes.atlasdemo` contains generated shapes, synthetic images and precomputed demonstration results. It contains no experimental microscopy or model weights. `.zenodo.json` supplies release metadata, including demo availability, to the GitHub–Zenodo integration. Publish a GitHub release to archive a new version under the existing Zenodo concept DOI.

For later changes, run from the repository root:

```bash
git add README.md VALIDATION.md config docs native requirements-lock.txt scripts tests
git diff --cached --stat
git commit -m "Update organoid analysis"
git push
```

Inspect the staged file list before committing; it should contain no source TIFFs, original masks, model weights, or large `results/` files. A public repository does not automatically grant others reuse rights: this project currently has no explicit code license. The owner should choose one after considering the separately licensed dependencies, rather than assuming the upstream u-Segment3D or Cellpose licenses automatically apply to this project's own code.
