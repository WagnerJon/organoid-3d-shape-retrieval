# GitHub repository and updates

The project is published at [WagnerJon/organoid-3d-shape-retrieval](https://github.com/WagnerJon/organoid-3d-shape-retrieval). It contains code, configs, tests, the report, analytical figures and summary tables. It deliberately excludes the large local workspace outputs, source TIFFs/masks, meshes, embeddings, checkpoints, and upstream code. Review the [data policy](DATA.md) and [report limitations](REPORT.md) before sharing additional artifacts.

For later changes, run from the repository root:

```bash
git add README.md VALIDATION.md config docs requirements-lock.txt scripts tests
git diff --cached --stat
git commit -m "Update organoid analysis"
git push
```

Inspect the staged file list before committing; it should contain no source TIFFs, original masks, model weights, or large `results/` files. A public repository does not automatically grant others reuse rights: this project currently has no explicit code license. The owner should choose one after considering the separately licensed dependencies, rather than assuming the upstream u-Segment3D or Cellpose licenses automatically apply to this project's own code.
