# Publishing this project on GitHub

The repository has been prepared for a small public GitHub upload. It contains code, configs, tests, this report, analytical figures and summary tables. It deliberately excludes the 9+ GB local workspace outputs, source TIFFs/masks, meshes, embeddings, checkpoints, and upstream code. Review the [data policy](DATA.md) and [report limitations](REPORT.md) before publication.

No GitHub remote or commit existed in the local checkout when this package was assembled. Once a GitHub account is available to the publishing environment and GitHub CLI is installed/authenticated, run from the repository root:

```bash
git add .gitignore README.md VALIDATION.md config docs requirements-lock.txt scripts tests
git diff --cached --stat
git commit -m "Publish 3D organoid reconstruction and representation experiments"
gh repo create organoid-3d-shape-retrieval --public --source=. --remote=origin --push
```

Inspect the staged file list before committing; it should contain no source TIFFs, original masks, model weights, or large `results/` files. The selected repository name is a suggestion and can be changed in the final command. A public repository does not automatically grant others reuse rights: this project currently has no explicit code license. The owner should choose one after considering the separately licensed dependencies, rather than assuming the upstream u-Segment3D or Cellpose licenses automatically apply to this project's own code.
