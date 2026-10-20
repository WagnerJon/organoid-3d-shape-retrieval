# Data provenance and artifact policy

The input stacks and supplied masks come from the [Mouse-Organoid-Cells-CBG release of EmbedSeg](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0). Download them from the source and cite the [EmbedSeg publication](https://arxiv.org/abs/2101.10033).

The public package contains project code, configs, reports, analytical plots, aggregate/statistical tables, and selected rendered reconstruction/retrieval figures and a viewer screenshot under `docs/figures/`. These illustrate results derived from Mouse-Organoid-Cells-CBG; they do not constitute new biological ground truth. Original TIFFs, supplied masks, predicted TIFF labels, per-cell crops, per-object embeddings, source meshes and model checkpoints/weights remain in the local ignored `results/` tree or are downloaded separately.

The reconstruction uses [u-Segment3D](https://github.com/DanuserLab/u-segment3D) at commit `33974282e436d456f4941a2c2b9a8c2d1d9ed2fb` and [Cellpose](https://github.com/MouseLand/cellpose) 4.2.1.1 with `cpsam_v2`. Upstream code is fetched separately. u-Segment3D states GPL-3.0 terms; Cellpose states BSD-3-Clause terms for code and separate restrictions for some model training data. Those terms do not establish rights to redistribute this particular organoid archive or weights.
