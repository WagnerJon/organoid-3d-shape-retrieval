# Data provenance and artifact policy

The input stacks and supplied masks come from the [Mouse-Organoid-Cells-CBG release of EmbedSeg](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0). Download them from the source and cite the [EmbedSeg publication](https://arxiv.org/abs/2101.10033).

The public package contains project code, configs, a report, analytical plots, and aggregate/statistical tables. It excludes original TIFFs, supplied masks, predicted TIFF labels, per-cell crops, per-object embeddings, model checkpoints/weights, and rendered raw-image or mesh retrieval examples. Those remain in the local ignored `results/` tree and can be regenerated after downloading the source data.

The reconstruction uses [u-Segment3D](https://github.com/DanuserLab/u-segment3D) at commit `33974282e436d456f4941a2c2b9a8c2d1d9ed2fb` and [Cellpose](https://github.com/MouseLand/cellpose) 4.2.1.1 with `cpsam_v2`. Upstream code is fetched separately. u-Segment3D states GPL-3.0 terms; Cellpose states BSD-3-Clause terms for code and separate restrictions for some model training data. Those terms do not establish rights to redistribute this particular organoid archive or weights.
