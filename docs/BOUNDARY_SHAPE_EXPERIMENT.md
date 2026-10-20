# Boundary-aware raw 3D shape experiment

This experiment follows the [earlier project report](REPORT.md). Its purpose was to make the raw-image embedding sensitive to individual 3D cell surfaces and multicell arrangement while testing on image-similarity-grouped stacks. **The completed run did not demonstrate a retrieval improvement over an untrained raw encoder.**

The existing dataset yields 1,670 fields: single cells, local neighborhoods, and complete image-stack fields. For each field, preparation resamples the original raw TIFF at **96³** within the same 40/48/64 µm physical fields of view as the previous experiment. The training targets are computed from **predicted Cellpose/u-Segment3D instance labels**, retaining individual cell boundaries rather than collapsing all cells into one foreground mask. The supplied EmbedSeg annotations are not training targets. Raw intensity is the model's only inference input; this is segmentation-guided representation learning, not pure self-supervision.

The encoder has a 128-dimensional global embedding and a lightweight dense decoder. The decoder predicts signed distance to the nearest instance boundary, foreground occupancy, and a boundary map. Training also aligns two independently brightness/noise-augmented XY-transformed views and ranks geometrically different examples with the same field level, cell count, and similar foreground volume. We restrict geometric views to XY rotations/reflections because the microscope's Z resolution differs substantially from XY. The distance-map and dense losses retain boundary detail; the ranking loss acts directly on the retrieval embedding.

Before training, pixel-aligned downsampled TIFF correlations are linked at **Pearson ≥0.90**; every connected component stays within one split. The resulting allocation is **74 train, 17 validation, 17 test stacks**, and its maximum observed cross-split correlation is **0.899** under this particular comparison. Image similarity is only a proxy for specimen/acquisition identity. Test labels are used for evaluation only. At test time, retrieval compares fields with equal cell count, foreground volume within 80–125%, and a different stack. Single-cell target similarity uses five conventional predicted-mask shape features; multicell target similarity uses normalized 3D cell-centroid distances. Raw-model, volume-only, and untrained-encoder rankings are reported, with paired inference clustered by similarity group.

Run the preparation and training pipeline:

```bash
.venv/bin/python -u scripts/prepare_boundary_shape.py
.venv/bin/python scripts/start_boundary_shape.py
```

Preparation and training logs/status live in ignored `results/boundary_shape/`. In a second Terminal window:

```bash
tail -f results/boundary_shape/prepare.log
tail -f results/boundary_shape/pipeline.log
cat results/boundary_shape/pipeline_status.json
```

The full run is configured for 20 epochs on Apple MPS in [`config/boundary_shape.json`](../config/boundary_shape.json). Training can resume from `results/boundary_shape/last.pt`. Its quantitative outcome should be judged against the earlier matched retrieval audit only with the changed split and targets clearly stated; for a direct model comparison, the former architecture should also be retrained on this new grouped split. Biological generalization still requires true specimen/acquisition metadata or newly collected independent organoids.

## Completed result

The best validation checkpoint was epoch 19. Weighted validation loss fell from 0.500 at epoch 1 to 0.239, and the individual signed-distance, occupancy, and boundary prediction losses all fell. This shows that the model learned the dense segmentation-derived tasks. It does **not** establish that the global 128-dimensional vector learned better shape similarity.

| Test field | Eligible queries | Trained raw top-1 | Untrained raw top-1 | Volume-only top-1 |
|---|---:|---:|---:|---:|
| Single cell | 165 | 32/165 (19.4%) | 38/165 (23.0%) | 13/165 (7.9%) |
| Multicell neighborhood | 46 | 24/46 (52.2%) | 23/46 (50.0%) | 19/46 (41.3%) |
| Whole image-stack field | 8 | 5/8 (62.5%) | 6/8 (75.0%) | 5/8 (62.5%) |

The neighborhood advantage over the untrained encoder is one query, while the single-cell and whole-field scores favor the untrained encoder. Mean target-distance and rank-percentile comparisons also fail to show a consistent trained-encoder gain. An exact two-sided sign-flip calculation across the 11/8/4 image-similarity groups respectively gives p=0.563/0.625/0.500 for trained versus untrained single/neighborhood/whole-field top-1 results. These small, dependent test sets cannot support a strong superiority claim. The volume baseline is weaker for single cells, but that alone does not establish that training added useful information beyond the raw encoder architecture.

An objective mismatch plausibly contributed. For one-cell training examples, the ranking target reduces to the **axis ratio alone**: relative volume is always one, and there are no pairwise centroid distances. Evaluation instead uses five standardized surface and shape descriptors. In addition, the dense decoder can use spatial skip connections, so a low reconstruction loss need not force the pooled embedding to encode the boundaries. These are mechanism hypotheses, not proven causes of the observed scores. A better next experiment would align the embedding-level loss with the evaluated shape descriptors and multicell geometry, use a bottleneck that requires the embedding for boundary prediction, and select checkpoints on held-out retrieval rather than dense loss alone.

The split groups stacks with pixel-aligned TIFF correlation at or above 0.90: 74 train, 17 validation, 17 test; the maximum measured cross-split correlation is 0.899. This is more cautious than the earlier nominal stack-ID split, where test-to-training similarity was high. Because the split and retrieval targets changed, the earlier 69.4% neighborhood top-1 score and this run's 52.2% are **not directly comparable**. True independent biological validation remains unavailable without specimen/acquisition IDs. “Whole image-stack field” also should not be interpreted as a confirmed whole organoid.

The generated [query-level results](../results/boundary_shape/retrieval_queries.csv), [score summary](../results/boundary_shape/retrieval_summary.csv), and [comparison figure](../results/boundary_shape/analysis_comparison.png) are local artifacts under the ignored `results/` directory; the accompanying [automatic report](../results/boundary_shape/REPORT.md) contains the complete run metrics.
