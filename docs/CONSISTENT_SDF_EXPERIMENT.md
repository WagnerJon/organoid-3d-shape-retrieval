# Paired-surface consistency pilot

This experiment stays outside Organoid Atlas. It tests whether sampling consistency improves encoder embeddings without sacrificing surface reconstruction. It does not publish embeddings or alter the app or dataset.

## Controlled comparison

Both arms start from the completed staged encoder's best checkpoint and use its frozen shared SDF decoder. Both receive the same 96 training cells, same two independent 2048-point surface samples, same SO(3) augmentations, optimizer, minibatches and 12 fine-tuning epochs. The 48 validation cells remain in separate TIFF similarity groups. No test cells are loaded.

- **Control:** SDF reconstruction on both surface samples. Removes the preceding experiment's arbitrary fitted-vector regression target.
- **Paired:** same reconstruction plus normalized vector agreement (weight 0.01) and symmetric instance discrimination between the two samples' 128 invariant norm features (weight 0.02, temperature 0.02). Each four-cell minibatch supplies three negative cells per query. Both sample views share a rotation so vector agreement compares the same pose.

These are precomputed independent surface samples, not newly resampled clouds every epoch. A third sample is reserved for validation repeatability. This is an independently implemented, reduced-compute experiment, not a reproduction of the published Allen architecture.

## Safeguards and evaluation

The decoder remains frozen and gradients flow through it into the encoder. There is no per-validation-cell code optimization. Reconstruction provides shape supervision; the contrastive objective alone is not a mathematical guarantee against collapse. Effective rank and geometry are therefore evaluated alongside repeatability.

Both arms select their best checkpoint by validation SDF error, including their unchanged starting checkpoint. Final-epoch results are reported separately to expose tradeoffs rather than select a favorable consistency score. Geometry is measured on independent 2048-point surface samples from 56³ SDF reconstructions. Results include Chamfer error, physical surface error, F1 at 5% of normalization radius, volume error, surface failures, rotation equivariance, sampling identity retrieval, and a morphometry-controlled geometric retrieval proxy. Group bootstrap intervals reflect 11 validation groups.

This is one seed and a short fine-tuning budget on development validation data, not an independent biological benchmark. The existing shared decoder and starting encoder have already used these validation cells for model selection. No claim of biological information beyond morphometry follows from geometry alone.

## Run

```sh
.venv/bin/python scripts/consistent_sdf.py --device mps
.venv/bin/python scripts/evaluate_consistent_sdf.py --device mps
```

The experiment refuses to overwrite checkpoints or a completed evaluation. Source snapshots, hashes, training histories, metrics and visual comparisons are saved under `results/consistent_sdf`. Results are summarized in that directory's `report.md` after evaluation.

## Completed result

Both arms completed 12 epochs on Mac MPS (about 130 seconds each). At the fixed final epoch, paired consistency improved sampling identity retrieval from 7/48 (14.6%) to 15/48 (31.3%), while increasing squared Chamfer from 0.02913 to 0.04944 (about 70% worse) and decreasing surface F1 from 26.5% to 21.7%. Volume error increased from 21.0% to 33.1%. Both retained 100% rotation identity retrieval. Every extracted surface was closed and inside the extraction box.

The paired arm's best SDF checkpoint remained epoch zero; no update improved its starting validation SDF error. The group bootstrap interval for final-epoch geometry favors the control. Visual comparisons still show loss of distinctive cell shapes. This short pilot does not justify Atlas integration. The successful previous per-cell SDF fitting remains a useful reconstruction reference, but requires the known segmentation's SDF observations.

Three new consistency checks and the six existing staged-model checks passed. Atlas's dataset hash remains unchanged. Full results and limitations are in `results/consistent_sdf/report.md`.
