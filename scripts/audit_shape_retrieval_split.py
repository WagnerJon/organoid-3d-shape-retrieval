"""Audit cross-split similarity between the original TIFF stacks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/shape_retrieval_multicell/matched_retrieval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "results/shape_retrieval/manifest.csv").drop_duplicates("sample").sort_values("sample")
    vectors = []
    for _, row in frame.iterrows():
        image = tifffile.imread(row.raw_image).astype(np.float32)[::4, ::8, ::8].ravel()
        vectors.append((image - image.mean()) / max(float(image.std()), 1e-6))
    features = np.stack(vectors)
    corr = features @ features.T / features.shape[1]
    samples = frame["sample"].to_numpy(); splits = frame["split"].to_numpy()
    train = np.flatnonzero(splits == "train")
    rows = []
    for i, (sample, split) in enumerate(zip(samples, splits)):
        if split not in {"test", "validation"}:
            continue
        match = train[np.argmax(corr[i, train])]
        rows.append(dict(sample=sample, split=split, nearest_train=samples[match],
                         full_stack_pearson=float(corr[i, match])))
    result = pd.DataFrame(rows); result.to_csv(OUT / "nearest_training_stack.csv", index=False)
    summary = {split: dict(stacks=int(len(g)), median_nearest_training_correlation=float(g.full_stack_pearson.median()),
                           minimum=float(g.full_stack_pearson.min()), maximum=float(g.full_stack_pearson.max()),
                           above_0_9=int((g.full_stack_pearson > .9).sum()))
               for split, g in result.groupby("split")}
    summary["method"] = "Pearson correlation of TIFF stacks downsampled by [4,8,8], pixel aligned, without registration"
    summary["interpretation"] = "High correlation suggests related fields but does not prove the same biological specimen"
    (OUT / "split_similarity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
