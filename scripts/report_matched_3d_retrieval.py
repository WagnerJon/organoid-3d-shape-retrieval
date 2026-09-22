"""Summarize count/volume-matched retrieval and link the actual 3D examples."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/shape_retrieval_multicell/matched_retrieval"


def main():
    summary = pd.read_csv(OUT / "summary.csv")
    inference = json.loads((OUT / "paired_inference.json").read_text())
    split = json.loads((OUT / "split_similarity.json").read_text())
    cases = json.loads((OUT / "visual_cases.json").read_text())
    detail = pd.read_csv(OUT / "query_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    levels = ["neighborhood", "organoid"]
    for ax, metric, label in zip(axes, ["top1_accuracy", "mean_rank_percentile"],
                                 ["Correct top match", "Mean rank percentile"]):
        student = [summary[(summary.level == level) & (summary.method == "raw_student")].iloc[0][metric] for level in levels]
        volume = [summary[(summary.level == level) & (summary.method == "volume_baseline")].iloc[0][metric] for level in levels]
        x = np.arange(2)
        ax.bar(x - .18, student, .35, label="Raw model", color="#2563eb")
        ax.bar(x + .18, volume, .35, label="Volume alone", color="#f59e0b")
        ax.set_xticks(x, ["Neighborhoods\n49 queries", "Stack fields\n8 queries"])
        ax.set_ylabel(label); ax.set_ylim(0, 1.05)
        for i, v in enumerate(student): ax.text(i - .18, v + .02, f"{v:.2f}", ha="center", fontsize=8)
        for i, v in enumerate(volume): ax.text(i + .18, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exact cell count; foreground volume within a factor of 1.25")
    fig.tight_layout(); fig.savefig(OUT / "matched_retrieval_scores.png", dpi=200); plt.close(fig)

    n = summary[(summary.level == "neighborhood") & (summary.method == "raw_student")].iloc[0]
    b = summary[(summary.level == "neighborhood") & (summary.method == "volume_baseline")].iloc[0]
    o = summary[(summary.level == "organoid") & (summary.method == "raw_student")].iloc[0]
    ob = summary[(summary.level == "organoid") & (summary.method == "volume_baseline")].iloc[0]
    blocks = []
    labels = ["Large neighborhood gain", "Five-cell neighborhood gain", "Neighborhood failure", "Whole stack field"]
    for name, case in zip(labels, cases):
        base = Path(case["figure"]).stem
        blocks.append(f"### {name}\n\n![{name}]({case['figure']})\n\n"
                      f"Rotatable 3D meshes: [query]({base}_query.glb), [raw model match]({base}_model.glb), [volume baseline match]({base}_volume.glb).")
    report = f'''# Count- and volume-matched 3D retrieval audit

## Question and method

Do the trained raw-image embeddings retrieve similar 3D cellular arrangements when the candidate fields have **exactly the same number of segmented cells** and **foreground volume within a factor of 1.25** of the query (80–125% of query volume)? Only fields from different image stacks are eligible. Queries with fewer than three eligible matches are excluded from the top-match calculation.

The evaluation target is the sorted list of all pairwise cell-centroid distances within each 3D field, divided by their root-mean-square distance. This removes absolute spatial scale and orientation. Lower target distance means a closer match in this particular measure of arrangement. It does not measure cell surface shape, contact topology or biology. The volume-only baseline ranks eligible candidates by the remaining difference in total foreground volume. Every reconstructed evaluation mask was checked voxel-for-voxel against the mask used by the trained model (IoU = 1.0).

## Results

| Field | Eligible queries | Raw model correct top match | Volume-only correct top match | Raw mean rank percentile | Volume mean rank percentile |
|---|---:|---:|---:|---:|---:|
| Local neighborhoods | {int(n.queries)} | {n.top1_accuracy:.3f} ({int(round(n.queries*n.top1_accuracy))}/{int(n.queries)}) | {b.top1_accuracy:.3f} ({int(round(b.queries*b.top1_accuracy))}/{int(b.queries)}) | {n.mean_rank_percentile:.3f} | {b.mean_rank_percentile:.3f} |
| Whole stack fields | {int(o.queries)} | {o.top1_accuracy:.3f} ({int(round(o.queries*o.top1_accuracy))}/{int(o.queries)}) | {ob.top1_accuracy:.3f} ({int(round(ob.queries*ob.top1_accuracy))}/{int(ob.queries)}) | {o.mean_rank_percentile:.3f} | {ob.mean_rank_percentile:.3f} |

The neighborhood gain is {inference['neighborhood']['top1_hit']['student_minus_volume_baseline']:.3f} in top-match accuracy. A paired sign-flip test at the image-stack level gives p={inference['neighborhood']['top1_hit']['paired_sample_signflip_p_two_sided']:.3f} across {inference['neighborhood']['top1_hit']['independent_query_organoids']} query stacks. This is suggestive but not decisive. The whole-field comparison uses only eight queries, all with six cells, and cannot support a general conclusion about larger organoids.

The raw panels below are maximum-intensity projections from the actual held-out 3D TIFF crops. The colored surfaces come from the corresponding CPSAM-v2 instance masks resampled into each field and are provided as rotatable GLB files. Colors distinguish cells within a panel; they do not assert a cell-to-cell match across panels. Examples were chosen to show two wins, one failure, and one whole-field case. Quantitative results above use every eligible query.

{chr(10).join(blocks)}

## Independence audit

The nominal test split is at the image-stack level. A pixel-aligned downsampled full-TIFF comparison found that the median test stack has Pearson correlation **{split['test']['median_nearest_training_correlation']:.3f}** with its most similar training stack; **{split['test']['above_0_9']}/{split['test']['stacks']}** test stacks exceed 0.9. This strongly suggests related image fields across splits, although it does not establish whether they are the same biological specimen. The examples also show some stacks contain separated cell clusters, so one stack is not guaranteed to be one organoid.

These scores describe retrieval among the observed stacks. They must not be treated as independent-organism generalization. The next proper benchmark needs acquisition/specimen identifiers, or at minimum image-similarity groups kept entirely within one split, followed by retraining and reevaluation. Image similarity grouping is only a proxy for specimen identity.

## Files

- `query_results.csv`: every eligible query and its retrieved match.
- `summary.csv`: grouped retrieval metrics.
- `paired_inference.json`: stack-clustered paired uncertainty and sign-flip test.
- `target_integrity.csv`: reconstructed instance-count checks.
- `nearest_training_stack.csv`: cross-split full-image similarity audit.
'''
    (OUT / "REPORT.md").write_text(report)
    print(report[:2500])


if __name__ == "__main__": main()
