"""Export compact public summaries and a figure from the completed three-seed run."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/shape_metric"
TABLES = ROOT / "docs/tables"
FIGURES = ROOT / "docs/figures"


def paired(result, level, target, competitor, measure):
    x = result[(result.level == level) & (result.target == target)]
    wide = x.pivot_table(index=["query", "similarity_group"], columns=["method", "seed"], values=measure)
    difference = wide["shape_metric"].mean(axis=1) - wide[competitor].mean(axis=1)
    if measure == "target_distance":
        difference = -difference  # Positive always means the shape-metric model is better.
    groups = difference.groupby(level="similarity_group").mean().to_numpy()
    observed = abs(groups.mean())
    null = np.array([abs(np.dot(groups, signs) / len(groups))
        for signs in itertools.product([-1, 1], repeat=len(groups))])
    return {"level": level, "target": target, "competitor": competitor, "measure": measure,
        "queries": len(difference), "groups": len(groups),
        "query_weighted_difference": float(difference.mean()),
        "equal_group_difference": float(groups.mean()),
        "exact_two_sided_p": float(np.mean(null >= observed - 1e-12))}


def main():
    seeds = json.loads((ROOT / "config/shape_metric.json").read_text())["seeds"]
    for seed in seeds:
        status = json.loads((OUT / f"seed_{seed}/status.json").read_text())
        if status["status"] != "complete":
            raise RuntimeError(f"Seed {seed} is not complete")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = []; queries = []
    for seed in seeds:
        s = pd.read_csv(OUT / f"seed_{seed}/test_retrieval_summary.csv")
        q = pd.read_csv(OUT / f"seed_{seed}/test_retrieval_queries.csv")
        s["seed"] = seed; q["seed"] = seed
        summary.append(s); queries.append(q)
    s = pd.concat(summary, ignore_index=True)
    q = pd.concat(queries, ignore_index=True)
    aggregate = s.groupby(["level", "target", "method"]).agg(
        queries=("queries", "first"), top1_mean=("top1_accuracy", "mean"),
        top1_min=("top1_accuracy", "min"), top1_max=("top1_accuracy", "max"),
        mean_rank_percentile=("mean_rank_percentile", "mean"),
        mean_target_distance=("mean_target_distance", "mean"),
        median_candidates=("median_candidates", "first")).reset_index()
    aggregate.to_csv(TABLES / "shape_metric_test_summary.csv", index=False)
    comparisons = []
    for level, target in [("single", "shape"), ("neighborhood", "shape"),
                          ("neighborhood", "arrangement"), ("neighborhood", "combined"),
                          ("organoid", "combined")]:
        for competitor in ["boundary_only", "untrained_raw", "volume_only"]:
            for measure in ["top1_hit", "rank_percentile", "target_distance"]:
                comparisons.append(paired(q, level, target, competitor, measure))
    pd.DataFrame(comparisons).to_csv(TABLES / "shape_metric_paired_comparisons.csv", index=False)

    levels = [("single", "shape", "Single-cell shape"),
              ("neighborhood", "combined", "Neighborhood shape + arrangement")]
    methods = [("shape_metric", "Shape metric", "#24608C"),
               ("boundary_only", "Boundary only", "#8A99A8"),
               ("untrained_raw", "Untrained raw", "#C18B3B"),
               ("volume_only", "Volume only", "#64876A")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (level, target, title) in zip(axes, levels):
        values = [aggregate[(aggregate.level == level) & (aggregate.target == target) &
                   (aggregate.method == method)].iloc[0] for method, _, _ in methods]
        for i, ((method, label, color), row) in enumerate(zip(methods, values)):
            ax.bar(i, row.top1_mean, color=color, width=.7)
            text_y = row.top1_mean
            if method == "shape_metric":
                seed_values = s[(s.level == level) & (s.target == target) &
                                (s.method == method)].top1_accuracy.to_numpy()
                ax.scatter(np.linspace(i-.14, i+.14, len(seed_values)), seed_values,
                           color="black", marker="o", s=23, zorder=4)
                text_y = max(text_y, float(seed_values.max()))
            ax.text(i, text_y + .025, f"{row.top1_mean:.1%}", ha="center", fontsize=9)
        ax.set_xticks(range(len(methods)), [label.replace(" ", "\n") for _, label, _ in methods])
        ax.set_title(f"{title}\n{int(values[0].queries)} queries")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Correct closest shape match (top-1)")
    axes[0].set_ylim(0, .75)
    fig.suptitle("Grouped test retrieval across three shape-metric training seeds", fontsize=12)
    fig.text(.02, .01, "Candidates: same cell count, volume within 80–125%, different image-similarity group. Dots are three seeds; controls are fixed.", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .05, 1, .92])
    fig.savefig(FIGURES / "shape_metric_test.png", dpi=180)
    print(aggregate[((aggregate.level == "single") & (aggregate.target == "shape")) |
        ((aggregate.level == "neighborhood") & (aggregate.target == "combined"))].to_string(index=False))


if __name__ == "__main__":
    main()
