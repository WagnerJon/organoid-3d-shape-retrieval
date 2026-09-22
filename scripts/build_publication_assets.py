"""Copy a small, inspectable set of result figures and aggregate tables for GitHub."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs"

FIGURES = {
    "cell_size_across_stacks.png": "results/cpsam_v2_all_analysis/across_samples.png",
    "cell_size_shape_distributions.png": "results/cpsam_v2_all_analysis/distributions.png",
    "traditional_vs_context_phenospace.png": "results/organoid_phenotype_report/phenotype_spaces.png",
    "exploratory_groups.png": "results/organoid_phenotype_groups/phenotype_groups.png",
    "cell_count_overlay.png": "results/organoid_phenotype_groups/cell_count_overlay/cell_count_on_context_phenospace.png",
    "raw_shape_retrieval.png": "results/shape_retrieval/retrieval_comparison.png",
    "multiscale_retrieval.png": "results/shape_retrieval_multicell/multiscale_retrieval.png",
    "matched_3d_retrieval_scores.png": "results/shape_retrieval_multicell/matched_retrieval/matched_retrieval_scores.png",
}

TABLES = {
    "morphology_cohorts.csv": "results/cpsam_v2_all_analysis/cohort_summary.csv",
    "stack_morphology.csv": "results/cpsam_v2_all_analysis/sample_summary.csv",
    "exploratory_groups.csv": "results/organoid_phenotype_groups/group_assignments.csv",
    "matched_3d_retrieval.csv": "results/shape_retrieval_multicell/matched_retrieval/summary.csv",
    "matched_3d_inference.json": "results/shape_retrieval_multicell/matched_retrieval/paired_inference.json",
    "split_similarity.json": "results/shape_retrieval_multicell/matched_retrieval/split_similarity.json",
}


def main():
    for subdir, mapping in (("figures", FIGURES), ("tables", TABLES)):
        target_dir = DEST / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, source in mapping.items():
            src = ROOT / source
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copyfile(src, target_dir / name)
            print(f"{source} -> docs/{subdir}/{name}")


if __name__ == "__main__":
    main()
