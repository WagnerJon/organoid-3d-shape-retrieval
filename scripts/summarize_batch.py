"""Collect completed reconstruction metrics without altering predictions."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.folder.glob("run_*/*/complete.json")):
        result = json.loads(path.read_text())
        settings = json.loads((path.parent / "settings.json").read_text())["config"]
        metrics = result["instance_metrics"]
        rows.append(dict(sample=path.parent.name, model=settings["model"],
                         reference_cells=result["reference_cells"], predicted_cells=result["predicted_cells"],
                         foreground_dice=result["foreground_dice"],
                         **{key: metrics[key] for key in ["precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]},
                         seconds=result["seconds"], output=str(path.parent.resolve())))
    rows.sort(key=lambda row: row["sample"])
    if not rows:
        raise ValueError("No completed reconstructions")
    with (args.folder / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Cellpose-SAM v2 reconstruction results", "",
             "Predictions use image data only. Existing annotations were loaded after reconstruction for comparison, not training, foreground constraints, or parameter selection.", "",
             "Voxel spacing: Z/Y/X = 1 / 0.1733 / 0.1733 µm (dataset literature). Working scale: 0.5; model: cpsam_v2; diameter: 30; Apple MPS enabled.", "",
             "Instance scores use one-to-one matching at IoU ≥ 0.5. These are per-image comparisons, not a held-out benchmark.", "",
             "| Stack | Reference cells | Predicted cells | Dice | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['sample']} | {row['reference_cells']} | {row['predicted_cells']} | {row['foreground_dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")
    lines.extend(["", "## Outputs", ""])
    for row in rows:
        out = Path(row["output"])
        lines.append(f"- {row['sample']}: [3D preview]({out / 'preview.png'}), [slice comparison]({out / 'comparison.png'}), [native labels]({out / 'labels_native.tif'}), [meshes and measurements]({out / 'meshes'}).")
    lines.extend(["", "Reference masks remain unchanged. Cells touching the image boundary describe cropped shapes. Surface and volume measurements depend on the dataset-derived calibration."])
    (args.folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
