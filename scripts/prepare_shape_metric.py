"""Build small predicted-mask shape targets; reuse boundary experiment raw volumes."""
from pathlib import Path
import json
from shape_metric_common import ROOT, make_targets


def main():
    cfg = json.loads((ROOT / "config/shape_metric.json").read_text())
    dataset = ROOT / cfg["dataset_dir"]
    output = ROOT / cfg["output_dir"]
    if not (dataset / "prepare_complete.json").exists():
        raise RuntimeError("Boundary-shape dataset must be prepared first")
    frame, *_ = make_targets(dataset, output)
    print(f"Prepared shape targets for {len(frame)} reused raw-image fields in {output}", flush=True)


if __name__ == "__main__":
    main()
