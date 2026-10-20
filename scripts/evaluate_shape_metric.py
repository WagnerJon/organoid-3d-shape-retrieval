"""Matched test retrieval for shape-metric training and fixed raw controls."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
import torch

from train_boundary_shape import ShapeEncoder
from train_shape_metric import embed
from shape_metric_common import ROOT, matched_retrieval


def exact_group_signflip(result, target, level, competitor):
    subset = result[(result.target == target) & (result.level == level)]
    wide = subset.pivot(index=["query", "similarity_group"], columns="method", values="top1_hit")
    if wide.empty:
        return None
    delta = wide["shape_metric"] - wide[competitor]
    group = delta.groupby(level="similarity_group").mean().to_numpy()
    observed = abs(group.mean())
    if len(group) > 20:
        raise ValueError("Exact sign flip would be too large")
    values = np.array([abs(np.dot(group, signs) / len(group))
        for signs in itertools.product([-1, 1], repeat=len(group))])
    return {"queries": int(len(delta)), "groups": int(len(group)),
        "query_weighted_difference": float(delta.mean()),
        "equal_group_difference": float(group.mean()),
        "exact_two_sided_p": float(np.mean(values >= observed - 1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/shape_metric.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    dataset = ROOT / cfg["dataset_dir"]
    artifacts = ROOT / cfg["output_dir"]
    out = artifacts / f"seed_{cfg['seed']}"
    checkpoint_path = out / "best.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("No retrieval-selected checkpoint; finish training first")
    torch.set_num_threads(4)
    device = torch.device(cfg["device"])
    frame = pd.read_csv(dataset / "manifest.csv").set_index("index", drop=False)
    groups = pd.read_csv(dataset / "split_groups.csv").set_index("sample")
    frame["similarity_group"] = frame["sample"].map(groups.similarity_group)
    selected = frame[frame.split == args.split].copy().reset_index(drop=True)
    ids = selected["index"].to_numpy(int)
    raw = np.load(dataset / "raw96.npy", mmap_mode="r")
    array = np.load(artifacts / "metric_targets.npy", mmap_mode="r")
    lengths = pd.read_csv(artifacts / "target_lengths.csv").set_index("index")
    targets = {name: [] for name in ["shape", "arrangement", "combined"]}
    for i in ids:
        shape_len = int(lengths.loc[i, "shape_length"])
        arrangement_len = int(lengths.loc[i, "arrangement_length"])
        combined = np.array(array[i, :shape_len + arrangement_len], copy=True)
        if arrangement_len:
            shape = combined[:shape_len] * np.sqrt(2)
            arrangement = combined[shape_len:] * np.sqrt(2)
        else:
            shape = combined; arrangement = np.array([], np.float32)
        targets["shape"].append(shape)
        targets["arrangement"].append(arrangement)
        targets["combined"].append(combined)
    model = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(trained["model"])
    old = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    old_checkpoint = torch.load(dataset / "best.pt", map_location="cpu", weights_only=False)
    old.load_state_dict(old_checkpoint["model"])
    old_cfg = json.loads((ROOT / "config/boundary_shape.json").read_text())
    torch.manual_seed(old_cfg["seed"] + 1)
    untrained = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    embeddings = {
        "shape_metric": embed(model, ids, raw, device),
        "boundary_only": embed(old, ids, raw, device),
        "untrained_raw": embed(untrained, ids, raw, device),
    }
    result = matched_retrieval(selected, embeddings, targets,
        ["shape_metric", "boundary_only", "untrained_raw", "volume_only"])
    if result.empty:
        raise RuntimeError("No eligible matched retrieval queries")
    prefix = "test" if args.split == "test" else "validation"
    result.to_csv(out / f"{prefix}_retrieval_queries.csv", index=False)
    summary = result.groupby(["level", "target", "method"]).agg(queries=("query", "size"),
        top1_accuracy=("top1_hit", "mean"), mean_rank_percentile=("rank_percentile", "mean"),
        mean_target_distance=("target_distance", "mean"), median_candidates=("candidates", "median")).reset_index()
    summary.to_csv(out / f"{prefix}_retrieval_summary.csv", index=False)
    comparisons = {}
    for (level, target), _ in result.groupby(["level", "target"]):
        comparisons[f"{level}:{target}"] = {competitor: exact_group_signflip(result, target, level, competitor)
            for competitor in ["boundary_only", "untrained_raw", "volume_only"]}
    (out / f"{prefix}_paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(f"{args.split.upper()} EVALUATION COMPLETE", flush=True)
    if args.split == "test":
        (out / "status.json").write_text(json.dumps({"status": "complete", "checkpoint_epoch": trained["epoch"],
            "test_stacks": int(selected["sample"].nunique())}, indent=2))


if __name__ == "__main__":
    main()
