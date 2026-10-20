"""Train raw 3D embeddings against multi-descriptor cell and arrangement similarity."""
from pathlib import Path
import argparse
import json
import math
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from train_boundary_shape import ShapeEncoder, matched_batches, random_xy_view
from shape_metric_common import ROOT, matched_retrieval


def shape_metric_loss(embedding, target, length, cfg):
    """Match the within-batch neighbor distribution of predicted-mask shape targets."""
    if len(embedding) < 3:
        return embedding.sum() * 0
    valid = int(length[0])
    if not np.all(length == valid):
        raise ValueError("Metric batch must have equal cell counts and target lengths")
    truth = torch.cdist(target[:, :valid], target[:, :valid])
    diagonal = torch.eye(len(embedding), device=embedding.device, dtype=torch.bool)
    teacher_logits = (-truth / cfg["metric_teacher_temperature"]).masked_fill(diagonal, -1e4)
    z = F.normalize(embedding, dim=1)
    student_logits = (z @ z.T / cfg["metric_student_temperature"]).masked_fill(diagonal, -1e4)
    teacher = F.softmax(teacher_logits.detach(), dim=1)
    return F.kl_div(F.log_softmax(student_logits, dim=1), teacher, reduction="batchmean")


def losses(model, raw, sdf, labels, targets, lengths, fov, cfg, rng, training):
    dense_target = torch.cat([sdf, (labels > 0).float()], 1)
    if training:
        first, dense_target = random_xy_view(raw, dense_target, cfg, rng)
        second, _ = random_xy_view(raw, None, cfg, rng)
    else:
        first, second = raw, raw
    embedding, prediction = model(first)
    truth_sdf, foreground = dense_target[:, :1], dense_target[:, 1:2]
    clip = cfg["sdf_clip_um"]
    pitch = torch.as_tensor(fov / raw.shape[-1] / clip, dtype=torch.float32, device=raw.device).view(-1, 1, 1, 1, 1)
    boundary = (truth_sdf.abs() <= 1.1 * pitch).float()
    near_weight = 1 + 4 * torch.exp(-truth_sdf.abs() * clip / 1.5)
    sdf_loss = (F.smooth_l1_loss(torch.tanh(prediction[:, :1]), truth_sdf, reduction="none") * near_weight).sum() / near_weight.sum()
    occupancy_loss = F.binary_cross_entropy_with_logits(prediction[:, 1:2], foreground, pos_weight=torch.tensor(8., device=raw.device))
    boundary_loss = F.binary_cross_entropy_with_logits(prediction[:, 2:3], boundary, pos_weight=torch.tensor(8., device=raw.device))
    metric_loss = shape_metric_loss(embedding, targets, lengths, cfg)
    if training:
        other, _ = model(second, decode=False)
        view_loss = (1 - (F.normalize(embedding, dim=1) * F.normalize(other, dim=1)).sum(1)).mean()
    else:
        view_loss = embedding.sum() * 0
    total = (cfg["loss_metric"] * metric_loss + cfg["loss_sdf"] * sdf_loss +
        cfg["loss_occupancy"] * occupancy_loss + cfg["loss_boundary"] * boundary_loss +
        cfg["loss_view"] * view_loss)
    return [total, metric_loss, sdf_loss, occupancy_loss, boundary_loss, view_loss]


def embed(model, ids, raw, device):
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(ids), 4):
            batch_ids = ids[start:start + 4]
            x = torch.from_numpy(np.array(raw[batch_ids], copy=True)).unsqueeze(1).float().to(device)
            h, _ = model(x, decode=False)
            vectors.append(h.cpu().numpy())
    return np.concatenate(vectors)


def validation_retrieval(frame, raw, target_array, target_length, model, device):
    validation = frame[frame.split == "validation"].copy().reset_index(drop=True)
    ids = validation["index"].to_numpy(int)
    vector = embed(model, ids, raw, device)
    target = [np.array(target_array[i, :target_length[i]], copy=True) for i in ids]
    rows = matched_retrieval(validation, {"shape_metric": vector},
        {"shape": target, "arrangement": target, "combined": target}, ["shape_metric"])
    # The separate target modes are assembled below by the evaluator. For checkpoint
    # choice, use the combined shape/arrangement target, including single-cell shape.
    chosen = rows[((rows.level == "single") & (rows.target == "shape")) |
                  ((rows.level == "neighborhood") & (rows.target == "combined"))]
    if chosen.empty:
        raise RuntimeError("No eligible validation retrieval queries")
    scores = chosen.groupby("level").rank_percentile.mean()
    if not {"single", "neighborhood"}.issubset(scores.index):
        raise RuntimeError("Missing single or neighborhood validation retrieval score")
    return float(scores.loc[["single", "neighborhood"]].mean()), len(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/shape_metric.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    dataset = ROOT / cfg["dataset_dir"]
    artifacts = ROOT / cfg["output_dir"]
    out = artifacts / f"seed_{cfg['seed']}"
    if not (artifacts / "metric_targets.npy").exists():
        raise RuntimeError("Run prepare_shape_metric.py first")
    if args.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(4)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    frame = pd.read_csv(dataset / "manifest.csv").set_index("index", drop=False)
    groups = pd.read_csv(dataset / "split_groups.csv").set_index("sample")
    frame["similarity_group"] = frame["sample"].map(groups.similarity_group)
    raw = np.load(dataset / "raw96.npy", mmap_mode="r")
    sdf = np.load(dataset / "sdf96.npy", mmap_mode="r")
    labels = np.load(dataset / "labels96.npy", mmap_mode="r")
    target_array = np.load(artifacts / "metric_targets.npy", mmap_mode="r")
    length_frame = pd.read_csv(artifacts / "target_lengths.csv").set_index("index")
    lengths = length_frame.target_length.to_numpy(int)
    model = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    checkpoint = out / "last.pt"; history = []; start = 1
    if checkpoint.exists() and not args.smoke:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history = state["history"]; start = int(state["epoch"]) + 1
        print(f"RESUME epoch {start}", flush=True)
    best = max((row["validation_retrieval"] for row in history), default=-float("inf"))
    train = frame[frame.split == "train"]
    val = frame[frame.split == "validation"]
    if args.smoke:
        train = train.head(12); val = val.head(12)

    def batch(ids):
        ids = np.asarray(ids, int)
        x = torch.from_numpy(np.array(raw[ids], copy=True)).unsqueeze(1).float().to(device)
        y = torch.from_numpy(np.array(sdf[ids], copy=True)).unsqueeze(1).float().to(device)
        l = torch.from_numpy(np.array(labels[ids], copy=True)).unsqueeze(1).to(device)
        t = torch.from_numpy(np.nan_to_num(np.array(target_array[ids], copy=True), nan=0)).float().to(device)
        return x, y, l, t, lengths[ids], frame.loc[ids, "fov_um"].to_numpy(float)

    epochs = 1 if args.smoke else int(cfg["epochs"])
    for epoch in range(start, epochs + 1):
        lr = cfg["learning_rate"] * .5 * (1 + math.cos(math.pi * (epoch - 1) / epochs))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, cfg["learning_rate"] * .01)
        model.train(); train_rows = []
        for ids in matched_batches(train, cfg["batch_size"], cfg["seed"] + epoch):
            parts = losses(model, *batch(ids), cfg, rng, True)
            optimizer.zero_grad(set_to_none=True)
            parts[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_rows.append([float(p.detach().cpu()) for p in parts])
            if args.smoke and len(train_rows) >= 2:
                break
        model.eval(); validation_rows = []
        with torch.no_grad():
            for ids in matched_batches(val, cfg["batch_size"], cfg["seed"]):
                validation_rows.append([float(p.detach().cpu()) for p in losses(model, *batch(ids), cfg, rng, False)])
                if args.smoke and len(validation_rows) >= 2:
                    break
        if not train_rows or not validation_rows:
            raise RuntimeError("No matched training or validation batches")
        train_mean = np.mean(train_rows, 0); val_mean = np.mean(validation_rows, 0)
        if args.smoke:
            retrieval_score, queries = float("nan"), 0
        else:
            retrieval_score, queries = validation_retrieval(frame, raw, target_array, lengths, model, device)
        row = {"epoch": epoch, "train_loss": float(train_mean[0]), "validation_loss": float(val_mean[0]),
            "train_metric_loss": float(train_mean[1]), "validation_metric_loss": float(val_mean[1]),
            "sdf": float(train_mean[2]), "occupancy": float(train_mean[3]),
            "boundary": float(train_mean[4]), "view": float(train_mean[5]),
            "validation_retrieval": retrieval_score, "validation_queries": queries,
            "train_batches": len(train_rows), "validation_batches": len(validation_rows)}
        history.append(row)
        print("EPOCH", json.dumps(row), flush=True)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "config": cfg}
        tmp = out / "last.tmp.pt"; torch.save(state, tmp); os.replace(tmp, checkpoint)
        if retrieval_score > best:
            best = retrieval_score
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        (out / "status.json").write_text(json.dumps({"status": "training" if epoch < epochs else "trained_pending_evaluation",
            "epoch": epoch, "epochs": epochs, "best_validation_retrieval": best}, indent=2))
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
