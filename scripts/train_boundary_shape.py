"""Train a raw-only-inference 3D encoder with predicted-mask geometry targets."""
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

ROOT = Path(__file__).resolve().parents[1]


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(4, cout), nn.GELU(),
                                  nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(4, cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.GroupNorm(4, cout))

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class ShapeEncoder(nn.Module):
    def __init__(self, base=8, embedding_dim=128):
        super().__init__()
        self.stem = Block(1, base)
        self.down1 = Block(base, base, 2)
        self.down2 = Block(base, base * 2, 2)
        self.down3 = Block(base * 2, base * 4, 2)
        self.down4 = Block(base * 4, base * 8, 2)
        self.embedding = nn.Linear(base * 8, embedding_dim)
        self.condition = nn.Linear(embedding_dim, base * 8)
        self.up3 = Block(base * 8 + base * 4, base * 4)
        self.up2 = Block(base * 4 + base * 2, base * 2)
        self.up1 = Block(base * 2 + base, base)
        self.up0 = Block(base + base, base)
        self.head = nn.Conv3d(base, 3, 1)  # signed distance, occupancy, instance boundary

    def forward(self, x, decode=True):
        a = self.stem(x)
        b = self.down1(a)
        c = self.down2(b)
        d = self.down3(c)
        e = self.down4(d)
        embedding = self.embedding(F.adaptive_avg_pool3d(e, 1).flatten(1))
        if not decode:
            return embedding, None
        e = e + self.condition(embedding)[:, :, None, None, None]
        def up(x, skip, block):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False), skip], 1))
        x = up(e, d, self.up3)
        x = up(x, c, self.up2)
        x = up(x, b, self.up1)
        x = up(x, a, self.up0)
        return embedding, self.head(x)


def matched_batches(frame, batch_size, seed):
    """Make hard candidate batches within level/count and nearby foreground volume."""
    rng = np.random.default_rng(seed)
    result = []
    for _, group in frame.groupby(["level", "member_count"]):
        if len(group) < 2:
            continue
        ordered = group.sort_values("volume_um3").index.to_numpy()
        for window in np.array_split(ordered, max(1, math.ceil(len(ordered) / (batch_size * 4)))):
            for ids in np.array_split(rng.permutation(window), max(1, math.ceil(len(window) / batch_size))):
                if len(ids) >= 2:
                    result.append(ids)
    rng.shuffle(result)
    return result


def random_xy_view(raw, targets, cfg, rng, photometric=True):
    k = rng.randrange(4)
    raw = torch.rot90(raw, k, dims=(-2, -1))
    if targets is not None:
        targets = torch.rot90(targets, k, dims=(-2, -1))
    for axis in (-2, -1):
        if rng.random() < .5:
            raw = torch.flip(raw, [axis])
            if targets is not None:
                targets = torch.flip(targets, [axis])
    if photometric:
        n = len(raw)
        gamma = torch.empty(n, 1, 1, 1, 1, device=raw.device).uniform_(*cfg["intensity_gamma"])
        gain = torch.empty(n, 1, 1, 1, 1, device=raw.device).uniform_(*cfg["intensity_gain"])
        raw = (raw.clamp(0, 1).pow(gamma) * gain + torch.randn_like(raw) * cfg["intensity_noise_std"]).clamp(0, 1)
    return raw, targets


def geometry_rank_loss(embedding, geometry, lengths):
    if len(embedding) < 3:
        return embedding.sum() * 0
    valid = int(lengths.min())
    if not np.all(lengths == valid):
        return embedding.sum() * 0
    target = torch.cdist(geometry[:, :valid], geometry[:, :valid])
    target.fill_diagonal_(float("inf"))
    near = target.argmin(1)
    target.fill_diagonal_(-1)
    far = target.argmax(1)
    z = F.normalize(embedding, dim=1)
    similarity = z @ z.T
    index = torch.arange(len(z), device=z.device)
    return F.softplus(similarity[index, far] - similarity[index, near] + .1).mean()


def losses(model, raw, sdf, labels, geometry, lengths, fov, cfg, rng, training):
    target = torch.cat([sdf, (labels > 0).float()], 1)
    if training:
        first, target = random_xy_view(raw, target, cfg, rng)
        second, _ = random_xy_view(raw, None, cfg, rng)
    else:
        first, second = raw, raw
    h, prediction = model(first)
    truth_sdf, foreground = target[:, :1], target[:, 1:2]
    clip = cfg["sdf_clip_um"]
    pitch = torch.as_tensor(fov / raw.shape[-1] / clip, dtype=torch.float32, device=raw.device).view(-1, 1, 1, 1, 1)
    boundary = (truth_sdf.abs() <= 1.1 * pitch).float()
    near_weight = 1 + 4 * torch.exp(-truth_sdf.abs() * clip / 1.5)
    sdf_loss = (F.smooth_l1_loss(torch.tanh(prediction[:, :1]), truth_sdf, reduction="none") * near_weight).sum() / near_weight.sum()
    occupancy_loss = F.binary_cross_entropy_with_logits(prediction[:, 1:2], foreground, pos_weight=torch.tensor(8., device=raw.device))
    boundary_loss = F.binary_cross_entropy_with_logits(prediction[:, 2:3], boundary, pos_weight=torch.tensor(8., device=raw.device))
    rank_loss = geometry_rank_loss(h, geometry, lengths)
    if training:
        h2, _ = model(second, decode=False)
        view_loss = (1 - (F.normalize(h, dim=1) * F.normalize(h2, dim=1)).sum(1)).mean()
    else:
        view_loss = h.sum() * 0
    total = (cfg["loss_sdf"] * sdf_loss + cfg["loss_occupancy"] * occupancy_loss +
             cfg["loss_boundary"] * boundary_loss + cfg["loss_view"] * view_loss +
             cfg["loss_ranking"] * rank_loss)
    parts = [total, sdf_loss, occupancy_loss, boundary_loss, view_loss, rank_loss]
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/boundary_shape.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.device:
        cfg["device"] = args.device
    out = ROOT / cfg["output_dir"]
    if not (out / "prepare_complete.json").exists():
        raise RuntimeError("Prepare the boundary-shape dataset before training")
    if args.smoke:
        out = out / "smoke"
        out.mkdir(exist_ok=True)
    device = torch.device(cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run from a normal Terminal")
    torch.set_num_threads(4)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    base = ROOT / cfg["output_dir"]
    manifest = pd.read_csv(base / "manifest.csv").set_index("index", drop=False)
    raw = np.load(base / "raw96.npy", mmap_mode="r")
    sdf = np.load(base / "sdf96.npy", mmap_mode="r")
    labels = np.load(base / "labels96.npy", mmap_mode="r")
    geoms = np.load(base / "geometry.npy", mmap_mode="r")
    train = manifest[manifest.split == "train"]
    val = manifest[manifest.split == "validation"]
    if args.smoke:
        train = train.head(8)
        val = val.head(8)
    model = ShapeEncoder(cfg["base_channels"], cfg["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = []
    start = 1
    checkpoint = out / "last.pt"
    if checkpoint.exists() and not args.smoke:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start = int(state["epoch"]) + 1
        print(f"RESUME epoch {start}", flush=True)
    best = min((row["validation_loss"] for row in history), default=float("inf"))

    def batch(ids):
        ids = np.asarray(ids, int)
        x = torch.from_numpy(np.array(raw[ids], copy=True)).unsqueeze(1).float().to(device)
        y = torch.from_numpy(np.array(sdf[ids], copy=True)).unsqueeze(1).float().to(device)
        l = torch.from_numpy(np.array(labels[ids], copy=True)).unsqueeze(1).to(device)
        g = torch.from_numpy(np.nan_to_num(np.array(geoms[ids], copy=True), nan=0)).float().to(device)
        length = manifest.loc[ids, "geometry_length"].to_numpy(int)
        fov = manifest.loc[ids, "fov_um"].to_numpy(float)
        return x, y, l, g, length, fov

    epochs = 1 if args.smoke else int(cfg["epochs"])
    for epoch in range(start, epochs + 1):
        lr = cfg["learning_rate"] * .5 * (1 + math.cos(math.pi * (epoch - 1) / epochs))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, cfg["learning_rate"] * .01)
        model.train(); train_rows = []
        for ids in matched_batches(train, cfg["batch_size"], cfg["seed"] + epoch):
            x, y, l, g, length, fov = batch(ids)
            parts = losses(model, x, y, l, g, length, fov, cfg, rng, True)
            optimizer.zero_grad(set_to_none=True)
            parts[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_rows.append([float(p.detach().cpu()) for p in parts])
            if args.smoke and len(train_rows) >= 2:
                break
        model.eval(); val_rows = []
        with torch.no_grad():
            for ids in matched_batches(val, cfg["batch_size"], cfg["seed"]):
                x, y, l, g, length, fov = batch(ids)
                val_rows.append([float(p.detach().cpu()) for p in losses(model, x, y, l, g, length, fov, cfg, rng, False)])
                if args.smoke and len(val_rows) >= 2:
                    break
        if not train_rows or not val_rows:
            raise RuntimeError("No usable matched train or validation batches")
        means = np.mean(train_rows, 0); validation = np.mean(val_rows, 0)
        row = {"epoch": epoch, "train_loss": float(means[0]), "validation_loss": float(validation[0]),
               "sdf": float(means[1]), "occupancy": float(means[2]), "boundary": float(means[3]),
               "view": float(means[4]), "ranking": float(means[5]), "train_batches": len(train_rows),
               "validation_batches": len(val_rows)}
        history.append(row)
        print("EPOCH", json.dumps(row), flush=True)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "history": history, "config": cfg}
        temporary = out / "last.tmp.pt"
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        (out / "status.json").write_text(json.dumps({"status": "training" if epoch < epochs else "trained_pending_evaluation",
                                                "epoch": epoch, "epochs": epochs, "best_validation_loss": best}, indent=2))
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
