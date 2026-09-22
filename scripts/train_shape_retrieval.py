"""Two-stage shape-specialized training. Test rows are never loaded in this module."""
from pathlib import Path
import argparse, json, math, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from simclr3d_core import ResNet3DSimCLR, augment, augment_raw, nt_xent, pair_metrics

ROOT = Path(__file__).resolve().parents[1]
SHAPE_FEATURES = ["volume_um3", "surface_area_um2", "sphericity", "aspect_ratio",
                  "elongation", "flatness", "mesh_solidity", "surface_to_volume_per_um"]


def synthetic_shapes(n, size, generator):
    """Smooth superellipsoids with low-frequency boundary deformations, voxelized last."""
    axis = torch.linspace(-1, 1, size)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz = torch.stack([z, y, x])
    out = []
    for _ in range(n):
        radii = .28 + .48 * torch.rand(3, generator=generator)
        exponent = 1.4 + 2.8 * torch.rand(1, generator=generator)
        q = F.normalize(torch.randn(4, generator=generator), dim=0)
        w, a, b, c = q
        rotation = torch.stack([1-2*(b*b+c*c), 2*(a*b-w*c), 2*(a*c+w*b),
                                2*(a*b+w*c), 1-2*(a*a+c*c), 2*(b*c-w*a),
                                2*(a*c-w*b), 2*(b*c+w*a), 1-2*(a*a+b*b)]).reshape(3, 3)
        p = torch.einsum("ij,jzyx->izyx", rotation, xyz)
        phase = 2 * math.pi * torch.rand(3, generator=generator)
        deformation = 1 + .10 * (torch.sin(2.3*p[0]+phase[0]) *
                                  torch.sin(2.0*p[1]+phase[1]) *
                                  torch.sin(1.7*p[2]+phase[2]))
        value = sum((p[k].abs() / (radii[k] * deformation)).pow(exponent) for k in range(3))
        out.append((value <= 1).float())
    return torch.stack(out).unsqueeze(1)


class ShapeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = ResNet3DSimCLR(cfg["base_channels"], cfg["embedding_dim"], cfg["projection_dim"])
        self.probe = nn.Linear(cfg["embedding_dim"], len(SHAPE_FEATURES))

    def forward(self, x):
        h, z = self.network(x)
        return h, z, self.probe(h)


def cosine_alignment(a, b):
    return 1 - (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1).mean()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/shape_retrieval.json")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--device", default=None); args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text()); data_out = ROOT / cfg["output_dir"]
    if args.device: cfg["device"] = args.device
    out = data_out / "smoke" if args.smoke else data_out; out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    device = torch.device(cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU unavailable; run from a normal Terminal")
    manifest = pd.read_csv(data_out / "manifest.csv")
    if set(manifest.split) != {"train", "validation", "test"}: raise ValueError("Expected three-way split")
    train_ids = manifest.index[manifest.split == "train"].to_numpy()
    val_ids = manifest.index[manifest.split == "validation"].to_numpy()
    # Deliberate leakage guard: no test index is constructed or loaded here.
    raw = np.load(ROOT / "results/simclr3d_raw/raw64.npy", mmap_mode="r")
    masks = np.load(data_out / "masks_fixed64.npy", mmap_mode="r")
    raw_indices = manifest["index"].to_numpy(int)
    feature_values = manifest[SHAPE_FEATURES].to_numpy(np.float32)
    feature_mean = feature_values[train_ids].mean(0); feature_std = feature_values[train_ids].std(0).clip(1e-6)
    normalized_features = (feature_values - feature_mean) / feature_std
    (out / "shape_feature_normalization.json").write_text(json.dumps({
        "features": SHAPE_FEATURES, "mean": feature_mean.tolist(), "std": feature_std.tolist()}, indent=2))
    epochs_teacher = 1 if args.smoke else cfg["teacher_epochs"]
    epochs_student = 1 if args.smoke else cfg["student_epochs"]
    batch_size = 4 if args.smoke else cfg["batch_size"]
    if args.smoke: train_ids, val_ids = train_ids[:8], val_ids[:8]
    rng = np.random.default_rng(seed); aug_rng = torch.Generator().manual_seed(seed + 1)

    def tensor(array, ids):
        return torch.from_numpy(np.array(array[ids], copy=True)).unsqueeze(1).float().to(device)
    def schedule(optimizer, epoch, total):
        lr = cfg["learning_rate"] * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(total, 1)))
        for group in optimizer.param_groups: group["lr"] = max(lr, cfg["learning_rate"] * .01)
    def save(stage, epoch, model, optimizer, history, best):
        payload = {"stage": stage, "epoch": epoch, "config": cfg, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "history": history, "best": best}
        torch.save(payload, out / f"{stage}_last.pt")

    teacher = ShapeModel(cfg).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = []; best = float("inf")
    for epoch in range(1, epochs_teacher + 1):
        teacher.train(); schedule(opt, epoch, epochs_teacher); losses = []
        order = rng.permutation(train_ids)
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            if len(ids) < 2: continue
            real = tensor(masks, ids)
            n_syn = max(2, int(round(len(ids) * cfg["synthetic_fraction"])))
            base = torch.cat([real, synthetic_shapes(n_syn, cfg["input_size"], aug_rng).to(device)])
            a = augment(base, aug_rng, cfg["augmentation_scale"], cfg["augmentation_translation"])
            b = augment(base, aug_rng, cfg["augmentation_scale"], cfg["augmentation_translation"])
            _, za, _ = teacher(a); _, zb, _ = teacher(b); loss = nt_xent(za, zb, cfg["temperature"])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(teacher.parameters(), 5); opt.step()
            losses.append(float(loss.detach().cpu()))
        teacher.eval(); vals = []
        with torch.no_grad():
            for start in range(0, len(val_ids), batch_size):
                ids = val_ids[start:start+batch_size]
                if len(ids) < 2: continue
                x = tensor(masks, ids); a = augment(x, aug_rng); b = augment(x, aug_rng)
                _, za, _ = teacher(a); _, zb, _ = teacher(b); vals.append(float(nt_xent(za, zb, cfg["temperature"]).cpu()))
        row = {"stage": "teacher", "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": float(np.mean(vals))}
        history.append(row); print(json.dumps(row), flush=True); save("teacher", epoch, teacher, opt, history, best)
        if row["validation_loss"] < best: best = row["validation_loss"]; torch.save({"model": teacher.state_dict(), "config": cfg, "epoch": epoch}, out / "teacher_best.pt")

    teacher.load_state_dict(torch.load(out / "teacher_best.pt", map_location="cpu", weights_only=False)["model"])
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    student = ShapeModel(cfg).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    student_history = []; best = float("inf")
    for epoch in range(1, epochs_student + 1):
        student.train(); schedule(opt, epoch, epochs_student); rows = []
        order = rng.permutation(train_ids)
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            if len(ids) < 2: continue
            xraw = tensor(raw, raw_indices[ids]); xmask = tensor(masks, ids)
            view1 = augment_raw(xraw, aug_rng, cfg); view2 = augment_raw(xraw, aug_rng, cfg)
            with torch.no_grad(): target, _, _ = teacher(xmask)
            h1, z1, pred1 = student(view1); h2, z2, pred2 = student(view2)
            ssl = nt_xent(z1, z2, cfg["temperature"])
            align = .5 * (cosine_alignment(h1, target) + cosine_alignment(h2, target))
            truth = torch.from_numpy(normalized_features[ids]).float().to(device)
            probe = .5 * (F.smooth_l1_loss(pred1, truth) + F.smooth_l1_loss(pred2, truth))
            loss = cfg["contrastive_weight"]*ssl + cfg["alignment_weight"]*align + cfg["shape_probe_weight"]*probe
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(student.parameters(), 5); opt.step()
            rows.append([float(loss.detach().cpu()), float(ssl.detach().cpu()), float(align.detach().cpu()), float(probe.detach().cpu())])
        student.eval(); vals = []
        with torch.no_grad():
            for start in range(0, len(val_ids), batch_size):
                ids = val_ids[start:start+batch_size]
                if len(ids) < 2: continue
                xraw = tensor(raw, raw_indices[ids]); target, _, _ = teacher(tensor(masks, ids)); h, _, pred = student(xraw)
                truth = torch.from_numpy(normalized_features[ids]).float().to(device)
                vals.append(float((cfg["alignment_weight"]*cosine_alignment(h, target) + cfg["shape_probe_weight"]*F.smooth_l1_loss(pred, truth)).cpu()))
        mean = np.mean(rows, axis=0)
        row = {"stage": "student", "epoch": epoch, "train_loss": float(mean[0]), "ssl": float(mean[1]),
               "alignment": float(mean[2]), "shape_probe": float(mean[3]), "validation_loss": float(np.mean(vals))}
        student_history.append(row); print(json.dumps(row), flush=True); save("student", epoch, student, opt, student_history, best)
        pd.DataFrame(history + student_history).to_csv(out / "training_history.csv", index=False)
        (out / "status.json").write_text(json.dumps({"status": "training", "stage": "student", "epoch": epoch,
                                                       "total_epochs": epochs_student, **row}, indent=2))
        if row["validation_loss"] < best: best = row["validation_loss"]; torch.save({"model": student.state_dict(), "config": cfg, "epoch": epoch}, out / "student_best.pt")
    (out / "status.json").write_text(json.dumps({"status": "trained_pending_test_evaluation",
                                                   "teacher_epochs": epochs_teacher, "student_epochs": epochs_student,
                                                   "test_rows_loaded": False}, indent=2))
    print("TRAINING COMPLETE; held-out test set remains unopened", flush=True)


if __name__ == "__main__": main()
