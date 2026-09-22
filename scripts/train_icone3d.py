"""Controlled IConE replacement for isolated-raw 3D SimCLR training."""
from pathlib import Path
import argparse, json, math, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from simclr3d_core import ResNet3DSimCLR, augment_raw, pair_metrics
from icone3d_core import losses

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config/icone3d_raw_isolated.json")
    args=parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    data_dir, out = ROOT / cfg["data_dir"], ROOT / cfg["output_dir"]
    out.mkdir(exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU unavailable; run outside the filesystem sandbox")
    data = np.load(data_dir / cfg["data_file"], mmap_mode="r")
    meta = pd.read_csv(data_dir / "manifest.csv")
    train_ids = meta.index[meta.split == "train"].to_numpy()
    validation_ids = meta.index[meta.split == "validation"].to_numpy()
    anchor_index = np.full(len(meta), -1, dtype=np.int64)
    anchor_index[train_ids] = np.arange(len(train_ids))
    backbone = ResNet3DSimCLR(cfg["base_channels"], cfg["embedding_dim"], 64).encoder.to(device)
    anchors = nn.Embedding(len(train_ids), cfg["embedding_dim"], device=device)
    nn.init.normal_(anchors.weight, mean=0, std=.02)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(anchors.parameters()),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    augmentation_rng = torch.Generator().manual_seed(seed + 1)
    history, start, started = [], 1, time.time()
    best_retrieval, best_epoch = -1., None
    if (out / "last.pt").exists():
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg:
            raise ValueError("Configuration changed; use a separate output directory")
        backbone.load_state_dict(checkpoint["backbone"])
        anchors.load_state_dict(checkpoint["anchors"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, start = checkpoint["history"], checkpoint["epoch"] + 1
        best_retrieval, best_epoch = checkpoint["best_retrieval"], checkpoint["best_epoch"]
        rng.bit_generator.state = checkpoint["numpy_rng"]
        augmentation_rng.set_state(checkpoint["augmentation_rng"])

    def batch(indices):
        return torch.from_numpy(np.array(data[indices], copy=True)).unsqueeze(1).float().to(device)

    def paired(x, generator):
        return augment_raw(x, generator, cfg), augment_raw(x, generator, cfg)

    def evaluate():
        backbone.eval(); generator = torch.Generator().manual_seed(seed + 999)
        records, features = [], []
        with torch.no_grad():
            for k in range(0, len(validation_ids), cfg["batch_size"]):
                ids = validation_ids[k:k + cfg["batch_size"]]
                if len(ids) < 2: continue
                a, b = paired(batch(ids), generator)
                z1, z2 = backbone(a), backbone(b)
                metrics = pair_metrics(z1, z2)
                metrics.update(n=len(ids), view_view=float(
                    (1 - (F.normalize(z1,dim=1)*F.normalize(z2,dim=1)).sum(1)).mean().cpu()))
                records.append(metrics); features.append(z1.cpu().numpy())
        result = {key: float(np.average([r[key] for r in records],
                  weights=[r["n"] for r in records])) for key in records[0] if key != "n"}
        h = np.concatenate(features); singular = np.linalg.svd(h-h.mean(0), compute_uv=False)
        probability = singular**2 / (singular @ singular + 1e-12)
        result["effective_rank"] = float(np.exp(-(probability*np.log(probability+1e-12)).sum()))
        result["encoder_std"] = float(h.std(0).mean())
        return result

    if start == 1:
        baseline = evaluate()
        (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
        print("Baseline " + json.dumps(baseline), flush=True)
    for epoch in range(start, cfg["epochs"] + 1):
        tick = time.time(); backbone.train(); order = rng.permutation(train_ids)
        totals = []
        learning_rate = cfg["learning_rate"] * .5 * (1 + math.cos(
            math.pi * (epoch-1) / cfg["epochs"]))
        for group in optimizer.param_groups: group["lr"] = learning_rate
        for k in range(0, len(order), cfg["batch_size"]):
            ids = order[k:k + cfg["batch_size"]]
            x = batch(ids); a, b = paired(x, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            z1, z2 = backbone(a), backbone(b)
            anchor_rows = anchors(torch.from_numpy(anchor_index[ids]).to(device))
            total, components = losses(z1, z2, anchor_rows, anchors.weight)
            if not torch.isfinite(total): raise RuntimeError("Nonfinite IConE loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters())+list(anchors.parameters()), 5.)
            optimizer.step()
            totals.append((len(ids), float(total.detach().cpu()),
                *(float(components[n].detach().cpu()) for n in
                  ["view_instance", "view_view", "diversity"])))
        weights = [r[0] for r in totals]
        row = dict(epoch=epoch, train_loss=float(np.average([r[1] for r in totals],weights=weights)),
            train_view_instance=float(np.average([r[2] for r in totals],weights=weights)),
            train_view_view=float(np.average([r[3] for r in totals],weights=weights)),
            train_diversity=float(np.average([r[4] for r in totals],weights=weights)),
            learning_rate=learning_rate)
        improved = False
        if epoch == 1 or epoch % cfg["validation_every"] == 0 or epoch == cfg["epochs"]:
            metrics = evaluate(); row.update({"validation_"+k:v for k,v in metrics.items()})
            improved = metrics["pair_retrieval"] > best_retrieval
            if improved: best_retrieval, best_epoch = metrics["pair_retrieval"], epoch
        row["seconds"] = time.time()-tick; history.append(row)
        checkpoint = dict(epoch=epoch, config=cfg, backbone=backbone.state_dict(),
            anchors=anchors.state_dict(), optimizer=optimizer.state_dict(), history=history,
            best_retrieval=best_retrieval, best_epoch=best_epoch,
            numpy_rng=rng.bit_generator.state, augmentation_rng=augmentation_rng.get_state())
        torch.save(checkpoint, out / "last.tmp"); (out / "last.tmp").replace(out / "last.pt")
        if improved: torch.save(checkpoint, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        (out / "status.json").write_text(json.dumps(dict(status="training",
            epoch=epoch, total_epochs=cfg["epochs"], best_epoch=best_epoch,
            best_validation_pair_retrieval=best_retrieval, **{k:v for k,v in row.items() if k!="epoch"}), indent=2))
        print(json.dumps(row), flush=True)
    # Fixed-budget final checkpoint is the primary IConE representation.
    backbone.eval(); features=[]
    with torch.no_grad():
        for k in range(0, len(data), cfg["batch_size"]):
            features.append(backbone(batch(np.arange(k,min(k+cfg["batch_size"],len(data))))).cpu().numpy())
    raw = np.concatenate(features); normalized = raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
    np.save(out / "embeddings_raw.npy", raw); np.save(out / "embeddings.npy", normalized)
    pd.concat([meta[["cell_id","sample","label","split","shape_qc_pass","original_volume_um3"]],
        pd.DataFrame(normalized,columns=[f"embedding_{i:03d}" for i in range(normalized.shape[1])])],axis=1).to_csv(out/"embeddings.csv",index=False)
    (out / "status.json").write_text(json.dumps(dict(status="complete", epochs=cfg["epochs"],
        exported_checkpoint="final fixed-budget epoch", diagnostic_best_retrieval_epoch=best_epoch,
        best_validation_pair_retrieval=best_retrieval, objects_embedded=len(normalized),
        encoder_parameters=sum(p.numel() for p in backbone.parameters()),
        anchor_parameters=anchors.weight.numel(), run_seconds=time.time()-started), indent=2))
    print("COMPLETE " + (out / "status.json").read_text(), flush=True)


if __name__ == "__main__": main()
