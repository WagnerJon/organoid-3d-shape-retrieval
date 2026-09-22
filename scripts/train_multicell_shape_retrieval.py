"""Fine-tune the shape teacher/student on count-matched single, neighborhood and organoid batches."""
from pathlib import Path
import argparse, json, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from simclr3d_core import augment, augment_raw, nt_xent
from train_shape_retrieval import ROOT, ShapeModel, cosine_alignment


def strata_batches(manifest, indices, batch_size, rng):
    frame = manifest.loc[indices]
    groups = []
    for _, group in frame.groupby(["level", "member_count"]):
        ids = group.index.to_numpy()
        if len(ids) < 2: continue
        ids = rng.permutation(ids)
        groups.extend(ids[start:start+batch_size] for start in range(0, len(ids)-1, batch_size))
    rng.shuffle(groups)
    return groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config/shape_retrieval_multicell.json")
    parser.add_argument("--smoke",action="store_true");parser.add_argument("--device",default=None);args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());data_out=ROOT/cfg["output_dir"]
    if args.device:cfg["device"]=args.device
    out=data_out/"smoke" if args.smoke else data_out;out.mkdir(parents=True,exist_ok=True)
    seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    device=torch.device(cfg["device"])
    if device.type=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
    manifest=pd.read_csv(data_out/"manifest.csv",index_col="index")
    train_ids=manifest.index[manifest.split=="train"].to_numpy();val_ids=manifest.index[manifest.split=="validation"].to_numpy()
    if args.smoke:train_ids=train_ids[:20];val_ids=val_ids[:20]
    raw=np.load(data_out/"raw64.npy",mmap_mode="r");masks=np.load(data_out/"masks64.npy",mmap_mode="r")
    feature_cols=[f"shape_{i}" for i in range(8)];features=manifest[feature_cols].to_numpy(np.float32)
    mean=features[train_ids].mean(0);std=features[train_ids].std(0).clip(1e-6);targets=(features-mean)/std
    (out/"shape_feature_normalization.json").write_text(json.dumps({"features":feature_cols,"mean":mean.tolist(),"std":std.tolist()},indent=2))
    epochs_t=1 if args.smoke else cfg["teacher_epochs"];epochs_s=1 if args.smoke else cfg["student_epochs"]
    batch=4 if args.smoke else cfg["batch_size"];rng=np.random.default_rng(seed);aug_rng=torch.Generator().manual_seed(seed+1)
    def tensor(a,ids):return torch.from_numpy(np.array(a[ids],copy=True)).unsqueeze(1).float().to(device)
    def update_lr(opt,epoch,total):
        lr=cfg["learning_rate"]*.5*(1+math.cos(math.pi*(epoch-1)/max(total,1)))
        for g in opt.param_groups:g["lr"]=max(lr,cfg["learning_rate"]*.01)
    def initialize(model,key):
        checkpoint=torch.load(ROOT/cfg[key],map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["model"])
    def checkpoint(name,epoch,model,opt,history):
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"config":cfg,"history":history},out/f"{name}_last.pt")

    teacher=ShapeModel(cfg).to(device);initialize(teacher,"initialize_teacher")
    opt=torch.optim.AdamW(teacher.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    history=[];best=float("inf")
    for epoch in range(1,epochs_t+1):
        teacher.train();update_lr(opt,epoch,epochs_t);losses=[]
        for ids in strata_batches(manifest,train_ids,batch,rng):
            x=tensor(masks,ids);a=augment(x,aug_rng,cfg["augmentation_scale"],cfg["augmentation_translation"]);b=augment(x,aug_rng,cfg["augmentation_scale"],cfg["augmentation_translation"])
            _,za,_=teacher(a);_,zb,_=teacher(b);loss=nt_xent(za,zb,cfg["temperature"])
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(teacher.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
        teacher.eval();vals=[]
        with torch.no_grad():
            for ids in strata_batches(manifest,val_ids,batch,np.random.default_rng(seed+epoch)):
                x=tensor(masks,ids);_,za,_=teacher(augment(x,aug_rng));_,zb,_=teacher(augment(x,aug_rng));vals.append(float(nt_xent(za,zb,cfg["temperature"]).cpu()))
        row={"stage":"teacher","epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":float(np.mean(vals))};history.append(row);print(json.dumps(row),flush=True);checkpoint("teacher",epoch,teacher,opt,history)
        if row["validation_loss"]<best:best=row["validation_loss"];torch.save({"epoch":epoch,"model":teacher.state_dict(),"config":cfg},out/"teacher_best.pt")

    teacher.load_state_dict(torch.load(out/"teacher_best.pt",map_location="cpu",weights_only=False)["model"]);teacher.eval()
    for p in teacher.parameters():p.requires_grad=False
    student=ShapeModel(cfg).to(device);initialize(student,"initialize_student")
    opt=torch.optim.AdamW(student.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);student_history=[];best=float("inf")
    for epoch in range(1,epochs_s+1):
        student.train();update_lr(opt,epoch,epochs_s);rows=[]
        for ids in strata_batches(manifest,train_ids,batch,rng):
            xr=tensor(raw,ids);xm=tensor(masks,ids);v1=augment_raw(xr,aug_rng,cfg);v2=augment_raw(xr,aug_rng,cfg)
            with torch.no_grad():target,_,_=teacher(xm)
            h1,z1,p1=student(v1);h2,z2,p2=student(v2);ssl=nt_xent(z1,z2,cfg["temperature"]);align=.5*(cosine_alignment(h1,target)+cosine_alignment(h2,target))
            truth=torch.from_numpy(targets[ids]).float().to(device);probe=.5*(F.smooth_l1_loss(p1,truth)+F.smooth_l1_loss(p2,truth))
            loss=cfg["contrastive_weight"]*ssl+cfg["alignment_weight"]*align+cfg["shape_probe_weight"]*probe
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(student.parameters(),5);opt.step();rows.append([float(loss.detach().cpu()),float(ssl.detach().cpu()),float(align.detach().cpu()),float(probe.detach().cpu())])
        student.eval();vals=[]
        with torch.no_grad():
            for ids in strata_batches(manifest,val_ids,batch,np.random.default_rng(seed+100+epoch)):
                target,_,_=teacher(tensor(masks,ids));h,_,pred=student(tensor(raw,ids));truth=torch.from_numpy(targets[ids]).float().to(device)
                vals.append(float((cfg["alignment_weight"]*cosine_alignment(h,target)+cfg["shape_probe_weight"]*F.smooth_l1_loss(pred,truth)).cpu()))
        avg=np.mean(rows,axis=0);row={"stage":"student","epoch":epoch,"train_loss":float(avg[0]),"ssl":float(avg[1]),"alignment":float(avg[2]),"shape_probe":float(avg[3]),"validation_loss":float(np.mean(vals))}
        student_history.append(row);print(json.dumps(row),flush=True);checkpoint("student",epoch,student,opt,student_history)
        pd.DataFrame(history+student_history).to_csv(out/"training_history.csv",index=False);(out/"status.json").write_text(json.dumps({"status":"training","stage":"student","epoch":epoch,"total_epochs":epochs_s,**row},indent=2))
        if row["validation_loss"]<best:best=row["validation_loss"];torch.save({"epoch":epoch,"model":student.state_dict(),"config":cfg},out/"student_best.pt")
    (out/"status.json").write_text(json.dumps({"status":"trained_pending_test_evaluation","test_rows_loaded":False},indent=2));print("TRAINING COMPLETE",flush=True)


if __name__=="__main__":main()
