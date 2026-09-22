"""Evaluate single-cell, local-neighborhood and organoid retrieval on untouched test stacks."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from train_shape_retrieval import ROOT, ShapeModel


def embed(model,array,ids,device,batch):
    result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            x=torch.from_numpy(np.array(array[ids[start:start+batch]],copy=True)).unsqueeze(1).float().to(device)
            h,_,_=model(x);result.append(torch.nn.functional.normalize(h,dim=1).cpu().numpy())
    return np.concatenate(result)


def metrics(embedding,truth,samples,k=5):
    pred=pairwise_distances(embedding,metric="cosine");actual=pairwise_distances(truth);allowed=samples[:,None]!=samples[None,:]
    recalls=[];ndcgs=[];px=[];ax=[];discount=1/np.log2(np.arange(2,k+2))
    for i in range(len(samples)):
        candidates=np.flatnonzero(allowed[i])
        if len(candidates)<k:continue
        true=candidates[np.argsort(actual[i,candidates])];rank=candidates[np.argsort(pred[i,candidates])]
        recalls.append(len(set(true[:k])&set(rank[:k]))/k)
        gain=np.exp(-actual[i,rank[:k]]);ideal=np.exp(-actual[i,true[:k]]);ndcgs.append(float(gain@discount/max(ideal@discount,1e-12)))
        px.extend(pred[i,candidates]);ax.extend(actual[i,candidates])
    return {"queries":len(recalls),"recall_at_5":float(np.mean(recalls)),"ndcg_at_5":float(np.mean(ndcgs)),"distance_spearman":float(spearmanr(px,ax).statistic)}


def main():
    cfg=json.loads((ROOT/"config/shape_retrieval_multicell.json").read_text());out=ROOT/cfg["output_dir"]
    status=json.loads((out/"status.json").read_text())
    if status["status"]!="trained_pending_test_evaluation":raise RuntimeError("Training incomplete")
    manifest=pd.read_csv(out/"manifest.csv",index_col="index");test_ids=manifest.index[manifest.split=="test"].to_numpy();train_ids=manifest.index[manifest.split=="train"].to_numpy()
    device=torch.device(cfg["device"]);student=ShapeModel(cfg).to(device);teacher=ShapeModel(cfg).to(device)
    student.load_state_dict(torch.load(out/"student_best.pt",map_location="cpu",weights_only=False)["model"]);teacher.load_state_dict(torch.load(out/"teacher_best.pt",map_location="cpu",weights_only=False)["model"])
    raw=np.load(out/"raw64.npy",mmap_mode="r");masks=np.load(out/"masks64.npy",mmap_mode="r")
    sh=embed(student,raw,test_ids,device,cfg["batch_size"]);th=embed(teacher,masks,test_ids,device,cfg["batch_size"])
    cols=[f"shape_{i}" for i in range(8)];values=manifest[cols].to_numpy(float);mean=values[train_ids].mean(0);std=values[train_ids].std(0).clip(1e-8);truth=(values[test_ids]-mean)/std
    test=manifest.loc[test_ids].reset_index();result={}
    for level in ["single","neighborhood","organoid"]:
        use=np.flatnonzero(test.level.to_numpy()==level);samples=test.loc[use,"sample"].to_numpy()
        result[level]={"cells_or_fields":len(use),"student_raw":metrics(sh[use],truth[use],samples),"teacher_mask":metrics(th[use],truth[use],samples)}
    # Arrangement-specific view removes total volume and occupancy descriptors.
    use=np.flatnonzero(test.level.to_numpy()=="neighborhood");result["neighborhood_arrangement_only"]={
        "student_raw":metrics(sh[use],truth[use][:,1:7],test.loc[use,"sample"].to_numpy()),
        "teacher_mask":metrics(th[use],truth[use][:,1:7],test.loc[use,"sample"].to_numpy())}
    np.save(out/"test_student_embeddings.npy",sh);np.save(out/"test_teacher_embeddings.npy",th)
    pd.concat([test[["index","sample","level","member_count","member_labels"]],pd.DataFrame(sh,columns=[f"embedding_{i:03d}" for i in range(sh.shape[1])])],axis=1).to_csv(out/"test_embeddings.csv",index=False)
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2));status.update({"status":"complete","test_evaluation_complete":True});(out/"status.json").write_text(json.dumps(status,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
