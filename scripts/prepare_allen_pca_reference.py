"""Cache the fixed spatial-PCA comparison meshes without changing their fit."""
import numpy as np
import pandas as pd
import torch
from allen_reference_core import OUT
from pointnet_shape_core import ROOT
from spatial_benchmark_core import mesh_for,surface_frame
from evaluate_spatial_benchmark import Encoder


def main():
    torch.set_num_threads(2);folder=OUT/'evaluation/meshes';folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(ROOT/'results/spatial_benchmark/development/manifest.csv');codes=np.load(ROOT/'results/spatial_benchmark/development/codes.npz')['pca_aligned_base'];model=Encoder('pca',True,'cpu')
    for row,code in zip(frame.to_dict('records'),codes):
        path=folder/f"{row['cell_id'].replace(':','_')}_spatial_pca.npz"
        if path.exists():continue
        v,f=mesh_for(row);center,axes,_,_=surface_frame(v,f);surface,field=model.surface(code[0])
        if surface is None:vertices=np.empty((0,3));faces=np.empty((0,3),int)
        else:vertices=surface[0]@axes.T+center;faces=surface[1]
        np.savez_compressed(path,vertices=vertices,faces=faces,field=field)
    print('Fixed PCA reference meshes ready:48 validation cells',flush=True)


if __name__=='__main__':main()
