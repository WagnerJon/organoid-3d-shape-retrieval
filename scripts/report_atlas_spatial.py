"""Readable report and scientific comparison figure for the Atlas export."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/atlas_spatial'


def main():
    result=json.loads((OUT/'evaluation.json').read_text());rows=result['methods'];summary={r['method']:r for r in rows}
    path=OUT/'report.md';lines=path.read_text().splitlines()
    lines=[line for line in lines if not line.startswith('| ') or line.startswith('| Comparison') or line.startswith('|---')]
    index=lines.index('|---|---:|---|')+1
    table=[]
    for r in rows:
        ci=r['vs_all9'];lo,hi=ci['ci95'];table.append(f"| {r['name']} | {r['mean_cost']:.6f} | {ci['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    lines[index:index]=table
    path.write_text('\n'.join(lines)+'\n')
    baseline=summary['morph_all']['mean_cost'];gain=1-summary['spatial_robust']['mean_cost']/baseline
    text=f'\nThe hybrid spatial comparison has {gain:.1%} lower group-mean geometry cost than all nine traditional features on 46 eligible validation queries across 11 groups. The independent fallback signature alone is not convincingly better than all nine features (its paired confidence interval includes zero). This supports a selective fallback, not replacing spatial geometry with global distance distributions.\n'
    with path.open('a') as out:out.write(text)
    names=[r['name'].replace(' · ','\n') for r in rows];means=np.array([r['vs_all9']['mean'] for r in rows]);ci=np.array([r['vs_all9']['ci95'] for r in rows])
    fig,ax=plt.subplots(figsize=(11,6));y=np.arange(len(rows))
    ax.barh(y,means,color=['#009d9b' if r['method'].startswith('spatial') else '#8899ae' for r in rows],alpha=.8)
    ax.errorbar(means,y,xerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='none',ecolor='#25394c',capsize=3)
    ax.axvline(0,color='#333',lw=.7);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('Reduction in normalized geometric neighbor cost vs all nine traditional features →')
    ax.set_title('Validation geometry comparison · 46 queries / 11 groups',loc='left',weight='bold');ax.spines[['right','top']].set_visible(False)
    fig.text(.02,.015,'Group bootstrap 95% intervals. Matched candidates from other groups. Exploratory; source segmentation geometry, not biological labels.',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,1]);fig.savefig(OUT/'feature_comparison.png',dpi=180);plt.close(fig)
    data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());cells=[c for c in data['cells'] if c['inMainOrganoid'] and c['qc']]
    methods=[m for m in data['methods'] if m.get('family')];colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(3,3,figsize=(13,11));reference='morph_controls'
    for ax,m in zip(axes.ravel(),methods):
        p=np.array([c['representations'][m['id']]['point'] for c in cells]);groups=np.array([c['representations'][reference]['cluster'] for c in cells])
        ax.scatter(p[:,0],p[:,1],c=[colors(int(g)) for g in groups],s=7,alpha=.7,linewidths=0)
        ax.set_title(m['name'],fontsize=11);ax.set_xlabel('MDS 1' if m['projection']=='mds' else 'PC 1');ax.set_ylabel('MDS 2' if m['projection']=='mds' else 'PC 2');ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Same included QC cells · colors follow volume / elongation / sphericity groups',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(OUT/'spaces.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
