"""Regenerate the three displayed result types from included dataset-level records."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]


def plots(output):
    output.mkdir(parents=True,exist_ok=True)
    one=json.loads((ROOT/'results/one_triplet/records.json').read_text())
    with (ROOT/'results/two_triplets/records.csv').open(newline='') as stream:
        two=list(csv.DictReader(stream))
    colors={.7:'#0072B2',.9:'#D55E00',.99:'#009E73'}
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,metric,label in zip(axes,['tv','covariance'],['Triplet support TV','Coefficient covariance error']):
        for rho in (.7,.9):
            rows=[r for r in one if r['p']==10 and r['rho']==rho]
            x=[r['methods']['MFVI'][metric] for r in rows];y=[r['methods']['Mixture'][metric] for r in rows]
            ax.scatter(x,y,s=17,color=colors[rho],label=f'Correlation {rho}')
        limit=max(*ax.get_xlim(),*ax.get_ylim());ax.plot([0,limit],[0,limit],color='gray',linestyle='--')
        ax.set(xlabel='MFVI',ylabel='Adaptive mixture',title=label)
    axes[0].legend();fig.tight_layout();fig.savefig(output/'05_original_joint_uncertainty.pdf');plt.close(fig)
    methods=['baseline','frozen_refinement','stagewise_boosting']
    names=['MFVI','Frozen','Stagewise']
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for j,rho in enumerate((.7,.9,.99)):
        rows=[r for r in two if float(r['rho'])==rho];by={(r['dataset'],r['method']):r for r in rows};ids=sorted({r['dataset'] for r in rows})
        for i,(metric,label) in enumerate([('posterior_kl','Reverse KL'),('group_union_tv','Grouped-support TV')]):
            ax=axes[i,j]
            for k,m in enumerate(methods):
                x=np.array([float(by[d,'original'][metric])-float(by[d,m][metric]) for d in ids])
                error=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
                ax.errorbar(x.mean(),k,xerr=error,fmt='o',color=colors[rho],capsize=4)
            ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(3),names);ax.set_title(f'{label}, correlation {rho}');ax.set_xlabel('Joint minus comparator')
    fig.tight_layout();fig.savefig(output/'07_two_group_paired_effects.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharex=True)
    for ax,rho in zip(axes,(.7,.9,.99)):
        for k,m in enumerate(['baseline','original','frozen_refinement','stagewise_boosting']):
            rows=[r for r in two if float(r['rho'])==rho and r['method']==m]
            support=np.mean([float(r['support_kl']) for r in rows]);conditional=np.mean([float(r['conditional_kl_estimate']) for r in rows])
            ax.barh(k,support,color='#0072B2',label='Support' if k==0 else None)
            ax.barh(k,conditional,left=support,color='#E69F00',label='Conditional' if k==0 else None)
        ax.set_yticks(range(4),['MFVI','Joint','Frozen','Stagewise']);ax.set(title=f'Correlation {rho}',xlabel='Mean reverse KL')
    axes[0].legend();fig.tight_layout();fig.savefig(output/'08_two_group_kl_decomposition.pdf');plt.close(fig)
    print(f'Regenerated three figures from included records in {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated/figures')
    args=parser.parse_args();plots(args.output)
