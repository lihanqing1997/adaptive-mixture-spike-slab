"""Refit one reported fixed-K comparison from its original saved candidate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amvi import Mixture,RegressionModel,Settings
from amvi.data import make_two_triplets
from amvi.local import fit_local
from run_dataset import encode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--arm',required=True,choices=['direct_joint','augmented_joint','direct_frozen','direct_stagewise'])
    parser.add_argument('--cap',type=int,choices=[16,128],default=16)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file.')
    candidate=ROOT/'results/local_refinement/candidates'/f'{args.dataset}.npz'
    if not candidate.exists():parser.error('Dataset is not in the reported 60-dataset local study.')
    configs={c['id']:c for c in json.loads((ROOT/'results/datasets.json').read_text())}
    c=configs[args.dataset];d=make_two_triplets(c)
    with np.load(candidate,allow_pickle=False) as z:
        initial=Mixture(*(z[k].copy() for k in ['weights','a','mu','var']))
    model=RegressionModel(d['x'],d['y'],c['sigma'],c['tau'],c['omega'])
    result=fit_local(model,initial,args.arm,[.7,.9,.99].index(c['rho']),c['replicate'],
        Settings(max_refreshes=args.cap,max_seconds=60))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(configuration=c,arm=args.arm,cap=args.cap,fit=result),default=encode,indent=2)+'\n')
    print(f'Saved {args.output}. Fitting decisions use the selected arm objective.')


if __name__=='__main__':main()
