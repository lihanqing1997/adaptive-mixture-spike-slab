"""Refit one reported dataset with the original design and adaptive method."""
import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amvi import RegressionModel, Settings, fit
from amvi.comparators import fit_variant
from amvi.data import make_one_triplet, make_two_triplets
from amvi.initialization import strong_meanfield
from amvi.solver import compare


def encode(value):
    if is_dataclass(value):return asdict(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',help='ID from results/datasets.json')
    parser.add_argument('--method',choices=['original','frozen_refinement','stagewise_boosting'],default='original')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file; existing results are preserved.')
    configs={c['id']:c for c in json.loads((ROOT/'results/datasets.json').read_text())}
    if args.dataset not in configs:parser.error('Unknown dataset ID')
    c=configs[args.dataset];two=c.get('groups')==2
    if two:
        d=make_two_triplets(c);x,y=d['x'],d['y']
    else:x,y,*_=make_one_triplet(c)
    model=RegressionModel(x,y,c['sigma'],c['tau'],c['omega'])
    baseline_seed=c['numerical_seed']+(0 if two else 110000009)
    _,baseline,starts=strong_meanfield(model,baseline_seed)
    settings=Settings(max_components=10,max_seconds=60 if c['p']==10 else 120)
    if args.method=='original':result=fit(model,settings,c['numerical_seed'],baseline)
    else:result=fit_variant(model,settings,c['numerical_seed'],baseline,args.method)
    assessment=compare(model,result['mixture'],baseline['mixture'],power=14,
        seed=c['numerical_seed']+(590000009 if two else 290000009),repeats=4)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(configuration=c,method=args.method,
        baseline_starts=starts,fit=result,independent_assessment=assessment),default=encode,indent=2)+'\n')
    print(f'Saved {args.output}; wall-clock stopping can differ from the original run.')


if __name__=='__main__':main()
