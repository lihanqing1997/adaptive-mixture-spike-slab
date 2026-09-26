"""Reproduce current-paper summaries using only the compact records in this repo."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def csv_rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def paired(values):
    x = np.asarray(values, float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("A reported comparison needs at least two finite observations")
    mean = float(x.mean())
    radius = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return dict(n=len(x), mean=mean, interval=[mean-radius, mean+radius])


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def check(stat, saved):
    assert stat['n'] == saved['n']
    equal(stat['mean'], saved['mean'])
    equal(stat['interval'], saved.get('interval', saved.get('ci95')))


def reproduce(output):
    configs = read('datasets.json')
    assert len(configs) == len({c['id'] for c in configs}) == 550
    one = read('one_triplet/records.json')
    assert len(one) == len({r['id'] for r in one}) == 400
    groups = []
    for p in (10,20,30,100):
        for rho in (.7,.9):
            rows=[r for r in one if r['p']==p and r['rho']==rho]
            assert len(rows)==50
            eligible=[r for r in rows if r['reference_eligible']]
            g=dict(p=p,rho=rho,n=50,eligible=len(eligible),objective=paired([r['delta'] for r in rows]),means={},paired={})
            for metric in ('pip','tv','mse'):
                subset=eligible if metric in ('pip','tv') else rows
                g['means'][metric]={m:float(np.mean([r['methods'][m][metric] for r in subset])) for m in ('MFVI','Mixture')}
                g['paired'][metric]=paired([r['methods']['Mixture'][metric]-r['methods']['MFVI'][metric] for r in subset])
            groups.append(g)
    assert [g['eligible'] for g in groups]==[50,50,49,37,49,29,45,24]
    for saved in read('expected/support.json'):
        g=next(g for g in groups if (g['p'],g['rho'])==(saved['p'],saved['rho']))
        check(g['paired']['tv'],saved)
        equal([g['means']['tv']['MFVI'],g['means']['tv']['Mixture']],[saved['mfvi'],saved['mixture']])
    for saved in read('expected/intermediate.json'):
        g=next(g for g in groups if (g['p'],g['rho'])==(saved['p'],saved['rho']))
        check(g['objective'],saved['objective_difference'])
        for metric,key in [('pip','pip_difference'),('tv','support_difference'),('mse','prediction_difference')]:
            check(g['paired'][metric],saved[key])
    for saved in read('expected/one_triplet_original.json'):
        rows=[r for r in one if (r['p'],r['rho'])==(saved['p'],saved['rho'])]
        check(paired([r['delta'] for r in rows]),saved['delta'])
        for method, metrics in saved['methods'].items():
            for metric,expected in metrics.items():
                if not expected or expected.get('n',0)==0:continue
                subset=[r for r in rows if r['methods'][method].get(metric) is not None and
                        (r['p']==10 or metric not in ('pip','block_pip') or r['reference_eligible'])]
                assert len(subset)==expected['n']
                equal(np.mean([r['methods'][method][metric] for r in subset]),expected['mean'])
        for metric,expected in saved['paired'].items():
            if not expected['n']:continue
            subset=[r for r in rows if r['methods']['MFVI'].get(metric) is not None and
                    (r['p']==10 or metric not in ('pip','block_pip') or r['reference_eligible'])]
            check(paired([r['methods']['Mixture'][metric]-r['methods']['MFVI'][metric] for r in subset]),expected)

    two=csv_rows('two_triplets/records.csv')
    assert len(two)==600 and len({r['dataset'] for r in two})==150
    assert len({(r['dataset'],r['method']) for r in two})==600
    expected_two=read('expected/two_triplets.json')
    for rho, saved in expected_two.items():
        subset=[r for r in two if float(r['rho'])==float(rho)]
        by={(r['dataset'],r['method']):r for r in subset}
        ids=sorted({r['dataset'] for r in subset})
        assert len(ids)==50
        for method,metrics in saved['means'].items():
            for metric,stat in metrics.items():
                check(paired([float(by[(ds,method)][metric]) for ds in ids]),stat)
        for method,metrics in saved['paired_joint_minus_comparator'].items():
            for metric,stat in metrics.items():
                check(paired([float(by[(ds,'original')][metric])-float(by[(ds,method)][metric]) for ds in ids]),stat)

    fit_rows=read('checks/fitting_strategies.json')
    assert len(fit_rows)==120
    for s in read('expected/comparison_pairs.json'):
        first,second=s['contrast'].split(' minus ')
        by={(r['id'],r['method']):r for r in fit_rows if r['rho']==s['rho']}
        ids=sorted({ds for ds,method in by})
        for metric,stat in s['differences'].items():
            if not stat['n']:continue
            check(paired([by[(ds,first)][metric]-by[(ds,second)][metric] for ds in ids]),stat)
    budget_rows=read('checks/time_budgets.json')
    assert len(budget_rows)==60
    for s in read('expected/budget_pairs.json'):
        by={(r['id'],r['budget']):r for r in budget_rows if r['rho']==s['rho']}
        ids=sorted({ds for ds,cap in by})
        for metric,stat in s['differences'].items():
            if not stat['n']:continue
            chosen=[ds for ds in ids if metric!='pip' or by[(ds,s['budget'])]['amended_eligible']]
            check(paired([by[(ds,s['budget'])][metric]-by[(ds,s['reference_budget'])][metric] for ds in chosen]),stat)
    assert len(read('checks/longer_references.json'))==6
    assert len(read('checks/reference_diagnostics.json'))==400

    local={cap:read(f'local_refinement/assessments_{cap}.json') for cap in (16,128)}
    by={cap:{(r['id'],r['arm']):r for r in rows} for cap,rows in local.items()}
    for cap in local:
        assert len(local[cap])==len(by[cap])==240
        assert len({r['id'] for r in local[cap]})==60
    def values(cap,rho,arm,metric):
        ids=sorted(r['id'] for r in local[cap] if r['rho']==rho and r['arm']=='direct_joint')
        if arm=='initial':return np.array([by[cap][(ds,'direct_joint')]['initial_metrics'][metric] for ds in ids])
        return np.array([by[cap][(ds,arm)]['metrics'][metric] for ds in ids])
    for name in ('arm_means.csv','within_budget_contrasts.csv'):
        for s in csv_rows('local_refinement/'+name):
            cap,rho=int(s['refresh_cap']),float(s['rho'])
            x=values(cap,rho,s['arm'],s['metric'])
            if name.startswith('within_'):x=values(cap,rho,'direct_joint',s['metric'])-x
            stat=paired(x)
            equal([stat['mean'],*stat['interval']],[float(s['mean']),float(s['ci95_low']),float(s['ci95_high'])])
    changes=csv_rows('local_refinement/budget_comparison.csv')
    for s in changes:
        rho=float(s['rho']);x16=values(16,rho,s['arm'],s['metric']);x128=values(128,rho,s['arm'],s['metric'])
        if s['kind']=='change_in_contrast':
            x16=values(16,rho,'direct_joint',s['metric'])-x16
            x128=values(128,rho,'direct_joint',s['metric'])-x128
        stat=paired(x128-x16)
        equal([x16.mean(),x128.mean(),stat['mean'],*stat['interval']],
              [float(s[k]) for k in ('mean16','mean128','change128minus16','ci95_low','ci95_high')])
    assert len(changes)==210
    result=dict(passed=True,independent_datasets=550,
        reference_eligibility_counts=[g['eligible'] for g in groups],
        local_fits=480,cross_budget_comparisons=210,
        all_retained_summary_means_and_intervals_match=True,
        scope='Recomputed from included per-dataset saved numerical outcomes; no refitting or raw-chain reanalysis.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'one_triplet_summary.json').write_text(json.dumps(groups,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated')
    args=parser.parse_args()
    reproduce(args.output)
