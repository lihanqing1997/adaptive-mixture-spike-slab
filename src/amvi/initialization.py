"""The ten-start MFVI initialization used in the paper."""
import time
import numpy as np
from .solver import RegressionModel, Mixture, best_meanfield, component_costs
from ._mixed_law import meanfield

def strong_meanfield(model,seed,extra_starts=8):
    began = time.perf_counter()
    basic = best_meanfield(model)
    best = dict(basic)
    rng = np.random.default_rng(seed)
    starts = [dict(kind='original_two_start_best',objective=basic['objective'],status=basic['status'])]
    for index in range(extra_starts):
        order = rng.permutation(model.p)
        reordered = RegressionModel(model.X[:,order],model.y,model.sigma,model.tau,model.omega)
        candidate = meanfield(reordered,start=index%2)
        inverse = np.argsort(order)
        q = candidate['mixture']
        candidate['mixture'] = Mixture(q.weights,q.a[:,inverse],q.mu[:,inverse],q.var[:,inverse])
        objective = float(component_costs(model,candidate['mixture'])[0])
        if not np.isclose(objective,candidate['objective'],rtol=1e-10,atol=1e-9):
            raise AssertionError('Column permutation changed the recomputed objective')
        candidate['objective'] = objective
        starts.append(dict(kind='permuted_order',index=index,objective=objective,status=candidate['status']))
        if objective < best['objective']:
            best = candidate
    return basic,best,dict(seconds=time.perf_counter()-began,starts=starts)
