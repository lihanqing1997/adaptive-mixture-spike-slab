"""Probability-specific binary diagnostic used for reference eligibility."""
import numpy as np
from .references import diagnostic, _rhat_and_ess, _split

def binary_diagnostic(draws):
    x = np.asarray(draws, float)
    if x.ndim != 2 or not np.all((x == 0) | (x == 1)):
        raise ValueError('Binary draws must have chains x iterations shape')
    d = diagnostic(x)
    d.update(probability=float(x.mean()), chain_probabilities=x.mean(1).tolist())
    if d['constant']:
        d.update(raw_ess=None, probability_mcse=None, eligible=True,
                 note='Constant observed event; no mixing evidence or zero-error assertion.')
        return d
    # Event and complement have identical autocorrelation and probability MCSE.
    _, ess = _rhat_and_ess(_split(x))
    se = float(x.std(ddof=1)/np.sqrt(ess)) if ess > 0 else None
    d.update(raw_ess=float(ess), probability_mcse=se,
             eligible=bool(d['rhat'] is not None and d['rhat'] <= 1.01
                           and d['bulk_ess'] >= 400 and ess >= 400
                           and se is not None and se <= .01))
    d['note'] = 'Eligibility concerns probability estimation; tail quantile ESS not used.'
    return d
