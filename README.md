# Adaptive mixture variational inference for spike-and-slab regression

Implementation, simulation results, and reproduction scripts for the accompanying
paper, **Adaptive Mixture Variational Inference for Spike-and-Slab Regression**.

## Setup

Python 3.12 was used for validation. From a virtual environment:

```sh
python -m pip install -r requirements-reproduce.txt
python -m pip install -e ".[test,reproduce]"
python -m pytest -q
python scripts/reproduce.py
python scripts/plot_results.py
```

The reproduction script recomputes means and paired intervals
from saved per-dataset outcomes, and writes reports to `generated/`. It does not
refit experiments or require the original research workspace. The plotting script
uses the same saved values with a clean layout; pixel-identical rendering of the
original figure PDFs is not promised.

## Contents

| Folder | Contents |
| --- | --- |
| `src/amvi/` | Adaptive mixture solver, local objective comparisons, complete fitting comparators, data designs, references, and diagnostics |
| `results/one_triplet/` | 400 dataset-level outcomes at p=10,20,30,100 |
| `results/two_triplets/` | 150 datasets with MFVI and three complete mixture procedures |
| `results/local_refinement/` | 480 fixed-K outcomes at caps 16 and 128, paired contrasts, stopping records, and 60 original candidates |
| `results/checks/` | Reported fitting-strategy, time-budget, and reference-reliability checks |
| `results/figures/` | Posterior-error, fitting-comparison, and KL-decomposition figures |
| `results/expected/` | Small saved comparison targets used to verify numerical reproduction |
| `scripts/` | Saved-result reproduction, figures, and explicit single-dataset refits |
| `tests/` | Numerical correctness tests for retained implementations |
| `docs/` | Result-to-paper map, methods and reproduction limits |

There are **550 independent datasets**. The local studies reuse 60 two-triplet
datasets; their 480 fits are not additional independent observations. The smaller
comparison and diagnostic checks also reuse existing data. Unfavorable outcomes,
reference exclusions, stopping decisions, and fallbacks are retained.

## Use the method

```python
from amvi import RegressionModel, Settings, fit
from amvi.initialization import strong_meanfield

model = RegressionModel(X, y, sigma=1.0, tau=1.0, omega=0.1)
_, baseline, _ = strong_meanfield(model, seed=123)
result = fit(model, Settings(max_components=10, max_seconds=60),
             seed=456, baseline=baseline)
q = result["mixture"]
inclusion_probabilities = q.weights @ q.a
posterior_mean = q.weights @ (q.a * q.mu)
```

Choose prior parameters for your application. Inputs are not standardized
automatically. `sigma` and `tau` are standard deviations. Check the returned
stopping status and fallback decision. The numerical acceptance checks do not
certify global convergence.

See [the result map](docs/RESULTS.md) and
[reproduction instructions](docs/REPRODUCIBILITY.md). Large raw chains, optimization
traces and historical backups are intentionally outside this code/results
distribution. The included metrics reproduce reported numerical summaries;
independent reanalysis of raw chains requires the original research data.
