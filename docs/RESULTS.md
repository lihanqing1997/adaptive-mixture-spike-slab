# Simulation results

This guide maps the reported experiments and supplementary checks to their
numerical records. Dataset counts and display labels are recorded in `scope.json`.

| Paper result or analysis | Included numerical support |
| --- | --- |
| Design table (`tab:designs`) | `results/datasets.json`, all 550 configurations and seeds |
| Posterior errors and paired changes (`tab:formal-posterior`, `tab:formal-paired`) | `one_triplet/records.json`, `two_triplets/records.csv` |
| Joint-uncertainty figure (`fig:joint-errors`) | 100 exact-reference one-triplet rows; `figures/05_original_joint_uncertainty.pdf` |
| Controlled contrasts and means (`tab:controlled-kl-contrasts`, `tab:controlled-local-means`) | `local_refinement/assessments_16.json`, `assessments_128.json`, arm means and within-budget contrasts |
| Refinement budget table (`tab:controlled-budget-within`) | `local_refinement/budget_comparison.csv`, all 210 endpoint/change comparisons |
| Complete procedures (`tab:two-group-errors`, `fig:refinement-effects`) | `two_triplets/records.csv`, `figures/07_two_group_paired_effects.pdf` |
| KL decomposition (`fig:two-group-decomposition`) | Per-dataset support and conditional KL in `two_triplets/records.csv`; figure 08 |
| One-group fitting-strategy check | `checks/fitting_strategies.json`, 120 fits on 40 datasets |
| p=100 search-time sensitivity | `checks/time_budgets.json`, 60 fits on 20 datasets |
| Reference reliability and sensitivity | `checks/reference_diagnostics.json`, 400 diagnostic records; six longer-chain comparisons in `checks/longer_references.json` |
| Local secondary endpoints and stopping | All ten endpoints at both caps, shared-scramble KL assessments, fallbacks and stopping records in `local_refinement/` |

One-triplet reference eligibility counts are 50,50,49,37,49,29,45,24 in increasing
`(p, rho)` order. All p=10 references are exact. At p=20,30,100, posterior-error
means use the recorded eligible MCMC references; objective and prediction outcomes
retain every dataset. Alternative diagnostic decisions are retained.

The manuscript explicitly retains additional comparator checks in its numerical
records. Accordingly, the one-triplet records include the existing MCMC and
fixed-marginal Sinkhorn summaries, and the relevant implementations are included.
This does not include earlier Xi-VI optimization studies or their pilots.

The three figure PDFs show joint uncertainty, complete fitting comparisons, and
the KL decomposition. Local-refinement results are provided as numerical records
with arm means, paired contrasts, and budget-sensitivity comparisons.

## Data fields

`delta` is the adaptive objective minus the MFVI objective. Within a dataset it is
also the reverse-KL difference. `pip`, `tv`, `mse`, and `covariance` denote PIP mean
absolute error, grouped-support total variation, noiseless test prediction MSE,
and coefficient covariance Frobenius error. `reference_eligible` governs primary
MCMC-based error summaries; `original_eligible` retains the alternative screen.

Two-triplet records use `posterior_kl`, `group_union_tv`, `pip_mae`,
`covariance_error`, and `prediction_mse`. Their support TV covers all six grouped
predictors. `support_kl` and `conditional_kl_estimate` sum to total KL.

Local-refinement metrics use `full_support_kl`, `full_support_tv`, and
`conditional_kl` for all ten predictors. Each arm uses its own objective for
acceptance; the independent direct-KL assessments do not select the returned fit.
The first four shapes coincide in nine initial candidates, and 13 candidates
required duplicate padding. These cases remain included.

Intervals are unadjusted descriptive 95% paired t intervals over independent
datasets within a comparison. Numerical integration uncertainty is distinct from
between-dataset uncertainty. No performance ordering is asserted beyond the
retained results, and no runtime advantage is claimed.
