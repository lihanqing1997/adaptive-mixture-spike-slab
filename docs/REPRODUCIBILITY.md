# Reproduction and methods

## Recompute the reported summaries

```sh
python scripts/reproduce.py
python scripts/plot_results.py
```

These commands use only files in this repository. The first checks dataset counts,
reference eligibility counts, primary means and paired intervals,
complete-procedure comparisons, the time-budget checks, and all 210 local
cross-budget comparisons. Its report is `generated/verification.json`. Original
outputs are not overwritten. The second regenerates three figures under
`generated/figures/` from the same per-dataset numbers.

The stored outcomes are a selective export of the completed experiments, not a
newly fitted study. Raw chains, full optimizer traces, duplicate source freezes,
machine logs, old pilots and unused experiments are excluded. Summary
reproduction therefore verifies the reported calculations from the saved
per-dataset metrics; it does not independently recompute each metric from full
draws or rerun the integration assessments. This distinction also applies to the
saved event-level reference diagnostics.

## Explicit refits

These commands start computation and write a new result file:

```sh
python scripts/run_dataset.py p10_rho0.7_r000 --output generated/new-fit.json
python scripts/run_dataset.py p10_g2_rho0.7_r000 --method stagewise_boosting --output generated/stagewise.json
python scripts/run_local.py p10_g2_rho0.7_r000 --arm direct_joint --cap 16 --output generated/local.json
```

The first two regenerate the specified design using its original configuration
and seed, compute the ten-start MFVI baseline, run the selected complete procedure,
and save the fit and independent objective comparison. The local command loads
the dataset's original saved five-component candidate, regenerates its design,
and applies the original arm-specific refinement and acceptance rules. It saves
the fit, not a new independent posterior assessment. All commands refuse an
existing output file.

The complete-procedure runner uses the documented 60-second search allowance at
p=10 and 120 seconds otherwise. The local runner retains 60 seconds and changes
only the refresh cap. Final validation can extend total elapsed time. Hardware,
load, and floating-point differences can change a refit's optimizer path, so new
fits are not promised to equal historical parameters bit for bit. These utilities
are sequential; old machine-specific process coordinators are not included.

## Implementation map

| Module | Purpose |
| --- | --- |
| `amvi.solver` | Adaptive births, joint refinement, numerical comparison, and fallback |
| `amvi._mixed_law` | Only the distribution/cost/cache/MFVI definitions required by the current solver |
| `amvi.initialization` | The reported ten-start MFVI initializer |
| `amvi.comparators` | Complete frozen-refinement and stagewise procedures |
| `amvi.objectives`, `amvi.local` | Direct/augmented fixed-K objectives and the four local arms |
| `amvi.data` | One- and two-triplet designs |
| `amvi.exact` | Small-model conjugate enumeration |
| `amvi.references`, `amvi.diagnostics` | MCMC reference, retained fixed-marginal comparator, and diagnostic screens |
| `amvi._transport`, `amvi._sinkhorn` | Only the numerical helpers required by the retained correction |
| `amvi.metrics` | Posterior, selection, coverage and prediction summaries |

The source-selection record identifies the scientific definitions extracted from
the research code. Unrelated algorithms and launchers were removed. The binary
diagnostic includes the float conversion used by the actual reference runs.
Numerical unit tests cover gradients, information identities, restricted updates,
fallbacks, exact support probabilities, MCMC checks and fixed-marginal preservation.

No manuscript files or broad archive bundle are needed by these commands.
