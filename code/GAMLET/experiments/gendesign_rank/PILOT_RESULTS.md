# First pilot result (2026-09-06)

The frozen `pilot.json` run completed successfully. This is preliminary evidence
for the ranking hypothesis within one small fixed-volume truss family.

## Primary finding

At **96 true evaluations per search**, RankNet achieved **11.50% lower compliance**
than the MSE regressor selected using validation data. The geometric mean paired
ratio was aggregated within task over model/search seeds, then across six test
tasks. The task-bootstrap exploratory 95% interval was **10.78% to 12.31% lower**.
Ranking had lower task-averaged compliance on all six tasks. These intervals are
conditional on the training sample and fitted models, not a confirmatory claim.

RankNet also achieved 12.75% lower compliance than log-MSE regression, 17.04%
lower than evolution without screening, and 22.43% lower than random search.
The regression baseline was selected before accessing any test objectives;
both regression variants are reported.

| Method | Mean improvement from the shared initial best | Mean online time |
|---|---:|---:|
| RankNet | 31.75% | 0.275 s |
| MSE regression | 22.84% | 0.273 s |
| Log-MSE regression | 21.73% | 0.267 s |
| Evolution without surrogate | 17.73% | 0.021 s |
| Random search | 11.98% | 0.016 s |

**There is no wall-clock speedup over the nonsurrogate search in this pilot.**
The linear mechanical solve is cheap. The demonstrated advantage is final
quality at a fixed expensive-evaluation budget; for an application where solves
really are expensive, wall-clock benefit remains to be measured.

## Evidence and checks

- 12 training tasks, 4 validation tasks, 6 independent test contexts.
- 4096 offline mechanical evaluations, including validation; no offline test labels.
- 2 model seeds and 3 search seeds; 36 paired runs per surrogate comparison.
- 144 searches and 13,824 online evaluations across all methods.
- Same 8801-parameter network, initialization per model seed, sample schedule,
  1920 optimizer steps and validation-selection criterion for each loss.
- Seven meaningful mechanics/learning/search tests passed.
- All 144 saved final structures rechecked with independent element assembly;
  maximum relative discrepancy in compliance: 7.35e-14.
- Maximum volume discrepancy: 2.43e-17 m^3.
- Exact budgets, shared initial populations, source hashes and globally disjoint
  offline canonical designs passed the result audit.
- Both generated figures were visually inspected.

The initial pilot did not undergo test-driven tuning or selective removal of
unfavorable tasks. MSE uses within-training-task standardization; log MSE is
included as a scale-robust alternative. No exhaustive hyperparameter tuning was
performed for either learning objective. A publication run should allow a fair,
validation-only tuning budget and use fresh held-out tasks after any pilot-based
changes.

## Files

- [Full run report](results/pilot_v1/REPORT.md)
- [Convergence and task effects](results/pilot_v1/convergence.png)
- [Example final structures](results/pilot_v1/designs.png)
- [Raw results](results/pilot_v1/results.csv)
- [Per-evaluation history](results/pilot_v1/trajectories.csv)
- [Independent verification](results/pilot_v1/verification.json)
- [Configuration and task manifest](results/pilot_v1/tasks.json)

## Scope of the next run

Keep this pilot as a fixed exploratory result. The most useful next step is to
predefine a balanced validation-only tuning procedure for the losses and a fresh,
larger test task sample. A second topology family would test whether the effect
survives a less restricted design representation. Neither superiority on other
optimization problems nor structural certification follows from this pilot.
