# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 1 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 1 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 1 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 1 |
| in_distribution/pool12/ranknet_vs_mse | +8.16% | 1 |
| in_distribution/pool12/ranknet_vs_log_mse | +8.16% | 1 |
| in_distribution/pool12/ranknet_vs_pool_random | +8.16% | 1 |
| in_distribution/pool12/ranknet_vs_ea | +8.16% | 1 |
| in_distribution/pool48/ranknet_vs_mse | -6.06% | 1 |
| in_distribution/pool48/ranknet_vs_log_mse | -6.06% | 1 |
| in_distribution/pool48/ranknet_vs_pool_random | +8.75% | 1 |
| in_distribution/pool48/ranknet_vs_ea | +16.93% | 1 |
| in_distribution/pool48/ranknet_vs_untrained | -6.06% | 1 |
| in_distribution/pool48/ranknet_vs_shuffled | -1.35% | 1 |
| in_distribution/pool48/ranknet_vs_reverse | -6.06% | 1 |
| in_distribution/pool192/ranknet_vs_mse | -9.48% | 1 |
| in_distribution/pool192/ranknet_vs_log_mse | -4.20% | 1 |
| in_distribution/pool192/ranknet_vs_pool_random | +1.34% | 1 |
| in_distribution/pool192/ranknet_vs_ea | +12.67% | 1 |
| larger_grid/pool48/ranknet_vs_mse | -1.34% | 1 |
| larger_grid/pool48/ranknet_vs_log_mse | +1.50% | 1 |
| larger_grid/pool48/ranknet_vs_pool_random | -1.34% | 1 |
| larger_grid/pool48/ranknet_vs_ea | -0.68% | 1 |
| larger_grid/pool48/ranknet_vs_untrained | +1.33% | 1 |
| larger_grid/pool48/ranknet_vs_shuffled | -1.34% | 1 |
| larger_grid/pool48/ranknet_vs_reverse | -1.34% | 1 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.527 | 0.000 | -5.35% |
| initial/log_mse | 0.519 | 0.000 | -5.35% |
| initial/ranknet | 0.529 | 0.000 | +4.63% |
| initial/untrained | 0.516 | 0.000 | +3.89% |
| initial/shuffled | 0.533 | 0.000 | -5.35% |
| near_ea_solution/mse | 0.500 | 0.000 | +2.66% |
| near_ea_solution/log_mse | 0.498 | 0.000 | +2.66% |
| near_ea_solution/ranknet | 0.575 | 0.000 | -6.53% |
| near_ea_solution/untrained | 0.504 | 0.000 | -1.49% |
| near_ea_solution/shuffled | 0.480 | 0.000 | +15.47% |

Equal online time: mean RankNet/EA compliance difference +30.48%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
