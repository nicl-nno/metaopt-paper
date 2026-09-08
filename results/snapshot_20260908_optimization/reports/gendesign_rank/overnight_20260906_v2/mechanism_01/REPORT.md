# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 8 |
| in_distribution/pool12/ranknet_vs_mse | -0.17% | 8 |
| in_distribution/pool12/ranknet_vs_log_mse | -0.13% | 8 |
| in_distribution/pool12/ranknet_vs_pool_random | -9.74% | 8 |
| in_distribution/pool12/ranknet_vs_ea | -8.88% | 8 |
| in_distribution/pool48/ranknet_vs_mse | -0.41% | 8 |
| in_distribution/pool48/ranknet_vs_log_mse | -0.32% | 8 |
| in_distribution/pool48/ranknet_vs_pool_random | -12.84% | 8 |
| in_distribution/pool48/ranknet_vs_ea | -13.38% | 8 |
| in_distribution/pool48/ranknet_vs_untrained | -21.07% | 8 |
| in_distribution/pool48/ranknet_vs_shuffled | -15.32% | 8 |
| in_distribution/pool48/ranknet_vs_reverse | -25.29% | 8 |
| in_distribution/pool192/ranknet_vs_mse | -1.02% | 8 |
| in_distribution/pool192/ranknet_vs_log_mse | -0.29% | 8 |
| in_distribution/pool192/ranknet_vs_pool_random | -13.30% | 8 |
| in_distribution/pool192/ranknet_vs_ea | -13.67% | 8 |
| larger_grid/pool48/ranknet_vs_mse | -0.95% | 8 |
| larger_grid/pool48/ranknet_vs_log_mse | -0.22% | 8 |
| larger_grid/pool48/ranknet_vs_pool_random | -13.68% | 8 |
| larger_grid/pool48/ranknet_vs_ea | -12.34% | 8 |
| larger_grid/pool48/ranknet_vs_untrained | -18.89% | 8 |
| larger_grid/pool48/ranknet_vs_shuffled | -12.81% | 8 |
| larger_grid/pool48/ranknet_vs_reverse | -24.68% | 8 |
| in_distribution/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.18% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.13% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -2.44% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -0.28% | 8 |
| larger_grid/pool48/ranknet_explore0_vs_ranknet_explore1 | -1.13% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.74% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -2.15% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -1.66% | 8 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.901 | 0.628 | -25.52% |
| initial/log_mse | 0.899 | 0.649 | -25.59% |
| initial/ranknet | 0.898 | 0.646 | -25.57% |
| initial/untrained | 0.485 | 0.010 | +6.86% |
| initial/shuffled | 0.542 | 0.017 | -2.29% |
| near_ea_solution/mse | 0.860 | 0.431 | -7.25% |
| near_ea_solution/log_mse | 0.866 | 0.378 | -7.16% |
| near_ea_solution/ranknet | 0.873 | 0.469 | -7.37% |
| near_ea_solution/untrained | 0.522 | 0.010 | +5.05% |
| near_ea_solution/shuffled | 0.507 | 0.003 | -0.70% |

Equal online time: mean RankNet/EA compliance difference +6.53%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
