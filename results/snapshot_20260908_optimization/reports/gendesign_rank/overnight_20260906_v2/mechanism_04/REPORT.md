# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 8 |
| in_distribution/pool12/ranknet_vs_mse | -0.34% | 8 |
| in_distribution/pool12/ranknet_vs_log_mse | +0.06% | 8 |
| in_distribution/pool12/ranknet_vs_pool_random | -11.11% | 8 |
| in_distribution/pool12/ranknet_vs_ea | -9.63% | 8 |
| in_distribution/pool48/ranknet_vs_mse | -0.18% | 8 |
| in_distribution/pool48/ranknet_vs_log_mse | +0.04% | 8 |
| in_distribution/pool48/ranknet_vs_pool_random | -13.95% | 8 |
| in_distribution/pool48/ranknet_vs_ea | -14.02% | 8 |
| in_distribution/pool48/ranknet_vs_untrained | -21.26% | 8 |
| in_distribution/pool48/ranknet_vs_shuffled | -14.65% | 8 |
| in_distribution/pool48/ranknet_vs_reverse | -25.54% | 8 |
| in_distribution/pool192/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool192/ranknet_vs_log_mse | -0.25% | 8 |
| in_distribution/pool192/ranknet_vs_pool_random | -14.44% | 8 |
| in_distribution/pool192/ranknet_vs_ea | -14.27% | 8 |
| larger_grid/pool48/ranknet_vs_mse | -0.74% | 8 |
| larger_grid/pool48/ranknet_vs_log_mse | -0.58% | 8 |
| larger_grid/pool48/ranknet_vs_pool_random | -15.91% | 8 |
| larger_grid/pool48/ranknet_vs_ea | -14.32% | 8 |
| larger_grid/pool48/ranknet_vs_untrained | -19.85% | 8 |
| larger_grid/pool48/ranknet_vs_shuffled | -13.85% | 8 |
| larger_grid/pool48/ranknet_vs_reverse | -25.43% | 8 |
| in_distribution/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.28% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.58% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -1.35% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -0.76% | 8 |
| larger_grid/pool48/ranknet_explore0_vs_ranknet_explore1 | -1.36% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.57% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -2.49% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -2.01% | 8 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.911 | 0.653 | -28.09% |
| initial/log_mse | 0.901 | 0.632 | -28.07% |
| initial/ranknet | 0.908 | 0.618 | -28.20% |
| initial/untrained | 0.481 | 0.010 | +8.33% |
| initial/shuffled | 0.516 | 0.021 | -4.34% |
| near_ea_solution/mse | 0.862 | 0.434 | -7.34% |
| near_ea_solution/log_mse | 0.864 | 0.427 | -7.31% |
| near_ea_solution/ranknet | 0.872 | 0.490 | -7.58% |
| near_ea_solution/untrained | 0.517 | 0.007 | +6.04% |
| near_ea_solution/shuffled | 0.498 | 0.007 | +1.00% |

Equal online time: mean RankNet/EA compliance difference +7.79%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
