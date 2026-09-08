# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 8 |
| in_distribution/pool12/ranknet_vs_mse | +0.19% | 8 |
| in_distribution/pool12/ranknet_vs_log_mse | +0.37% | 8 |
| in_distribution/pool12/ranknet_vs_pool_random | -9.63% | 8 |
| in_distribution/pool12/ranknet_vs_ea | -9.38% | 8 |
| in_distribution/pool48/ranknet_vs_mse | +0.04% | 8 |
| in_distribution/pool48/ranknet_vs_log_mse | -0.05% | 8 |
| in_distribution/pool48/ranknet_vs_pool_random | -14.29% | 8 |
| in_distribution/pool48/ranknet_vs_ea | -14.29% | 8 |
| in_distribution/pool48/ranknet_vs_untrained | -21.93% | 8 |
| in_distribution/pool48/ranknet_vs_shuffled | -15.39% | 8 |
| in_distribution/pool48/ranknet_vs_reverse | -26.22% | 8 |
| in_distribution/pool192/ranknet_vs_mse | -0.29% | 8 |
| in_distribution/pool192/ranknet_vs_log_mse | -0.43% | 8 |
| in_distribution/pool192/ranknet_vs_pool_random | -15.09% | 8 |
| in_distribution/pool192/ranknet_vs_ea | -14.81% | 8 |
| larger_grid/pool48/ranknet_vs_mse | -1.97% | 8 |
| larger_grid/pool48/ranknet_vs_log_mse | -1.66% | 8 |
| larger_grid/pool48/ranknet_vs_pool_random | -15.87% | 8 |
| larger_grid/pool48/ranknet_vs_ea | -15.65% | 8 |
| larger_grid/pool48/ranknet_vs_untrained | -20.27% | 8 |
| larger_grid/pool48/ranknet_vs_shuffled | -15.81% | 8 |
| larger_grid/pool48/ranknet_vs_reverse | -25.56% | 8 |
| in_distribution/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.41% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -1.95% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -1.63% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -0.20% | 8 |
| larger_grid/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.87% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.90% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -2.58% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -1.05% | 8 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.903 | 0.722 | -25.50% |
| initial/log_mse | 0.902 | 0.705 | -25.43% |
| initial/ranknet | 0.896 | 0.667 | -25.23% |
| initial/untrained | 0.492 | 0.010 | +6.89% |
| initial/shuffled | 0.510 | 0.010 | +0.72% |
| near_ea_solution/mse | 0.874 | 0.431 | -7.48% |
| near_ea_solution/log_mse | 0.883 | 0.469 | -7.65% |
| near_ea_solution/ranknet | 0.874 | 0.483 | -7.73% |
| near_ea_solution/untrained | 0.521 | 0.003 | +7.03% |
| near_ea_solution/shuffled | 0.500 | 0.014 | +0.62% |

Equal online time: mean RankNet/EA compliance difference +6.99%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
