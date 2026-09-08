# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 8 |
| in_distribution/pool12/ranknet_vs_mse | -0.03% | 8 |
| in_distribution/pool12/ranknet_vs_log_mse | -0.10% | 8 |
| in_distribution/pool12/ranknet_vs_pool_random | -10.20% | 8 |
| in_distribution/pool12/ranknet_vs_ea | -8.76% | 8 |
| in_distribution/pool48/ranknet_vs_mse | -0.34% | 8 |
| in_distribution/pool48/ranknet_vs_log_mse | -0.77% | 8 |
| in_distribution/pool48/ranknet_vs_pool_random | -13.72% | 8 |
| in_distribution/pool48/ranknet_vs_ea | -13.48% | 8 |
| in_distribution/pool48/ranknet_vs_untrained | -20.90% | 8 |
| in_distribution/pool48/ranknet_vs_shuffled | -14.80% | 8 |
| in_distribution/pool48/ranknet_vs_reverse | -25.69% | 8 |
| in_distribution/pool192/ranknet_vs_mse | -0.62% | 8 |
| in_distribution/pool192/ranknet_vs_log_mse | -1.59% | 8 |
| in_distribution/pool192/ranknet_vs_pool_random | -14.73% | 8 |
| in_distribution/pool192/ranknet_vs_ea | -13.81% | 8 |
| larger_grid/pool48/ranknet_vs_mse | +0.95% | 8 |
| larger_grid/pool48/ranknet_vs_log_mse | +0.10% | 8 |
| larger_grid/pool48/ranknet_vs_pool_random | -14.13% | 8 |
| larger_grid/pool48/ranknet_vs_ea | -12.25% | 8 |
| larger_grid/pool48/ranknet_vs_untrained | -19.02% | 8 |
| larger_grid/pool48/ranknet_vs_shuffled | -12.85% | 8 |
| larger_grid/pool48/ranknet_vs_reverse | -23.81% | 8 |
| in_distribution/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.23% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.04% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -1.49% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -0.34% | 8 |
| larger_grid/pool48/ranknet_explore0_vs_ranknet_explore1 | -1.40% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -1.81% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -0.95% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -1.46% | 8 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.904 | 0.608 | -26.64% |
| initial/log_mse | 0.895 | 0.573 | -26.27% |
| initial/ranknet | 0.895 | 0.587 | -26.73% |
| initial/untrained | 0.474 | 0.007 | +8.79% |
| initial/shuffled | 0.565 | 0.021 | -4.66% |
| near_ea_solution/mse | 0.847 | 0.333 | -6.97% |
| near_ea_solution/log_mse | 0.841 | 0.312 | -6.65% |
| near_ea_solution/ranknet | 0.849 | 0.354 | -7.04% |
| near_ea_solution/untrained | 0.524 | 0.049 | +4.34% |
| near_ea_solution/shuffled | 0.524 | 0.010 | -0.54% |

Equal online time: mean RankNet/EA compliance difference +9.00%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
