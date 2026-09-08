# Mechanism checks

Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.

| Comparison | Difference | Tasks |
|---|---:|---:|
| in_distribution/pool4/ranknet_vs_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_log_mse | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_pool_random | +0.00% | 8 |
| in_distribution/pool4/ranknet_vs_ea | +0.00% | 8 |
| in_distribution/pool12/ranknet_vs_mse | -0.69% | 8 |
| in_distribution/pool12/ranknet_vs_log_mse | +0.18% | 8 |
| in_distribution/pool12/ranknet_vs_pool_random | -9.18% | 8 |
| in_distribution/pool12/ranknet_vs_ea | -8.20% | 8 |
| in_distribution/pool48/ranknet_vs_mse | -0.66% | 8 |
| in_distribution/pool48/ranknet_vs_log_mse | +0.27% | 8 |
| in_distribution/pool48/ranknet_vs_pool_random | -13.23% | 8 |
| in_distribution/pool48/ranknet_vs_ea | -12.23% | 8 |
| in_distribution/pool48/ranknet_vs_untrained | -20.21% | 8 |
| in_distribution/pool48/ranknet_vs_shuffled | -16.43% | 8 |
| in_distribution/pool48/ranknet_vs_reverse | -24.27% | 8 |
| in_distribution/pool192/ranknet_vs_mse | +0.02% | 8 |
| in_distribution/pool192/ranknet_vs_log_mse | +1.10% | 8 |
| in_distribution/pool192/ranknet_vs_pool_random | -12.87% | 8 |
| in_distribution/pool192/ranknet_vs_ea | -11.71% | 8 |
| larger_grid/pool48/ranknet_vs_mse | +1.07% | 8 |
| larger_grid/pool48/ranknet_vs_log_mse | +0.61% | 8 |
| larger_grid/pool48/ranknet_vs_pool_random | -13.91% | 8 |
| larger_grid/pool48/ranknet_vs_ea | -14.04% | 8 |
| larger_grid/pool48/ranknet_vs_untrained | -19.54% | 8 |
| larger_grid/pool48/ranknet_vs_shuffled | -15.93% | 8 |
| larger_grid/pool48/ranknet_vs_reverse | -24.16% | 8 |
| in_distribution/pool48/ranknet_explore0_vs_ranknet_explore1 | -0.62% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -2.21% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -1.56% | 8 |
| in_distribution/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -0.64% | 8 |
| larger_grid/pool48/ranknet_explore0_vs_ranknet_explore1 | -1.00% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_mse_explore1 | -0.59% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_online_log_mse_explore1 | -1.33% | 8 |
| larger_grid/pool48/online_ranknet_explore1_vs_ranknet_explore1 | -1.70% | 8 |

## Separate common-pool diagnostics

These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.

| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |
|---|---:|---:|---:|
| initial/mse | 0.901 | 0.597 | -26.42% |
| initial/log_mse | 0.911 | 0.635 | -26.79% |
| initial/ranknet | 0.902 | 0.608 | -26.64% |
| initial/untrained | 0.499 | 0.003 | +3.83% |
| initial/shuffled | 0.502 | 0.000 | +3.55% |
| near_ea_solution/mse | 0.851 | 0.302 | -6.66% |
| near_ea_solution/log_mse | 0.866 | 0.358 | -7.13% |
| near_ea_solution/ranknet | 0.869 | 0.292 | -6.91% |
| near_ea_solution/untrained | 0.518 | 0.010 | +7.32% |
| near_ea_solution/shuffled | 0.487 | 0.010 | +1.68% |

Equal online time: mean RankNet/EA compliance difference +7.50%. Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch.

Read effects jointly: a gain over random selection from the same pool supports useful screening; a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority.
