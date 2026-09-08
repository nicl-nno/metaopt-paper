# Pooled-normalization robustness addendum

Added after the coverage outcome was seen. Descriptive analysis; primary comparisons and Holm family remain unchanged. Same offline labels, GNN architecture, initial weights, minibatches, optimizer steps,80 epochs and96 true evaluations; these extra fits use one CPU thread rather than the primary runs’ two. No new offline labels.

| Benchmark | RankNet vs | Gain % | Wins/36 | Block gains % |
|---|---|---:|---:|---|
|coverage|global_mse|+5.485|34/36|+6.162, +4.095, +6.213|
|coverage|global_log_mse|+5.537|35/36|+5.895, +5.007, +5.712|
|coverage|selected_among_four|+5.006|33/36|+6.162, +4.095, +4.772|
|cascade|global_mse|+2.342|17/36|+0.498, +1.282, +5.310|
|cascade|global_log_mse|+2.309|18/36|+4.220, +0.420, +2.323|
|cascade|selected_among_four|+2.342|17/36|+0.498, +1.282, +5.310|

All576 additional endpoint designs independently recomputed. Selection among four regressors uses validation only. This addendum cannot be represented as preregistered confirmatory evidence.
