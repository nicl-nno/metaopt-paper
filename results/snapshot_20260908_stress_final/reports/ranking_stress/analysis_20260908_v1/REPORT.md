# Exploratory coverage and cascade experiments

All preselected benchmarks are included. Each has 3 offline blocks,36 test tasks,648 searches,62208 online evaluations and6144 offline evaluations. Seeds are averaged within tasks; conditional tests do not establish generalization across new fitted blocks. No zero final objective was excluded.

| Benchmark | RankNet vs | Gain % | Conditional 95% CI | Wins/36 | Holm8 p | Block p |
|---|---|---:|---|---:|---:|---:|
|coverage|selected_regression|+5.691|[+4.340,+7.105]|33/36|1.82e-06|0.25|
|coverage|mse|+5.691|[+4.340,+7.105]|33/36|descriptive|0.25|
|coverage|log_mse|+4.121|[+2.903,+5.385]|30/36|descriptive|0.25|
|coverage|ea|+2.887|[+1.501,+4.249]|25/36|0.144|0.25|
|coverage|pool_random|+3.311|[+1.839,+4.836]|26/36|0.068|0.25|
|coverage|random|+7.498|[+5.976,+9.051]|33/36|1.82e-06|0.25|
|cascade|selected_regression|+4.378|[+2.380,+6.681]|20/36|0.2|0.25|
|cascade|mse|+3.869|[+2.505,+5.286]|24/36|descriptive|0.25|
|cascade|log_mse|+4.378|[+2.380,+6.681]|20/36|descriptive|0.25|
|cascade|ea|+5.458|[+2.786,+8.421]|21/36|0.2|0.25|
|cascade|pool_random|+4.140|[+1.572,+6.898]|16/36|0.856|0.25|
|cascade|random|+4.286|[+2.039,+6.743]|22/36|0.2|0.25|

Positive effects favor ranking; negative effects favor the comparator. Model-free controls reuse the same run across model seeds without adding independent replicates. These synthetic adaptations are not reproductions of the cited papers or claims of real deployment safety.
