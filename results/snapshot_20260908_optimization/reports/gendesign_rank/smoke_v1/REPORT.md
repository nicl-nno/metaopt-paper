# Exploratory structural-design pilot

Primary comparator selected on validation: **Regression (MSE)**.

RankNet compliance difference: **-13.64%**; task-bootstrap interval [-13.64%, -13.64%]. Negative favors ranking.

Independent test tasks: 1; paired model/search runs: 1. Repeated seeds are averaged within tasks, not counted as independent tasks.

| Method | Mean reduction from identical initial best | Mean online seconds |
|---|---:|---:|
| RankNet | 22.63% | 0.018 |
| Regression (MSE) | 10.40% | 0.020 |
| Regression (log MSE) | 10.40% | 0.019 |
| Evolution without surrogate | 8.41% | 0.004 |
| Random search | 10.76% | 0.004 |

## Scope

This pilot compares losses in one matched architecture and one fixed-volume truss family. All search choices use predictions and already evaluated candidates only. Test labels never select checkpoints or the regression baseline. The primary endpoint is compliance after an equal number of true evaluations; the global optimum is unknown. A small linear solver can be faster than GNN screening, so evaluation savings are not wall-clock acceleration.

Mandatory triangulation and area normalization guarantee the encoded stability/volume constraints; stress, buckling, fatigue, manufacturing and geometric nonlinearity are not modeled. Topology varies only in optional diagonals; unseen graph families and cross-domain transfer are not tested.

Intervals are exploratory and based on few tasks. No hyperparameters were tuned on these test outcomes. Publication claims require a frozen protocol and a new independent task set after pilot-driven changes.

![Convergence](convergence.png)

![Designs](designs.png)

Raw results: results.csv; per-evaluation trajectories: trajectories.csv; checkpoint selection: training_summary.csv; offline cost: offline_cost.json; configuration, task manifest, checkpoints and source hashes are included.
