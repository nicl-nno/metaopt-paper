# Exploratory structural-design pilot

Primary comparator selected on validation: **Regression (MSE)**.

RankNet compliance difference: **-11.50%**; task-bootstrap interval [-12.31%, -10.78%]. Negative favors ranking.

Independent test tasks: 6; paired model/search runs: 36. Repeated seeds are averaged within tasks, not counted as independent tasks.

| Method | Mean reduction from identical initial best | Mean online seconds |
|---|---:|---:|
| RankNet | 31.75% | 0.275 |
| Regression (MSE) | 22.84% | 0.273 |
| Regression (log MSE) | 21.73% | 0.267 |
| Evolution without surrogate | 17.73% | 0.021 |
| Random search | 11.98% | 0.016 |

## Scope

This pilot compares losses in one matched architecture and one fixed-volume truss family. All search choices use predictions and already evaluated candidates only. Test labels never select checkpoints or the regression baseline. The primary endpoint is compliance after an equal number of true evaluations; the global optimum is unknown. A small linear solver can be faster than GNN screening, so evaluation savings are not wall-clock acceleration.

Mandatory triangulation and area normalization guarantee the encoded stability/volume constraints; stress, buckling, fatigue, manufacturing and geometric nonlinearity are not modeled. Topology varies only in optional diagonals; unseen graph families and cross-domain transfer are not tested.

Intervals are exploratory and based on few tasks. No hyperparameters were tuned on these test outcomes. Publication claims require a frozen protocol and a new independent task set after pilot-driven changes.

![Convergence](convergence.png)

![Designs](designs.png)

Raw results: results.csv; per-evaluation trajectories: trajectories.csv; checkpoint selection: training_summary.csv; offline cost: offline_cost.json; configuration, task manifest, checkpoints and source hashes are included.
