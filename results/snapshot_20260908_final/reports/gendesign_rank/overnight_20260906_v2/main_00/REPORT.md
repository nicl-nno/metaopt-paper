# Exploratory structural-design pilot

Primary comparator selected on validation: **Regression (log MSE)**.

RankNet compliance difference: **-0.54%**; task-bootstrap interval [-0.62%, -0.46%]. Negative favors ranking.

Independent test tasks: 24; paired model/search runs: 1200. Repeated seeds are averaged within tasks, not counted as independent tasks.

| Method | Mean reduction from identical initial best | Mean online seconds |
|---|---:|---:|
| RankNet | 32.61% | 0.879 |
| Regression (MSE) | 32.23% | 0.880 |
| Regression (log MSE) | 32.24% | 0.876 |
| Evolution without surrogate | 22.71% | 0.054 |
| Random search | 10.65% | 0.042 |

| True evaluation budget | RankNet vs validation-selected regression |
|---|---:|
| 32 | -0.14% |
| 96 | -0.46% |
| 256 | -0.54% |

## Scope

This pilot compares losses in one matched architecture and one fixed-volume truss family. All search choices use predictions and already evaluated candidates only. Test labels never select checkpoints or the regression baseline. The primary endpoint is compliance after an equal number of true evaluations; the global optimum is unknown. A small linear solver can be faster than GNN screening, so evaluation savings are not wall-clock acceleration.

Mandatory triangulation and area normalization guarantee the encoded stability/volume constraints; stress, buckling, fatigue, manufacturing and geometric nonlinearity are not modeled. Topology varies only in optional diagonals; unseen graph families and cross-domain transfer are not tested.

Intervals are exploratory and based on few tasks. No hyperparameters were tuned on these test outcomes. Publication claims require a frozen protocol and a new independent task set after pilot-driven changes.

![Convergence](convergence.png)

![Designs](designs.png)

Raw results: results.csv; per-evaluation trajectories: trajectories.csv; checkpoint selection: training_summary.csv; offline cost: offline_cost.json; configuration, task manifest, checkpoints and source hashes are included.
