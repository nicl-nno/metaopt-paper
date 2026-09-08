# Exploratory ranking stress tests, 2026-09-08

Question: does RankNet screening improve the best truly evaluated design relative
to validation-selected MSE/log-MSE with the same GAMLET GraphSAGE model?
These tasks were chosen after the truss and adaptive-attack results were seen,
because discontinuous objectives may favor ranking. Report every chosen task,
including null/negative results. This is not a confirmatory benchmark selection.

1. Directional sensor placement: synthetic 2-D sites and demand points, six
   sensors at 24 possible sites, continuous orientations, hard 90-degree field
   of view, range 0.55, two opaque axis-aligned rectangular obstacles. Forty
   demand points have lognormal weights (sigma 1). Weighted collaborative
   coverage is 1-product(1-p), with p=0.95 exp(-distance/0.8) within visibility,
   range and field of view. Maximize weighted covered demand. Site swaps and
   angular mutations preserve exactly six sensors. This is a small adaptation
   inspired by Wu et al., arXiv:2501.07375v2, not their 3-D DEM benchmark or a
   reproduction of their hybrid optimizer. Static visibility is input geometry;
   features do not contain candidate coverage values or counts.
2. Cascade resilience: connected ER/BA graphs, n=32/48, degree-preserving rewiring
   with at most ceil(0.2m) new edges. Motter--Lai model (PRE 66,065102): capacities
   are (1+alpha) times each candidate's pre-attack unnormalized betweenness;
   remove the highest-load node, recompute shortest-path loads, synchronously
   remove overloaded nodes until stable. Maximize final LCC/n. Alpha is sampled
   uniformly from [0.1,0.5]. Ties use a fixed per-task random priority. This is
   topology design with capacities recalibrated per candidate, not fixed-capacity
   infrastructure protection; that distinction limits interpretation.

Allocation fixed before test outcomes: three independent offline-data seeds
26090871, 26090872, 26090873 per benchmark; each block 12 training, 4 validation,
12 test tasks. Balanced ER/BA and size cells in cascades. 128 offline designs per
training/validation task; 80 epochs, batch32, Adam1e-3, weight decay1e-4, clip5.
Model seeds17,23 and search seeds101,202. Actual RepositoryGraphSurrogate:
GraphSAGE3x32, task MLP16x16, scalar head32; same initial weights, batches,
optimizer steps, labels and validation criterion for all losses. Per-task
standardized deficit or log deficit regression; RankNet uses all pairs in the
same-task batch. Pick checkpoints and the MSE/log-MSE recipe by held-out
validation log regret, before test searches. No tuning from test outcomes.

Budget96 includes16 common initial evaluations. Archive16; pool48; select3 by
model and1 uniformly. Controls: EA (4 proposals), pool_random (4 of48), random
restart. Candidate duplicates cannot consume budget. Models stay frozen.
Two GNN model seeds per loss plus three controls =9 variants; 216 searches per
block, 648 per benchmark, 62,208 online evaluations per benchmark. Each
benchmark separately uses 6,144 offline true evaluations. No wall-clock claim.

Independent unit is the task, with all model/search seeds averaged within task;
tasks are nested in offline-data blocks. Primary effect is the mean task log
ratio of final objective scores (ranking/comparator), transformed to percent.
Report absolute differences and wins as well; near-zero objectives require
explicit reporting, not hidden deletion. Use conditional task sign tests with
Holm across eight comparisons (four comparators times two benchmarks), and
separate exact block sign tests. Three blocks cannot establish p<.05 in a
two-sided exact block test. Allocation is computational, not power based.
Archive config, task definitions, offline arrays, checkpoints, endpoint designs,
trajectories, model initial hashes, source hashes and independent oracle checks.
Smoke tests use separate seeds and do not contribute to the analysis. Resume
only from completed records with identical frozen configuration/source hashes.
