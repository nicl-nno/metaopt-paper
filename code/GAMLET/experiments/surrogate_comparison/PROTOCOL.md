# Fixed-surrogate comparison, allocated before new test outcomes (2026-09-08)

Question: how does the proposed GNN RankNet screening compare with other
surrogate losses and model families on exactly the existing four benchmarks?
This is a post-hoc extension of already inspected benchmarks, not a newly
preregistered confirmatory study. All selected methods and outcomes are reported.

Methods, frozen before new test searches:
1. GNN ListMLE: negative Plackett–Luce log likelihood, averaged over the same
   within-task minibatch. Smaller scores mean better designs. Exact target ties
   use stable sorting of the already randomly shuffled minibatch; no extra RNG
   draws change the paired training schedule.
2. GNN Huber: delta=1 on within-task standardized targets, raw or log cost selected
   by validation. Same GraphSAGE model, initial weights, batches, 80 epochs,
   optimizer steps, Adam1e-3, weight decay1e-4 and clip5 as original GNN fits.
3. Random Forest: 128 trees, bootstrap, max_features=0.5, minimum leaf 1 or5.
4. RFF-GP: posterior mean of a 256-random-Fourier-feature RBF Gaussian process,
   equivalent to Bayesian linear regression in the finite feature space. Unit
   prior variance, noise variance0.01 or0.1, length scale0.5/1/2 times median
   distance of1024 training-only random pairs. Center/scale input descriptors
   using training data only. This is an approximate GP, not an exact GP or BO.

RF and RFF-GP compare raw/log costs and within-task/pooled standardization.
Finite tuning grids have8 RF and24 GP recipes; Huber has2 and ListMLE1.
For every recipe fit model seeds17,23; select one recipe per family/block by
mean validation finite-pool log regret across these seeds. Earliest checkpoint
and first recipe win exact ties. No test-task labels/statistics guide training,
normalization, model selection or acquisition. Model capacities/tuning costs
are not equal across classical and neural families and are explicitly reported.

Classical models receive a deterministic permutation-invariant descriptor of
the SAME four tensors supplied to GNNs: mean/std/min/max of original node
features and two row-normalized neighborhood-diffusion steps, mean absolute
original-minus-one-step features, context, log active-node and directed-edge
counts. Only active nodes/edges enter pooling. No fitness, betweenness loads,
simulated responses or task identifiers are added. This representation is not
a learned GNN and can lose graph information; differences mix representation
and model family. It supports varying node counts including200-node transfer.

Use every existing offline block and held-out source task: truss5×24=120;
network5×32=160 ID plus5×8=40 OOD; coverage3×12=36; cascades3×12=36.
Reuse exactly the existing offline arrays, with hashes and no new labels.
Common technical repeats: model seeds17,23 and search seeds1001,1002 for truss,
101,202 otherwise. Compare against the identical archived RankNet seed slice,
not against its wider-seed aggregate. Repeated seeds never add independent tasks.
This standardizes replication across four benchmarks without dropping tasks.

Original search code and oracles remain unchanged. Frozen screening,16 shared
initial evaluations, archive16, pool48, choose3 lowest scores+1 uniform random.
Truss B256; network B192 with primary96 checkpoint and32 fresh scenarios;
coverage/cascade B96. GP uncertainty does not change the acquisition rule.
No time-based termination or optional stopping. One CPU thread per process;
training-step equality is verified from original initial-weight hashes/records.

Expected new searches: truss1920, network3200, coverage576, cascade576 =6272.
Expected online true evaluations:491520+614400+55296+55296=1216512.
Offline evaluation cost is reused, not zero total historical cost. Independent
audits and network fresh scenarios incur separate calls, not search feedback.

Analyze paired task-level mean log ratios (RankNet/comparator), equal offline
block weights and equal network family/size strata. Negative compliance ratios
favor RankNet; positive utility ratios favor RankNet. Report both objective
ratio and improvement-oriented percentage, absolute differences, wins/ties and
individual block effects. Conditional stratified bootstrap10000 samples;
exact two-sided task sign tests with Holm over16 primary comparisons (4 methods
×4 benchmarks; network96 ID search bank). Network192/fresh/OOD descriptive.
Separate exact block sign tests: minimum p0.0625 for5 and0.25 for3 blocks.
No removal of unfavorable outcomes or zero values; zeros require explicit
additive reporting instead of logarithms. This allocation is computational,
not based on prospective power. Audits recompute all endpoints independently.

Smoke runs only check implementation, using two training/one validation/one
test task, two epochs and one seed; they are excluded from statistical evidence.
No tuning decision is made from smoke optimization outcomes. Freeze worker
sources and input hashes; resume only identical code/configuration. Histories,
all tuning records, selected model weights and all endpoints are archived.
