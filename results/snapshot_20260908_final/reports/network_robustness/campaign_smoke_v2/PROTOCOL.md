# Independent replication: robustness optimization with a fixed evaluation budget

Protocol fixed on 7 September 2026, after the exploratory pilot and before generating this campaign's offline fitness or test search outcomes. The pilot is excluded from the analysis. This is a computational replication, not an externally preregistered study.

## Question and primary outcome

Does RankNet produce larger robustness R than validation-selected GNN regression at **96 exact evaluations**, including the common 16 initialization evaluations, on previously unseen ER/BA source networks of 50 and 100 nodes? A larger secondary budget of 192 is taken from the same search trajectory. No wall-clock matching or speed claim is part of this experiment, per the user's instruction. Budgets are fixed independently of observed effects.

R, degree-preserving rewiring, constraints, features, proposal mechanism and all four classical controls are as in README.md. The equivalent union-find oracle is used. Every exact evaluation averages the same four fixed attack scenarios within its task. Thus every method also has the same number of individual attack trajectories at a given search budget.

## Replication and allocation

Five independent blocks use the five seeds specified in campaign.json. Each block contains 32 training source graphs (8 per family/size cell), 12 validation source graphs (3 per cell), 32 in-distribution test graphs (8 per cell), and 8 size-extrapolation test graphs (4 per family, 200 nodes). Training uses sizes 50/100 only. All graph variants stay within their source graph's split. The full source manifest is generated before any model fitting; all source graphs are checked for repeated isomorphic graphs across blocks and against the pilot. No pilot labels or checkpoints enter the campaign.

The independent experimental units are the offline-data block and source graph, not search or initialization seeds. Each source graph has three search seeds; each GNN has two initialization seeds. All methods share initial candidates within a task/search seed. Run order is randomized within each task/search-seed block. Two OS worker processes may run concurrently; time is only operational telemetry.

The sample size is a fixed computational allocation, not a claimed power calculation. There are 160 new in-distribution and 40 extrapolation test source graphs, nested within five offline training blocks. Five top-level blocks still limit precision about variability across future training sets.

## Matched training and regression selection

Each training and validation task has 128 distinct feasible designs. RankNet, MSE, log-MSE, globally normalized MSE and globally normalized log-MSE use the same GAMLET GraphSAGE encoder, scalar head, node/context features, initial parameter values, task/batch order, optimizer settings and 80 full epochs. Pairwise RankNet comparisons stay within a task. Ordinary MSE variants standardize within a training task; global variants use mean and standard deviation pooled over training tasks only. No test-task normalization statistics or test fitness are supplied to a model.

Every checkpoint minimizes the same validation mean log(best R in pool / R selected by the surrogate). Each block selects one of the four regression recipes by mean validation regret across its two initialization seeds, before any test searches. Deterministic ties use the order in campaign.json. All five models are evaluated on test tasks and reported; selected-regression results are never chosen from test outcomes. Selection and checkpoint history are saved. Surrogates remain frozen during search.

## Budget accounting and sensitivity checks

There are 14 variants per test task/search seed: five losses times two model seeds, plus EA, random selection from the same proposal pool, hill climbing, and random restarts. This yields **8,400 search runs and 1,612,800 online exact evaluations** at budget 192; the 96-budget endpoint reuses prefixes. Across five blocks, training/validation consumes **28,160 additional offline exact evaluations**, identical data for all loss functions. These costs are reported separately and are not hidden in a claim about end-to-end sample efficiency.

The best graph at each budget is saved. After selection it is scored on 32 fresh priority scenarios shared across methods within the task. These are a held-out evaluation, not extra information for search or model selection. The 16,800 endpoint graphs therefore require 537,600 fresh attack trajectories. Audits require additional evaluations and are separately counted. Test trajectories are not used to change the protocol, budgets, hyperparameters, or stopping rule.

Primary endpoint: R on the four-scenario search bank at budget 96, in-distribution tasks, RankNet versus the per-block validation-selected regression. Prespecified secondary results: fresh-bank R, budget 192, transfer to 200 nodes, each individual regression, classical controls, and convergence with evaluation count. Secondary intervals are descriptive, without a multiple-testing significance claim. No assertion about a global optimum is made.

## Analysis

For each task, average log(R_rank / R_comparator) over matched model/search seeds. For classical controls the same control run is shared across both model seeds; it is not counted as two independent observations. Average task effects with equal weight within family/size cells, then average cells and finally the five offline blocks equally. Report 100*(exp(mean log ratio)-1), individual block effects, and source-task win counts.

Use 10,000 hierarchical bootstrap replicates: resample the five offline blocks and, within each sampled block and family/size cell, resample source graphs. Do not independently resample repeated seed results. The interval addresses nested offline/test variability but is approximate with five blocks. Show block effects explicitly. Main results use all five completed blocks; interim reports must be labelled incomplete. No stopping based on sign, magnitude or significance is allowed.

## Integrity, recovery and outputs

Save the full configuration and task manifest before offline evaluation, source hashes/snapshots, locked dependencies, offline graph/fitness arrays, training checkpoints and histories, validation selection, per-run histories and both endpoint designs. Each finished search is committed atomically and resumed by identity; an interrupted uncommitted search is rerun from its original seeds. Audit exact evaluation counts, common initial graph hashes, monotone incumbent quality, all graph constraints and every endpoint's four attack trajectories with independent NetworkX. Audit one fresh trajectory per endpoint and recompute the full fresh bank with the production evaluator. Completion is written only after the block audit and aggregate report succeed.

README.md records the literature and experimental-design workflow assistance already used for this benchmark. This extension preserves that attribution. Its hypotheses, source counts and statistical units were laid out with the same experimental-design skill.
