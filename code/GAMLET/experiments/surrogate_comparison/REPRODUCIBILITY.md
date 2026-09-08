# Other surrogate methods: reproduce the comparison

This extension compares GNN RankNet with GNN ListMLE, GNN Huber, Random
Forest and a random-Fourier-feature GP posterior mean. Read `PROTOCOL.md`
for the fixed allocation and `LITERATURE.md` for method attribution and
the limits of the implementation. All four original benchmarks and all
their offline blocks are retained. The new comparison uses two model
seeds and two search seeds per test task for every method, including the
matching slice of the archived RankNet results.

## Environment and archived inputs

Restore the combined result snapshot with the publication repository's
`tools/research_artifacts.py --extract code/GAMLET`. The existing truss,
network, coverage and cascade offline datasets are required; the worker
does not silently regenerate or relabel them. Its `input_hashes.json`
identifies each actual input. Run commands from the restored GAMLET root.

Python 3.12 and CPU PyTorch were used. Create a separate virtual environment,
then install the optimization dependencies from `code/requirements-optimization.txt`
and the additional pinned packages below. `requirements-lock.txt` records
the complete observed environment, including unrelated transitive packages.
For the `torch==2.8.0+cpu` build, use the official PyTorch CPU package index.

```powershell
python -m pip install scipy==1.16.1 scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0
python -m unittest experiments.surrogate_comparison.test_models experiments.surrogate_comparison.test_analysis -v
```

The original local installation reused an existing optimization environment
through a `.pth` file. That local path is not needed in a clean environment.

## Technical checks

For each `kind` in `truss`, `network`, `coverage`, `cascade`, the replay command
loads the original RankNet checkpoint and repeats an archived search through
the adapter. It must reproduce the stored outcome and available trajectory
and initialization checks. Smoke runs use two epochs and a tiny offline/task
subset, but the original online budgets. They are excluded from inference.

```powershell
python -m experiments.surrogate_comparison.replay --kind coverage --out experiments/surrogate_comparison/results/new_preflight
python -m experiments.surrogate_comparison.worker --kind coverage --block 0 --smoke --out experiments/surrogate_comparison/results/new_smoke
```

Repeat those commands for the other three kinds. Keep smoke output names
containing `smoke` if using the aggregate smoke auditor.

## Full campaign

Use a fresh output directory. A complete run performs 6,272 new searches and
1,216,512 online true evaluations, reusing the previously acquired offline
labels. Training, independent audits and fresh attack scenarios have separate
computational costs. They do not increase the online search budget or feed
new labels back to a surrogate. Four single-threaded workers were used.

```powershell
python -m experiments.surrogate_comparison.schedule --out experiments/surrogate_comparison/results/new_comparison --workers 4
python -m experiments.surrogate_comparison.audit experiments/surrogate_comparison/results/new_comparison
python -m experiments.surrogate_comparison.analyze experiments/surrogate_comparison/results/new_comparison
```

Alternatively, start the auditor and analyzer in separate terminals with
`--watch` while the scheduler runs. They wait for completion markers; a
failed worker does not count as a completed block. Check `state.json`, block
logs and any `LAST_ERROR.json` if the campaign stops. The scheduler can
resume completed families and searches, but frozen worker source hashes
must match. Do not modify source data or configurations during a run.

All neural fits verify the same initial-weight hash, parameter count and
optimizer-step count as the corresponding historical GNN fit. Model recipe
selection uses only validation finite-pool regret, averaged over both model
seeds. Every recipe's validation record and selected checkpoint is retained.

## Interpretation and outputs

`REPORT.md` and `statistics.json` report all 16 primary comparisons with
Holm adjustment. A positive advantage favors RankNet: relative compliance
reduction for trusses and relative utility increase for the other tasks.
Four technical pairs are averaged within each independent test task.
Confidence intervals and task sign tests condition on the fitted offline
blocks. Separate block sign tests expose the limited retraining replication:
their smallest possible two-sided p-values are 0.0625 with five blocks and
0.25 with three blocks. Network budget-192, fresh-bank and 200-node transfer
analyses are descriptive. All positive, null and negative outcomes remain.

`AUDIT.json` confirms the independently recomputed endpoints and valid
designs, original initial populations, budgets and source/input hashes.
`analysis_input_hashes.json` identifies the exact old and new records used.
`ANALYSIS_COMPLETED.json` is written only after the complete audited analysis.

The comparison isolates the loss within the shared GNN. RF and RFF-GP use
fixed permutation-invariant graph descriptors, so their differences also
reflect representation and capacity. RF is not a reproduction of SMAC;
RFF-GP is neither an exact GP nor a complete Bayesian-optimization method.
All methods use the same frozen top-three-plus-random screening policy.

Research workflow assistance used ScientificAgentSkills experimental-design
and research-lookup guidance. Author scientific review remains necessary.
