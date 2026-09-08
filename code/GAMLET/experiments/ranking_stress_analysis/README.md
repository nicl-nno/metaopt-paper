# Analysis and audit

Analysis is kept outside the frozen worker module. Both preselected benchmark
families must finish before the aggregate analysis can run; all outcomes are
included. From the GAMLET root, with the statistical dependencies installed:

```
python -m experiments.ranking_stress_analysis.analyze experiments/ranking_stress/results
python -m experiments.ranking_stress_analysis.audit experiments/ranking_stress/results
```

The audit additionally needs PyTorch/PyG through the original graph-design
imports only if imported transitively by future code; the current oracle module
uses NumPy/NetworkX. It compares every final cascade design against a separate
enumeration of all shortest paths and every coverage design against a scalar
oracle, checks source split isomorphism, frozen source hashes, training matching,
and fixed-budget accounting. These post-search calls never guide candidate
selection. `verification.json` is required before export of the supplement.

Conditional tests use 36 source tasks per benchmark after averaging both model
and search seeds. Holm covers eight predeclared primary comparisons; raw MSE
and log-MSE comparisons remain descriptive. Three independent offline blocks
are reported separately. No claim of adequate power or p<0.05 across training
blocks follows from this computational allocation.
