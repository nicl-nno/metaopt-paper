# Reproducing the exploratory extensions

From the GAMLET root, with the existing optimization environment (NumPy2.2.6,
PyTorch2.8.0 CPU, PyG2.7, NetworkX3.6.1), use new output directories:

```
python -m unittest experiments.ranking_stress.test_problems
python -m experiments.ranking_stress.run --benchmark coverage --out experiments/ranking_stress/results/coverage_20260908_v1
python -m experiments.ranking_stress.run --benchmark cascade --out experiments/ranking_stress/results/cascade_20260908_v1
```

The named paths above are the exact paths expected by the downstream analysis.
Use a new checkout/result root for reproduction; never overwrite published
results. A completed worker can be resumed in its original directory only with
the identical configuration and worker source hashes. Smoke mode uses separate
data seed91911, one test task,20 evaluations and two training epochs; it is not
part of the scientific sample.

The primary workers train per-task MSE/log-MSE and RankNet. The following
separate addendum was introduced after seeing the coverage result; it checks
pooled normalization without altering the original evidence:

```
python -m experiments.ranking_stress_analysis.regression_addendum experiments/ranking_stress/results --benchmark coverage
python -m experiments.ranking_stress_analysis.regression_addendum experiments/ranking_stress/results --benchmark cascade
```

Both new regressors use the identical offline labels, initial model weights,
minibatch order, optimizer settings, checkpoint criterion and96-evaluation search
budget. The addendum uses one CPU thread rather than two; no wall-clock endpoint
or bitwise equality of trained floating-point weights is claimed. It adds288
searches and27648 online evaluations per benchmark, with no new offline labels.

In the statistical environment (NumPy2.2.6,SciPy1.16.1,NetworkX3.6.1,Matplotlib3.10.3):

```
python -m unittest experiments.ranking_stress_analysis.test_analysis
python -m experiments.ranking_stress_analysis.audit experiments/ranking_stress/results
python -m experiments.ranking_stress_analysis.analyze experiments/ranking_stress/results
python -m experiments.ranking_stress_analysis.analyze_addendum experiments/ranking_stress/results
python -m experiments.ranking_stress_analysis.coverage_example
```

`--watch` on the cascade audit and addendum analyzer waits for immutable
completed outputs; it never feeds outcomes back to search. Export only after
both primary campaigns, both regression addenda, their analyses and independent
audits finish. Complete allocation:1872 searches,179712 online objective calls,
12288 original offline calls, plus separately recorded post-search audits.

The standard-library export/verification helpers in the research repository
preserve exact bytes and archive member hashes. Main truss/network archives are
reused by reference in the combined manifest rather than recompressed.

Scientific verification remains the responsibility of the human authors.
Codex assisted with code, analysis and drafting; experimental-design procedures
in Scientific Agent Skills informed blocking, control matching and inference
separation. No model or tool is an author or has approved scientific claims.
