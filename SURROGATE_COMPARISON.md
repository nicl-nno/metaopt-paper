# Сравнение с другими суррогатами

**Полная серия завершена и независимо проверена:** 6 272 новых поиска, 1 216 512 онлайн-оценок, без новых offline-меток. Все четыре метода и четыре бенчмарка включены, включая отрицательные и незначимые результаты.

[Численные результаты](results/snapshot_20260908_surrogate_comparison_final/reports/surrogate_comparison/comparison_20260908_v1/statistics.json), [аудит](results/snapshot_20260908_surrogate_comparison_final/reports/surrogate_comparison/comparison_20260908_v1/AUDIT.json), [составной манифест](results/snapshot_20260908_surrogate_comparison_final/MANIFEST.json). Текущий полный снимок: 20,629 исходных файлов в 143 архивах, включая архивы предыдущих серий.

[Методы и первичные источники](code/GAMLET/experiments/surrogate_comparison/LITERATURE.md), [фиксированный протокол](code/GAMLET/experiments/surrogate_comparison/PROTOCOL.md), [воспроизведение](code/GAMLET/experiments/surrogate_comparison/REPRODUCIBILITY.md).

# Other surrogate methods on the same benchmarks

Complete paired slice: two model seeds and two search seeds, all original offline blocks and test tasks. Positive advantage favors RankNet. No new offline labels. GNN fits use matched architecture/training steps; RF/GP use deterministic graph descriptors and validation-selected grids. Conditional task tests do not establish retraining generalization.

| Benchmark | Comparator | RankNet advantage % | Conditional 95% CI | Wins / tasks | Holm16 p | Block p |
|---|---|---:|---|---:|---:|---:|
|truss|listmle|+0.306|[+0.144,+0.478]|68/120|1|0.375|
|truss|huber|+0.537|[+0.416,+0.664]|87/120|7.81e-06|0.375|
|truss|rf|+16.196|[+15.715,+16.691]|120/120|2.26e-35|0.0625|
|truss|rff_gp|+12.767|[+12.474,+13.062]|120/120|2.26e-35|0.0625|
|network|listmle|+0.116|[-0.009,+0.241]|86/160|1|1|
|network|huber|+1.019|[+0.853,+1.190]|127/160|3.97e-13|0.0625|
|network|rf|+2.286|[+2.063,+2.501]|144/160|8.11e-26|0.0625|
|network|rff_gp|+1.750|[+1.587,+1.916]|154/160|4.83e-37|0.0625|
|coverage|listmle|+0.397|[-0.658,+1.420]|16/36|1|1|
|coverage|huber|+5.251|[+3.839,+6.843]|34/36|1.94e-07|0.25|
|coverage|rf|+5.266|[+3.736,+6.846]|32/36|1.55e-05|0.25|
|coverage|rff_gp|+7.125|[+5.531,+8.824]|35/36|1.18e-08|0.25|
|cascade|listmle|+1.930|[-0.033,+4.032]|22/36|0.561|1|
|cascade|huber|+1.527|[-0.646,+3.780]|19/36|1|0.25|
|cascade|rf|+2.300|[+0.455,+4.390]|19/36|1|0.25|
|cascade|rff_gp|-0.029|[-2.320,+2.127]|16/36|1|1|

The wider historical RankNet aggregates are not used as comparators for this reduced technical-seed slice. All null and negative outcomes remain. Network192/fresh-bank/OOD estimates are descriptive and available in statistics.json. RFF-GP is a finite-feature posterior mean, not full GP/BO; RF is not a reproduction of SMAC. Few training blocks, prior benchmark inspection and graph-descriptor information loss limit interpretation.
