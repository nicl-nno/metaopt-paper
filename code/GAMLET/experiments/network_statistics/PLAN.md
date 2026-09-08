# Statistical addendum: fixed-budget network robustness

Added on 8 September 2026 at the user's request, after the first two blocks' in-distribution effect estimates had already been inspected. All tests in this addendum are therefore exploratory. The original campaign, endpoint selection, stopping rule and code are unchanged. A small p-value will not trigger early stopping or extra replication to cross a threshold.

The main campaign endpoint remains mean log(R_RankNet/R_selected_regression) at 96 real evaluations on 50/100-node test networks. The primary comparator is selected on validation independently in each offline-data block. Average all paired model/search seeds inside each source graph, then graph effects within family/size strata, then strata and offline blocks equally. Run seeds are not independent experimental units.

Analyze only offline blocks whose entire in-distribution test set has been saved. Include every such block, regardless of outcomes. Snapshot input identities, hashes and budget endpoint values. At the first look these are blocks 00/01, 64 independent source graphs conditional on the trained models. Future reports use the same script and new immutable output directories.

## Two inferential targets

1. **Generalization across offline training sets:** exact two-sided sign test on the independent offline-block mean effects. Null: a new block is equally likely to favor either method (conditional on a non-tie). Also give an exact two-sided block sign-flip test on the mean effect under sign-symmetry/exchangeability, and a block-level Student t-test/CI as a explicitly parametric sensitivity analysis. No task-level p-value is substituted for these block-level tests. With B=2, the smallest two-sided exact p is 0.5; with B=5 it is 0.0625. The parametric model cannot be validated with only two blocks; a non-rejecting normality diagnostic at five blocks would not establish normality either.

2. **New test source graphs conditional on the already trained models:** two-sided sign tests on per-source-graph effects, within each completed block and pooled with blocks treated as fixed. This tests win probability, not the exact mean percentage improvement. The binomial calibration assumes independent fair signs under the null; show results by family/size stratum because a single homogeneous population should not be silently presumed. Also report Wilcoxon signed-rank tests as sensitivity to using magnitudes, explicitly requiring symmetric differences under its null. All seeds stay averaged within a source graph. Conditional inference does not establish robustness to generating a new training set.

Report positive/negative/tied source counts, geometric relative effect, absolute R difference, median relative effect, mean/SD R of task averages, and confidence intervals. Use a stratified bootstrap with offline blocks held fixed for the conditional mean effect, and the campaign's hierarchical bootstrap (blocks plus source tasks within strata) for descriptive broader intervals. With very few blocks the latter can be misleadingly narrow; it must not override the block-level tests.

## Multiple testing, diagnostics and verification

The main two-sided comparison is specified above. For conservative reporting of this post-hoc addendum, additionally apply Holm correction to the fixed family of 36 conditional sign-test p-values: nine comparators (validation-selected regression, each of four individual regressions, and the four original non-surrogate controls), two budgets (96/192), and two scenario banks (search/fresh). Apply the same correction separately to the 36 block sign tests. Do not select the correction family after seeing p-values. Per-block/stratum sign tests and Wilcoxon/t sensitivities are explicitly descriptive; they do not constitute additional independent confirmations. Holm correction does not correct for looking at interim results, previously chosen model recipes or all earlier research decisions.

Use the statistical-analysis skill's normality and IQR diagnostic helpers on paired source-level log differences; plot all points and Q-Q diagnostics before interpreting tests. Do not remove outliers. Symmetry and independence are design/model assumptions and cannot be established by Shapiro-Wilk. No Levene test is needed for paired differences. Round log differences to 12 decimal places for tie/rank tests only; use unrounded values for point estimates and bootstrap.

Verify input completeness, non-finite values, common initialization hashes, exact budgets and all saved trajectory endpoints. For the primary ranking/regression comparison verify every source graph's saved endpoint at both budgets independently through NetworkX, plus recompute the fresh bank with the previously validated exact oracle. Record the scope and cost of this additional audit. These checks never select designs or influence ongoing search.

## References and research assistance

- SciPy sign-test implementation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
- SciPy paired signed-rank assumptions and tie handling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- SciPy paired permutation tests: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- The statistical-analysis skill informed the separation of inferential targets, diagnostics and reporting. Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 . This is attribution for research assistance, not evidence for the ranking hypothesis.
