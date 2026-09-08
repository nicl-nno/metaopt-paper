# Statistical tests for network robustness

See PLAN.md for inferential targets and the post-hoc status of this addendum. Code and dependencies are isolated from the frozen optimization campaign.

From the GAMLET repository root, using the separate environment:

```powershell
& 'D:/agents/gamlet/.venv-network-stats/Scripts/python.exe' -m unittest discover -s experiments/network_statistics/tests -v
& 'D:/agents/gamlet/.venv-network-stats/Scripts/python.exe' experiments/network_statistics/analyze.py --campaign experiments/network_robustness/results/replication_20260907_v1 --out experiments/network_statistics/results/new_analysis
```

Use a new output folder for each analysis. Only offline blocks with all 50/100-node task runs are included; partial blocks and 200-node extrapolation are not used by this script. Input hashes and endpoint values are snapshotted before testing. Both budgets and scenario banks are analyzed, with all nine comparator recipes retained. Regression selection always comes from the original validation record.

The additional `after_campaign.py` process waits for the original full-campaign completion marker, then runs this same analysis once on all five blocks. Its state is saved separately; it never changes or stops the optimization campaign. Intermediate p-values do not trigger any action.

The sign test addresses a probability of winning on a source task, conditional on the trained models. Mean-effect intervals are stratified bootstrap intervals. Generalization across training sets is tested separately on independent offline-block means. With five independent blocks an exact two-sided sign test cannot attain p<0.05; the minimum is 0.0625. The report does not replace this test with a task-level p-value.
