# Evidence map — working draft, 2026-09-08

All entries are machine-inspected draft support; accountable human verification is **pending**. No submission approval is implied. Paths below are relative to this directory. Source hashes and original local paths are in `source_manifest.json`.

| Claim group / manuscript locator | Evidence / locator | Scope |
|---|---|---|
| Matched model, loss and training — Section 3 | `code/experiments/gendesign_rank/model.py`, `experiment.py`; `code/gamlet/surrogate/encoders/`; `truss/main_*/training_summary.csv`; network frozen source | Same encoder/adapter within a domain; node input dimension differs between domains; not the legacy pipeline-specific training wrapper |
| Truss formulation and input features — Section 4.1 | `code/experiments/gendesign_rank/truss.py`, `truss/main_*/config.json` | Linear pin-jointed fixed-volume structures; limited optional topology |
| Completed truss primary effects, table 1, abstract/conclusion | `truss/SUMMARY.json`, `effects/{primary,mse,log_mse,ea,random}` | Five independent training-data blocks, 120 test contexts, budget 256; negative compliance delta favors ranking |
| Truss verification and counts | `truss/main_*/verification.json`, configs and completion markers | 4,080 searches per block; offline=(24+8)×512×5=81,920; online=20,400×256=5,222,400; controls do not multiply independent task count |
| Mechanism controls and common pools — Section 5.2 | `truss/REPORT.md`, `truss/mechanism_*/mechanism_summary.json`, `protocol.json` | Separate diagnostic contexts; descriptive aggregation; completed lower-budget/transfer/time analyses remain in source reports, not falsely labeled pending |
| Network problem and allocation — Section 4.2 | `network_protocol/PROTOCOL.md`, `campaign.json`, `source_graph_audit.json`; frozen network source | Five-block allocation is planned protocol, not a completion claim; 4 search scenarios and 32 fresh scenarios |
| Interim network effects and all printed p-values — Section 5.3 / table 3 | `network_interim/statistics.json`, `comparisons/96/best_r/selected_regression` and 96/192, search/fresh entries | Exactly blocks 00/01, 64 ID graphs; task seeds aggregated; conditional estimates separated from training-block tests |
| Interim audit | `network_interim/primary_comparison_audit.json` | 1,536 endpoint graphs in the primary paired comparison; not an audit of the entire unfinished five-block campaign |
| Assumptions and multiplicity | `network_interim/PLAN.md`, `assumption_diagnostics.json`, `code/experiments/network_statistics/analyze.py` | Post hoc exploratory addendum; Holm family 36; no correction for interim peeking; Wilcoxon sensitivity only |
| Pending complete network effects — Section 5.4 | No final numerical evidence used | TODO N1–N4 must remain until the full audited aggregate and final analysis are available |
| Inherited authors | Original ZIP, `template.tex`, author block | Author order retained; applicability to revised study and declarations pending |

## Literature checked against public primary records

Public bibliographic metadata and relevant text were opened on 2026-09-08. This is not a systematic review or a claim to include every 2026 paper. The draft uses 2025 work directly relevant to the hypothesis rather than asserting that ranking surrogates are novel.

| BibTeX key | Primary source and support |
|---|---|
| burges2005 | Microsoft Research publication page: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/ — neural pairwise ranking; cited as the verified technical report |
| runarsson2006 | Author institution record: https://iris.hi.is/en/publications/ordinal-regression-in-evolutionary-computation-parallel-problem-s/ — ordinal regression in evolutionary computation, 2006, pp. 1048–1057; cited only as prior existence |
| volz2016 | https://arxiv.org/abs/1611.00260 — abstract on partial-order surrogate support for exact evaluation in survival selection |
| lobanov2024 | https://arxiv.org/abs/2402.09014v3 — current v3 title replaces the obsolete title in the old bibliography; order access is not a learned GNN guarantee |
| hamilton2017 | https://arxiv.org/abs/1706.02216 — GraphSAGE, NIPS 2017 |
| liu2024 | https://link.springer.com/article/10.1007/s40747-024-01465-5 — author list, journal metadata and survey scope |
| tan2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/768c19273e20fa09147885d03da7550f-Abstract-Conference.html and linked full paper — ranking for offline MBO, no adaptive online oracle loop |
| tom2025 | https://arxiv.org/abs/2410.09290 and https://arxiv.org/html/2410.09290v1 sections 2–3 — ranking/regression including GNN; metadata confirms APL Machine Learning 3, 036113 (2025), DOI 10.1063/5.0272663; publisher redirect inaccessible through browser fetch, author preprint read |
| wu2025 | https://arxiv.org/abs/2501.07375v2 — RankNet-inspired hybrid coverage optimization; cited as a preprint |
| louzada2013 | https://arxiv.org/abs/1303.5269 — targeted-attack rewiring and journal DOI; no claim to reproduce the optimizer |
| wang2019 | https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Surrogate-Assisted-Robust-Optimization-of-Large-scale-Networks/99513647102346 — institution's primary publication record and abstract; early-online date 2019 is used, rather than guessing final issue metadata |
| yang2023 | https://github.com/yangysc/ResiNet — official authors' repository and BibTeX confirm Kaili Ma (secondary metadata reverses the name), title, TMLR 2023 and OpenReview ID; OpenReview currently presents a browser challenge |
| kassis2026 | https://arxiv.org/abs/2609.00065 — research workflow attribution only; not evidence for optimization results |

## Human review still required

Verify every scientific claim and source locator, the preserved authorship, new funding/declarations, and the final venue format. The author-verification state has deliberately not been upgraded by successful compilation or numerical checks.
