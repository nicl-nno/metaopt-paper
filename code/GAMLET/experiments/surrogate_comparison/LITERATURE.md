# Bounded comparator search, 2026-09-08

The user requested comparable surrogate methods on the same benchmarks, not a
systematic literature review or reproduction of complete published optimizers.
The installed research-lookup guidance informed source verification. Parallel
CLI/authentication were unavailable; the available web search/browser was used.
No private data were sent in search queries. Selection preceded new test results.

| Comparator | Primary source checked | Role and implementation boundary |
|---|---|---|
| GNN ListMLE | Xia, Liu, Wang, Zhang and Li, ICML2008, pp1192–1199, DOI10.1145/1390156.1390306; [official full paper](https://icml.cc/Conferences/2008/papers/167.pdf) | Plackett–Luce list likelihood, implemented with the existing GNN rather than the original information-retrieval features. Tests whether RankNet is preferable to another ordinal loss. |
| GNN Huber | [PyTorch HuberLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) | Quadratic near zero and linear for large errors, delta1 on standardized targets. A robust pointwise-regression control, not a claimed new optimizer. |
| Random Forest | Lindauer et al., SMAC3, JMLR23(54):1–9,2022, [full paper](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf); [official sklearn estimator](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | Established tree-based surrogate family. Our frozen RF screening is not SMAC3: no online retraining, expected improvement or SMAC search procedure. |
| RFF-GP mean | Rahimi and Recht, NIPS2007, [official paper](https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf) | Random Fourier features approximate the RBF kernel. Bayesian linear posterior mean in that finite space provides a scalable approximate GP regression surrogate; it is not exact full-kernel GP or Bayesian optimization. |

The existing Liu et al.2024 survey ([publisher](https://link.springer.com/article/10.1007/s40747-024-01465-5)) supports RF and GP as relevant surrogate families in expensive combinatorial optimization. It is contextual secondary literature; implementation follows primary papers/documentation above.

A maximum-clique GNN decoder, ResiNet and other learned construction policies
change the search mechanism and are not interchangeable fitness estimators.
Their complete reproduction would answer a different question. GP expected
improvement/LCB would also change acquisition; this comparison fixes acquisition
to isolate the surrogate as far as each representation permits.

The four models are not claimed to exhaust the literature or represent all
state-of-the-art methods. Classical methods use invariant graph descriptors
derived from GNN inputs, so cross-family effects cannot be attributed to loss
alone. Hyperparameter grids and limitations are explicit in PROTOCOL.md.

Workflow attribution (repository documentation, following the user's removal of
workflow citations from the manuscript): Kassis, Agarwal, He, Patel and Brueckner,
Scientific Agent Skills,2026, https://arxiv.org/abs/2609.00065.
