# Ranking vs regression for evolutionary structural design

A self-contained **exploratory pilot** for the hypothesis:

> With the same evaluated training candidates and online evaluation budget,
> a ranking-trained graph surrogate finds lower-compliance structures than
> a regression-trained surrogate with the same architecture.

The original `pilot.json` uses a small standalone graph encoder. The new
`pilot_repo.json` and overnight experiments **directly import GAMLET's
`gamlet.surrogate.encoders.simple_graph_encoder.SimpleGNNEncoder`**. Regression
and ranking both use that same GraphSAGE encoder (3 layers, 32 channels), task
MLP and scalar head; only the training loss changes. The adapter removes absent
members before PyG batching and mean pooling. Input embedding is linear because
mechanical features are continuous. Dropout is zero in all models.

The legacy `PipelineDatasetSurrogateModel` training shell contains unresolved
merge-conflict markers in this checkout. We reuse its working GNN encoder
directly, with a common mechanical-task MLP/head, rather than importing that
shell. AutoML weights are not transferred. The original pilot remains preserved.

## Run

From the GAMLET repository root, with Python 3.12:

```powershell
py -3.12 -m venv .venv-gendesign
.venv-gendesign/Scripts/python.exe -m pip install -r experiments/gendesign_rank/requirements.txt
.venv-gendesign/Scripts/python.exe -m unittest discover -s experiments/gendesign_rank/tests -v
.venv-gendesign/Scripts/python.exe -m experiments.gendesign_rank.experiment --config experiments/gendesign_rank/smoke.json --out experiments/gendesign_rank/results/smoke_v1
.venv-gendesign/Scripts/python.exe -m experiments.gendesign_rank.experiment --config experiments/gendesign_rank/pilot.json --out experiments/gendesign_rank/results/pilot_v1
.venv-gendesign/Scripts/python.exe -m experiments.gendesign_rank.audit experiments/gendesign_rank/results/pilot_v1
```

In the current workspace the environment is already installed at
`D:/agents/gamlet/.venv-gendesign/Scripts/python.exe` (one directory above GAMLET).
Use that interpreter for the commands above. Each run requires a **new** output
directory and preserves previous results. The smoke run only checks execution.
`requirements-lock.txt` records the complete environment used for the first run;
install it with `--extra-index-url https://download.pytorch.org/whl/cpu` to resolve
the pinned CPU torch build.

## Frozen pilot protocol

`pilot.json` was defined before observing test outcomes. Do not overwrite the
pilot configuration to make its observed result more favorable. Changes based on
this pilot require a new experiment name, disclosed changes, and new held-out
tasks for confirmatory claims.

### Mechanical problem

- A 4 x 3 grid of 12 pin-jointed nodes; fixed translations on the left boundary.
- 23 mandatory members (9 horizontal, 8 vertical, 6 diagonals) form a stable
  triangulated backbone. Six optional opposite diagonals vary topology.
- Mandatory area weights are integers 1..4; optional weights are 0..4.
- Physical areas are `A_e = V * w_e / sum_e(L_e * w_e)`, with `V = 0.03 m^3`.
  Every candidate has exactly the same material volume. Uniform rescaling of
  all genes is canonicalized to prevent duplicate physical evaluations.
- The true objective to minimize is compliance `C = f^T u`, in joules, with
  `K u = f` and Young's modulus 210 GPa. Small-displacement linear axial elements
  are assembled and solved in float64, without artificial stiffness or delays.
- Task contexts vary span (3..6 m), height (1..2.5 m), load magnitude (50..150 kN),
  load direction (-0.35..0.35 rad around downward), and load sharing between
  the upper and lower right corners (0.15..0.85).
- Bar intersections do not add implicit joints. This is an idealized benchmark;
  buckling, stress limits, fatigue, fabrication and nonlinear geometry are outside
  the model. The benchmark tests constrained compliance optimization, not a
  certified minimum-mass structure. Stability is enforced by the encoding.

### Data and transfer

| Split | Independent task contexts | Evaluated candidates per task |
|---|---:|---:|
| Train | 12 | 256 |
| Validation | 4 | 256 |
| Test | 6 | No offline labels; online evaluations only |

All models see exactly the same 3072 training evaluations. The 1024 validation
evaluations are separate and counted in offline cost. Train and validation
canonical genotypes are also globally disjoint. Entire task contexts are held
out; individual pair comparisons are not treated as independent tasks.

Test contexts come from the same task-generating distribution. This tests
in-distribution transfer to new geometries/load settings, **not** extrapolation
to different graph families or transfer of AutoML weights to mechanics.

### Shared network and learning

Each truss member becomes a node in a line graph. Two nodes exchange messages
when their members share a physical joint. Optional absent members are masked
from adjacency, message passing and readout. Node inputs: normalized endpoints,
direction, length, cross-section, material fraction, endpoint forces and supports.
Task context is encoded by a separate MLP. No displacements, fitness, or solver
outputs appear in the input features.

The same 3-layer, 32-channel mean-neighbor graph network, context MLP and scalar
output are used for all three losses:

1. `mse`: regression of compliance, standardized separately within each training
   task using only its training observations.
2. `log_mse`: regression of log compliance, similarly standardized. This is an
   intentionally stronger scale-robust regression comparator.
3. `ranknet`: binary cross-entropy on pairwise score differences, with lower
   score meaning better compliance. All unordered pairs within the same-task
   batch are used; ties receive target 0.5.

For each of two model seeds, initial weights, sample batches, optimizer, learning
rate, epoch count, and gradient-clipping rule are the same. Ranking reuses pairs
from the same evaluated candidates; it obtains no additional oracle observations.
Its extra loss computation is included in recorded offline training time.

All models train for 40 epochs. Checkpoints are selected by the same metric:
mean log regret of the selected validation candidate relative to the best
candidate in that task's fixed validation pool. This is validation pool regret,
not global-optimum regret. The regression variant for the **primary** comparison
is chosen by mean validation regret across model seeds, before any test searches.
No target scaler is fitted on a test task; deployment uses score order only.

### Search

Methods: ranking surrogate, both regression surrogates, evolutionary search
without a surrogate, and random search.

- 96 distinct true mechanical evaluations per search, including initialization.
- Identical initial set of 16 designs, including a uniform all-member reference.
- Parents are the best 16 **truly evaluated** designs; never predicted-only elites.
- Uniform crossover probability 0.25; mutation changes 1..3 area-weight genes.
- Each surrogate screens 48 offspring and evaluates the best 3 predictions plus
  1 randomly selected remaining candidate. All three surrogate methods use the
  identical acquisition and variation procedure.
- Evolution without a surrogate produces and evaluates 4 offspring each round;
  random search draws 4 independent designs. More cheap proposals for the
  surrogate methods are an explicit part of the procedure, not uncounted oracle
  evaluations. Proposal/screening time is recorded.
- Surrogates stay frozen during search. Their arbitrary score scales are never
  mixed with true compliance values.
- Three search seeds per task. Surrogate searches are repeated for both model
  seeds; model-independent baselines run once per task/search seed.

The pilot comprises 144 searches and 13,824 online mechanical evaluations.
Offline generation and training are counted separately. Online time includes
initialization, variation, feature construction, screening and actual solves;
it excludes offline training and already loaded model construction. Timing
should be interpreted cautiously: a tiny linear solve is usually faster than
GNN screening, and this benchmark must not be advertised as wall-clock
acceleration based only on evaluation efficiency.

### Endpoints and interpretation

Primary endpoint: final compliance after 96 true evaluations, RankNet versus
the regression variant selected **on validation**. Lower is better.

Within each task, first average paired log compliance ratios across model/search
seeds. Then average across tasks and exponentiate. This gives a geometric mean
relative effect. The reported 95% interval is an exploratory bootstrap of **task
blocks**, not of individual seeds or graph pairs. Six tasks do not provide a
strong confirmatory inference. Intervals are conditional on the fixed training
data and fitted models; they do not quantify uncertainty over new training sets.
No claim of superiority follows from a ranking
metric alone.

Secondary results: both individual regression baselines, search without a
surrogate, random search, best-so-far curves, reduction from the shared initial
best, online time, and concrete candidate designs. The global optimum is unknown.
The displayed structures always use the first test task, first model seed and
first search seed; they are not selected to make one method look best.

## Outputs and verification

The output directory includes config and task manifest, dataset and source
hashes, offline evaluations, checkpoint files, learning curves, validation-only
baseline selection, all final designs, per-evaluation trajectories, JSON summary,
`REPORT.md`, and PNG/PDF figures. `COMPLETED.json` appears only after training,
search and reporting have completed successfully. Raw result directories are
ignored by Git; archive them separately for publication.
`audit.py` checks budgets, shared initialization, validation-only selection,
source hashes, disjoint offline candidates and independently reassembles the
finite-element system for every saved final structure. These post-run verification
solves do not influence search choices and are recorded separately in
`verification.json`.

Tests cover an analytical axial bar, independent conventional finite-element
assembly, strain energy consistency, load/material/volume scaling, stable
variable topologies, rank orientation/ties, graph permutation invariance,
absent-member masking, equal initial populations, exact oracle budget, and
deterministic candidate selection.

The initial protocol intentionally avoids reusing the legacy implementation's
possible issues with metric signs, validation/test overlap, or surrogate fitness
mixing. This experiment can later be integrated with GAMLET after the hypothesis
has been assessed.

## Literature context

- Damke and Huellermeier, *Ranking Structured Objects with Graph Neural Networks*
  (2021): https://arxiv.org/abs/2104.08869 . Establishes graph ranking as an
  existing approach; our primary endpoint is downstream optimization quality.
- Nourian et al., *Design Optimization of Truss Structures Using a Graph Neural
  Network-Based Surrogate Model* (2023): https://doi.org/10.3390/a16080380 .
  Relevant engineering precedent, including cases where fewer solves do not
  yield lower wall time. This pilot is not a reproduction of their benchmark.

## Next experiment

The extended series is implemented in `nightly.py`, with mechanism checks in
`mechanism.py`. See [NIGHT_PROTOCOL.md](NIGHT_PROTOCOL.md) for its frozen scope,
controls, counting rules and stopping procedure. All workers execute a saved
source snapshot. The manager stops at completion, failure, a `STOP` file, or
the wall-time cap; reports update after completed blocks.

Before a publication run, use this pilot to check learning stability, available
headroom beyond the starting population, and whether a more challenging family
is needed. Choose any adjustments on training/validation data and freeze them.
Then evaluate on a fresh independent task manifest, with more model seeds and
task contexts. Add graph-family extrapolation, online retraining, and different
search budgets only as explicit extensions; do not silently fold them into the
pilot comparison.
