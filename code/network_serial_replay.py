"""Serial timing replay with a faster exact oracle; original outcomes preserved."""
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent/"GAMLET"
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from experiments.network_robustness import problem as problem_module
from experiments.network_robustness.problem import NetworkDesign, Task, robustness as original_robustness
from experiments.network_robustness.fast_oracle import fast_robustness
from experiments.network_robustness.experiment import search, model, save_csv, save_json
from experiments.network_robustness.audit import table


def run(source):
    start = time.monotonic()
    while not (source/"COMPLETED.json").exists():
        if (source/"FAILED.json").exists() or time.monotonic()-start > 3600:
            raise RuntimeError("Original pilot failed or timed out")
        time.sleep(5)
    out = source/"serial_fast_oracle_replay"
    out.mkdir(exist_ok=False)
    config = json.loads((source/"config.json").read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((source/"baseline_selection.json").read_text())["baseline"]
    tasks = [Task(**t) for t in json.loads((source/"tasks.json").read_text()) if t["split"] == "test"]
    designs = json.loads((source/"best_designs.json").read_text())
    checked = 0
    for d in designs:
        task = Task(**d["task"])
        problem = NetworkDesign(task, config)
        adj = np.zeros_like(problem.base)
        for a, b in d["edges"]:
            adj[a, b] = adj[b, a] = True
        fast, curves = fast_robustness(adj, problem.priorities)
        assert fast == d["best_r"]
        slow, slow_curves = original_robustness(adj, problem.priorities, return_curves=True)
        assert fast == slow
        np.testing.assert_array_equal(curves, slow_curves)
        checked += 1
    def replacement(adj, priorities, return_curves=False):
        r, curves = fast_robustness(adj, priorities)
        return (r, curves) if return_curves else r
    problem_module.robustness = replacement
    networks = {}
    ms, ss = config["model_seeds"][0], config["search_seeds"][0]
    for loss in ("ranknet", baseline):
        net = model(config)
        net.load_state_dict(torch.load(source/f"{loss}_{ms}.pt", weights_only=True))
        networks[loss] = net.eval()
    original = {(r["task_id"], r["method"], r["model_seed"], r["search_seed"]): r for r in table(source/"results.csv")}
    first = {}
    for task in tasks:
        first.setdefault((task.family, task.n), task)
    rows = []
    rng = np.random.default_rng(7171)
    for task in first.values():
        variants = [("ea", None, -1), ("hillclimb", None, -1), ("ranknet", networks["ranknet"], ms), (baseline, networks[baseline], ms)]
        for i in rng.permutation(len(variants)):
            method, net, seed = variants[i]
            row, _, design = search(task, config, method, net, seed, ss)
            ref = original[task.task_id, method, str(seed), str(ss)]
            assert row["best_r"] == float(ref["best_r"])
            assert row["initial_graphs_sha256"] == ref["initial_graphs_sha256"]
            rows.append(row)
        print("Serial replay:", task.task_id, flush=True)
    save_csv(out/"results.csv", rows)
    summary = {"checked_final_networks": checked, "every_attack_curve_exactly_equal": True,
               "replayed_searches": len(rows), "same_search_outcomes": True,
               "additional_oracle_evaluations": sum(r["evaluations"] for r in rows),
               "extra_equivalence_attack_trajectories": 2*checked*config["attack_scenarios"],
               "methods_mean_seconds": {m: float(np.mean([r["elapsed_seconds"] for r in rows if r["method"] == m])) for m in ("ranknet", baseline, "ea", "hillclimb")},
               "scope": "Serial replay, first task per family/size, first model/search seeds; not a new statistical replicate",
               "oracle_source_sha256": hashlib.sha256((REPO/'experiments/network_robustness/fast_oracle.py').read_bytes()).hexdigest()}
    save_json(out/"summary.json", summary)
    save_json(out/"COMPLETED.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
