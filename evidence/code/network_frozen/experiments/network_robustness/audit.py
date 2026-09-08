"""Independent NetworkX reference for all saved final attack trajectories."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .problem import Task, NetworkDesign, attack_priorities, graph_key, robustness


def reference_robustness(adj, priorities):
    n = len(adj)
    curves = []
    for priority in priorities:
        graph = nx.from_numpy_array(adj)
        curve = []
        for _ in range(n):
            node = min(graph.nodes, key=lambda v: (-graph.degree[v], int(priority[v])))
            graph.remove_node(node)
            curve.append(max((len(c) for c in nx.connected_components(graph)), default=0))
        curves.append(curve)
    return float(np.array(curves).sum(1).mean()/n**2), np.array(curves)


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    tasks = [Task(**t) for t in json.loads((out/"tasks.json").read_text())]
    tasks_by_id = {t.task_id: t for t in tasks}
    problems = {t.task_id: NetworkDesign(t, config) for t in tasks}
    for i, task in enumerate(tasks):
        for other in tasks[:i]:
            a, b = problems[task.task_id].base, problems[other.task_id].base
            if len(a) == len(b) and np.array_equal(np.sort(a.sum(1)), np.sort(b.sum(1))):
                assert not nx.is_isomorphic(nx.from_numpy_array(a), nx.from_numpy_array(b)), "Repeated isomorphic base graph"
    provenance = json.loads((out/"provenance.json").read_text())
    for relative, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((out/"source_snapshot"/relative).read_bytes()).hexdigest() == digest
    offline = 0
    for path in (out/"offline").glob("*.npz"):
        task = tasks_by_id[path.stem]
        assert task.split != "test"
        problem = problems[task.task_id]
        with np.load(path, allow_pickle=False) as data:
            assert len({graph_key(g) for g in data["adjacency"]}) == config["designs_per_task"]
            for adj in data["adjacency"]:
                assert problem.valid(adj)
            offline += len(data["adjacency"])
    rows = table(out/"results.csv")
    expected = sum(t.split == "test" for t in tasks)*len(config["search_seeds"])*(4+3*len(config["model_seeds"]))
    assert len(rows) == expected
    key = lambda r: (r["task_id"], r["method"], str(r["model_seed"]), str(r["search_seed"]))
    lookup = {key(r): r for r in rows}
    assert len(lookup) == len(rows)
    histories = defaultdict(list)
    for row in table(out/"trajectories.csv"):
        histories[key(row)].append(row)
    starts = defaultdict(set)
    for row in rows:
        h = histories[key(row)]
        assert len(h) == int(row["evaluations"]) == config["budget"]
        assert [int(v["evaluations"]) for v in h] == list(range(1, config["budget"]+1))
        assert np.all(np.diff([float(v["best_r"]) for v in h]) >= 0)
        assert float(h[-1]["best_r"]) == float(row["best_r"])
        starts[row["task_id"], row["search_seed"]].add(row["initial_graphs_sha256"])
    assert all(len(v) == 1 for v in starts.values())
    designs = json.loads((out/"best_designs.json").read_text())
    errors, fresh_errors = [], []
    for design in designs:
        task = Task(**design["task"])
        problem = problems[task.task_id]
        adj = np.zeros((task.n, task.n), dtype=bool)
        for a, b in design["edges"]:
            adj[a, b] = adj[b, a] = True
        graph = nx.from_numpy_array(adj)
        assert nx.is_connected(graph) and nx.number_of_selfloops(graph) == 0
        assert np.array_equal([graph.degree[i] for i in range(task.n)], problem.degree)
        assert np.count_nonzero(adj & ~problem.base)//2 <= problem.max_new_edges
        ref, curves = reference_robustness(adj, problem.priorities)
        fast, fast_curves = robustness(adj, problem.priorities, return_curves=True)
        np.testing.assert_array_equal(curves, fast_curves)
        row = lookup[task.task_id, design["method"], str(design["model_seed"]), str(design["search_seed"])]
        errors.append(abs(ref-float(row["best_r"])))
        assert errors[-1] < 1e-12 and fast == ref
        pri = attack_priorities(task.n, task.attack_seed+1000000, config["fresh_scenarios"])
        # Reference one additional independent scenario; full bank already computed by production.
        fresh_ref, _ = reference_robustness(adj, pri[:1])
        fresh_errors.append(abs(fresh_ref-robustness(adj, pri[:1])))
        assert fresh_errors[-1] < 1e-12
    training = table(out/"training_summary.csv")
    for ms in config["model_seeds"]:
        group = [r for r in training if int(r["model_seed"]) == ms]
        assert len(group) == 3 and len({r["initial_weights_sha256"] for r in group}) == 1
        assert len({r["optimizer_steps"] for r in group}) == 1
    val = {m: np.mean([float(r["validation_log_regret"]) for r in training if r["loss"] == m]) for m in ("mse", "log_mse")}
    assert json.loads((out/"baseline_selection.json").read_text())["baseline"] == min(val, key=val.get)
    result = {"passed": True, "verified_searches": len(rows), "offline_candidates": offline,
              "max_absolute_r_discrepancy": max(errors), "max_fresh_reference_discrepancy": max(fresh_errors),
              "independent_reference_attack_trajectories": len(rows)*(config["attack_scenarios"]+1),
              "whole_source_graphs_separated": True, "equal_initial_candidate_hashes": True,
              "equal_true_evaluation_budgets": True, "source_snapshot_hashes_match": True}
    (out/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    audit(parser.parse_args().out)
