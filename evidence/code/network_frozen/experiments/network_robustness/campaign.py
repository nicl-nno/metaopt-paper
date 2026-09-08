"""Resumable, fixed-evaluation-budget independent replication campaign."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .experiment import features, model, train, search, save_csv, save_sources, atomic_json as atomic
from .problem import Task, NetworkDesign, make_tasks, graph_key, attack_priorities, initial_graph
from .fast_oracle import fast_robustness
from .audit import reference_robustness

CONTROLS = ("ea", "pool_random", "hillclimb", "random")


def read(path):
    for attempt in range(40):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def state(out, status, **kwargs):
    atomic(out/"state.json", {"status": status, "updated_utc": stamp(), "pid": os.getpid(), **kwargs})


def adj_from_edges(n, edges):
    adj = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        adj[a, b] = adj[b, a] = True
    return adj


def load_offline(tasks, cfg, out):
    folder = out/"offline"
    folder.mkdir(exist_ok=True)
    datasets, costs = {}, []
    offline_tasks = [t for t in tasks if t.split != "test"]
    for j, task in enumerate(offline_tasks):
        state(out, "offline", task=task.task_id, tasks_completed=j, tasks_total=len(offline_tasks))
        problem = NetworkDesign(task, cfg)
        path = folder/(task.task_id+".npz")
        if not path.exists():
            start = time.perf_counter()
            rng = np.random.default_rng(task.graph_seed+11000)
            graphs, seen = [problem.base.copy()], {graph_key(problem.base)}
            while len(graphs) < cfg["designs_per_task"]:
                graph = problem.sample(rng)
                key = graph_key(graph)
                if key not in seen:
                    graphs.append(graph)
                    seen.add(key)
            values = np.array([problem.evaluate(g) for g in graphs])
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, adjacency=np.stack(graphs), robustness=values,
                                priorities=problem.priorities, seconds=time.perf_counter()-start)
            os.replace(tmp, path)
        with np.load(path, allow_pickle=False) as data:
            datasets[task.task_id] = {"features": features(problem, data["adjacency"]),
                                     "values": data["robustness"].copy(), "split": task.split}
            costs.append({"task_id": task.task_id, "split": task.split,
                          "oracle_evaluations": len(data["robustness"]),
                          "attack_trajectories": len(data["robustness"])*cfg["attack_scenarios"],
                          "seconds": float(data["seconds"])})
        save_csv(out/"offline_cost.csv", costs)
        print(f"offline {j+1}/{len(offline_tasks)}: {task.task_id}", flush=True)
    return datasets


def run_identity(task_id, method, ms, ss):
    return f"{task_id}__{method}__{ms}__{ss}"


def expected_variants(cfg):
    return [(name, -1) for name in CONTROLS] + [(name, ms) for ms in cfg["model_seeds"] for name in cfg["losses"]]


@torch.no_grad()
def validation_diagnostics(models, datasets, out):
    records = []
    for (method, ms), net in models.items():
        for task_id, data in datasets.items():
            if data["split"] != "validation":
                continue
            values = data["values"]
            score = net(*data["features"]).numpy()
            difference = values[:, None]-values[None, :]
            valid = np.triu(difference != 0, 1)
            product = (score[:, None]-score[None, :])*difference
            accuracy = float(np.mean((product[valid] < 0)+.5*(product[valid] == 0))) if valid.any() else None
            records.append({"method": method, "model_seed": ms, "task_id": task_id,
                            "pair_accuracy": accuracy, "non_tied_pairs": int(valid.sum()),
                            "top1_log_regret": float(np.log(values.max()/values[int(score.argmin())]))})
    save_csv(out/"validation_diagnostics.csv", records)


def verify_source(root):
    for relative, digest in read(root/"source_hashes.json").items():
        repo = Path(__file__).resolve().parents[2]
        if hashlib.sha256((repo/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Live source differs from frozen campaign: {relative}")
        assert hashlib.sha256((root/"source_snapshot"/relative).read_bytes()).hexdigest() == digest


def audit_block(out, root):
    cfg = read(out/"config.json")
    tasks = [Task(**t) for t in read(out/"tasks.json")]
    verify_source(root)
    problems = {t.task_id: NetworkDesign(t, cfg) for t in tasks}
    training = [read(out/f"{m}_{ms}_metadata.json") for ms in cfg["model_seeds"] for m in cfg["losses"]]
    for ms in cfg["model_seeds"]:
        group = [r for r in training if r["model_seed"] == ms]
        assert len({r["initial_weights_sha256"] for r in group}) == 1
        assert len({r["optimizer_steps"] for r in group}) == 1
        assert len({r["parameters"] for r in group}) == 1
    vals = {m: float(np.mean([r["validation_log_regret"] for r in training if r["loss"] == m]))
            for m in cfg["losses"] if m != "ranknet"}
    selection = read(out/"baseline_selection.json")
    assert selection["baseline"] == min(vals, key=vals.get) and selection["test_labels_used"] == 0
    offline_count = 0
    for task in tasks:
        if task.split == "test":
            assert not (out/"offline"/(task.task_id+".npz")).exists()
            continue
        problem = problems[task.task_id]
        with np.load(out/"offline"/(task.task_id+".npz"), allow_pickle=False) as data:
            graphs = data["adjacency"]
            assert len(graphs) == len({graph_key(g) for g in graphs}) == cfg["designs_per_task"]
            np.testing.assert_array_equal(data["priorities"], problem.priorities)
            assert all(problem.valid(g) for g in graphs)
            # Reference the first offline graph of every source task.
            assert reference_robustness(graphs[0], problem.priorities)[0] == float(data["robustness"][0])
            offline_count += len(graphs)
    starts = defaultdict(set)
    checked, endpoints, records, per_run = 0, 0, [], []
    expected = {run_identity(t.task_id, m, ms, ss) for t in tasks if t.split == "test"
                for ss in cfg["search_seeds"] for m, ms in expected_variants(cfg)}
    paths = list((out/"runs").glob("*.json"))
    assert {p.stem for p in paths} == expected
    for path in paths:
        obj = read(path)
        row, history, design = obj["row"], obj["history"], obj["design"]
        task = Task(**design["task"])
        problem = problems[task.task_id]
        assert path.stem == run_identity(task.task_id, row["method"], row["model_seed"], row["search_seed"])
        assert row["task_id"] == task.task_id
        assert len(history) == row["evaluations"] == cfg["budget"]
        assert [r["evaluations"] for r in history] == list(range(1, cfg["budget"]+1))
        assert np.all(np.diff([r["best_r"] for r in history]) >= 0)
        assert history[-1]["best_r"] == row["best_r"] == design["best_r"]
        starts[task.task_id, row["search_seed"]].add(row["initial_graphs_sha256"])
        assert [cp["evaluations"] for cp in design["checkpoints"]] == cfg["evaluation_checkpoints"]
        fresh_pri = attack_priorities(task.n, task.attack_seed+1000000, cfg["fresh_scenarios"])
        for cp in design["checkpoints"]:
            adj = adj_from_edges(task.n, cp["edges"])
            graph = nx.from_numpy_array(adj)
            assert nx.is_connected(graph) and nx.number_of_selfloops(graph) == 0
            np.testing.assert_array_equal([graph.degree[i] for i in range(task.n)], problem.degree)
            assert np.count_nonzero(adj & ~problem.base)//2 <= problem.max_new_edges
            ref, curves = reference_robustness(adj, problem.priorities)
            actual, fast_curves = fast_robustness(adj, problem.priorities)
            np.testing.assert_array_equal(curves, fast_curves)
            assert ref == actual == cp["best_r"] == history[cp["evaluations"]-1]["best_r"]
            assert fast_robustness(adj, fresh_pri)[0] == cp["fresh_r"]
            fresh_ref, fresh_curve = reference_robustness(adj, fresh_pri[:1])
            fresh_actual, fresh_fast_curve = fast_robustness(adj, fresh_pri[:1])
            assert fresh_ref == fresh_actual
            np.testing.assert_array_equal(fresh_curve, fresh_fast_curve)
            records.append({"task_id": task.task_id, "family": task.family, "n": task.n,
                            "scope": "id" if task.n in cfg["sizes"] else "extrapolation",
                            "method": row["method"], "model_seed": row["model_seed"],
                            "search_seed": row["search_seed"], "budget": cp["evaluations"],
                            "best_r": cp["best_r"], "fresh_r": cp["fresh_r"],
                            "initial_r": row["initial_r"], "base_r": row["base_r"]})
            endpoints += 1
        checked += 1
        per_run.append(row)
        if checked % 50 == 0:
            state(out, "audit", verified_searches=checked, searches_total=len(paths))
            print(f"audit {checked}/{len(paths)}", flush=True)
    assert all(len(v) == 1 for v in starts.values())
    save_csv(out/"endpoint_results.csv", records)
    save_csv(out/"results.csv", per_run)
    result = {"passed": True, "verified_searches": checked, "verified_endpoints": endpoints,
              "offline_evaluations": offline_count, "online_evaluations": checked*cfg["budget"],
              "fresh_attack_trajectories": endpoints*cfg["fresh_scenarios"],
              "reference_attack_trajectories": endpoints*(cfg["attack_scenarios"]+1)
                  + sum(t.split != "test" for t in tasks)*cfg["attack_scenarios"],
              "audit_production_attack_trajectories": endpoints*(cfg["attack_scenarios"]+cfg["fresh_scenarios"]+1),
              "max_absolute_reference_discrepancy": 0, "source_hashes_match": True,
              "equal_initial_hashes": True, "equal_evaluation_budgets": True}
    atomic(out/"verification.json", result)
    return result


def worker(root, index):
    root = root.resolve()
    out = root/f"block_{index:02d}"
    cfg = read(out/"config.json")
    if (out/"COMPLETED.json").exists():
        return
    verify_source(root)
    spec = read(root/"campaign.json")
    assert cfg == dict(spec["config"], seed=spec["block_seeds"][index], name=f"{spec['name']}_block_{index:02d}")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    tasks = [Task(**t) for t in read(out/"tasks.json")]
    assert tasks == make_tasks(cfg)
    datasets = load_offline(tasks, cfg, out)
    models, training = {}, []
    for ms in cfg["model_seeds"]:
        for loss in cfg["losses"]:
            state(out, "training", method=loss, model_seed=ms, models_completed=len(training),
                  models_total=len(cfg["model_seeds"])*len(cfg["losses"]))
            metadata = out/f"{loss}_{ms}_metadata.json"
            if metadata.exists():
                row = read(metadata)
                net = model(cfg)
                net.load_state_dict(torch.load(out/f"{loss}_{ms}.pt", map_location="cpu", weights_only=True))
                net.eval()
            else:
                net, row = train(loss, ms, datasets, cfg, out)
                atomic(metadata, row)
            models[loss, ms] = net
            training.append(row)
            save_csv(out/"training_summary.csv", training)
    vals = {m: float(np.mean([r["validation_log_regret"] for r in training if r["loss"] == m]))
            for m in cfg["losses"] if m != "ranknet"}
    selection = {"baseline": min(vals, key=vals.get), "validation": vals, "test_labels_used": 0}
    if (out/"baseline_selection.json").exists():
        assert read(out/"baseline_selection.json") == selection
    else:
        atomic(out/"baseline_selection.json", selection)
    validation_diagnostics(models, datasets, out)
    del datasets
    folder = out/"runs"
    folder.mkdir(exist_ok=True)
    order = np.random.default_rng(cfg["seed"]+99000)
    variants = expected_variants(cfg)
    total = sum(t.split == "test" for t in tasks)*len(cfg["search_seeds"])*len(variants)
    done = 0
    for task in [t for t in tasks if t.split == "test"]:
        for ss in cfg["search_seeds"]:
            for i in order.permutation(len(variants)):
                method, ms = variants[i]
                path = folder/(run_identity(task.task_id, method, ms, ss)+".json")
                state(out, "search", task=task.task_id, method=method, model_seed=ms, search_seed=ss,
                      searches_completed=done, searches_total=total)
                if not path.exists():
                    net = models[method, ms] if ms != -1 else None
                    row, history, design = search(task, cfg, method, net, ms, ss)
                    pri = attack_priorities(task.n, task.attack_seed+1000000, cfg["fresh_scenarios"])
                    for cp in design["checkpoints"]:
                        cp["fresh_r"] = fast_robustness(adj_from_edges(task.n, cp["edges"]), pri)[0]
                    atomic(path, {"row": row, "history": history, "design": design})
                done += 1
            print(f"search {task.task_id} seed={ss}: {done}/{total}", flush=True)
    state(out, "audit", searches_completed=done, searches_total=total)
    result = audit_block(out, root)
    atomic(out/"COMPLETED.json", {"completed_utc": stamp(), **result})
    state(out, "completed", searches_completed=done, searches_total=total)


def initialize(spec_path, root):
    spec = read(spec_path)
    root.mkdir(parents=True, exist_ok=False)
    atomic(root/"campaign.json", spec)
    atomic(root/"STARTED.json", {"started_utc": stamp(), "spec": str(spec_path.resolve())})
    shutil.copy2(Path(__file__).with_name("CAMPAIGN_PROTOCOL.md"), root/"PROTOCOL.md")
    atomic(root/"source_hashes.json", save_sources(root))
    atomic(root/"environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
           "networkx": nx.__version__, "device": "cpu", "git_head": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], text=True).strip(), "executable": sys.executable})
    (root/"requirements-lock.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    manifest = []
    for i, seed in enumerate(spec["block_seeds"]):
        cfg = dict(spec["config"], seed=seed, name=f"{spec['name']}_block_{i:02d}")
        assert cfg["primary_budget"] in cfg["evaluation_checkpoints"]
        assert cfg["evaluation_checkpoints"] == sorted(set(cfg["evaluation_checkpoints"]))
        assert cfg["evaluation_checkpoints"][-1] == cfg["budget"]
        assert all(cfg["initial_evaluations"] <= b <= cfg["budget"] for b in cfg["evaluation_checkpoints"])
        out = root/f"block_{i:02d}"
        out.mkdir()
        atomic(out/"config.json", cfg)
        tasks = make_tasks(cfg)
        atomic(out/"tasks.json", [t.to_dict() for t in tasks])
        manifest.extend({"block": i, **t.to_dict()} for t in tasks)
        state(out, "pending")
    atomic(root/"task_manifest.json", manifest)
    # Compare structural identity against all campaign graphs and the original pilot.
    groups, hashes, checked = defaultdict(list), set(), 0
    pilot = Path(__file__).parent/"results/pilot_v1/tasks.json"
    entries = [("pilot", Task(**t)) for t in read(pilot)] if pilot.exists() else []
    entries += [(f"block_{r['block']:02d}", Task(**{k: v for k, v in r.items() if k != "block"})) for r in manifest]
    for block, task in entries:
        graph = initial_graph(task)
        key = graph_key(graph)
        assert key not in hashes, "Duplicate labeled source graph"
        hashes.add(key)
        signature = (task.n, tuple(sorted(graph.sum(1))))
        nxg = nx.from_numpy_array(graph)
        for previous in groups[signature]:
            assert not nx.is_isomorphic(nxg, previous), "Repeated isomorphic source graph"
        groups[signature].append(nxg)
        checked += 1
    atomic(root/"source_graph_audit.json", {"passed": True, "campaign_graphs": len(manifest),
              "graphs_including_pilot": checked, "no_repeated_isomorphic_sources": True,
              "test_graphs": sum(r["split"] == "test" for r in manifest),
              "manifest_sha256": hashlib.sha256((root/"task_manifest.json").read_bytes()).hexdigest()})
    atomic(root/"READY.json", {"ready_utc": stamp()})


def manager(spec_path, root, resume=False):
    root = root.resolve()
    if not resume:
        initialize(spec_path, root)
    else:
        assert read(root/"campaign.json") == read(spec_path)
        assert (root/"READY.json").exists(), "Incomplete initialization requires a new output directory"
    verify_source(root)
    spec = read(root/"campaign.json")
    lock_path = root/"manager.lock"
    if lock_path.exists():
        import psutil
        old = read(lock_path)
        if psutil.pid_exists(old["pid"]):
            raise RuntimeError("Campaign manager is already running")
        for child in old.get("child_pids", []):
            if psutil.pid_exists(child):
                raise RuntimeError("Previous worker still running; do not start duplicate workers")
    atomic(lock_path, {"pid": os.getpid(), "child_pids": []})
    pending = [i for i in range(len(spec["block_seeds"])) if not (root/f"block_{i:02d}/COMPLETED.json").exists()]
    running, attempts, failed = {}, defaultdict(int), []
    if os.name == "nt":
        import ctypes
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        # Scoped to this manager thread: permit display sleep, keep the requested computation running.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while pending or running:
            while pending and len(running) < spec["workers"]:
                index = pending.pop(0)
                attempts[index] += 1
                out = root/f"block_{index:02d}"
                log = (out/f"worker_attempt_{attempts[index]}_{time.time_ns()}.log").open("w", encoding="utf-8")
                env = dict(os.environ, OMP_NUM_THREADS=str(spec["config"]["threads"]),
                           MKL_NUM_THREADS=str(spec["config"]["threads"]), OPENBLAS_NUM_THREADS="1",
                           PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
                proc = subprocess.Popen([sys.executable, "-m", "experiments.network_robustness.campaign",
                                         "--out", str(root), "--worker", str(index)],
                                        stdout=log, stderr=subprocess.STDOUT, env=env,
                                        cwd=Path(__file__).resolve().parents[2],
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                log.close()
                running[index] = proc
                atomic(lock_path, {"pid": os.getpid(), "child_pids": [p.pid for p in running.values()]})
                print(f"started block {index}, pid={proc.pid}, attempt={attempts[index]}", flush=True)
            for index, proc in list(running.items()):
                if proc.poll() is None:
                    continue
                del running[index]
                done = (root/f"block_{index:02d}/COMPLETED.json").exists()
                if proc.returncode != 0 or not done:
                    if attempts[index] < 2:
                        pending.append(index)
                    else:
                        failed.append(index)
                print(f"finished block {index}: exit={proc.returncode}, verified={done}", flush=True)
            blocks = {f"block_{i:02d}": read(root/f"block_{i:02d}/state.json") for i in range(len(spec["block_seeds"]))}
            state(root, "running", active_blocks=list(running), pending_blocks=pending, failed_blocks=failed, blocks=blocks)
            atomic(lock_path, {"pid": os.getpid(), "child_pids": [p.pid for p in running.values()]})
            if pending or running:
                time.sleep(15)
        if failed:
            atomic(root/"FAILED.json", {"failed_blocks": failed, "updated_utc": stamp()})
            raise RuntimeError(f"Blocks failed after one retry: {failed}")
        state(root, "reporting")
        from .campaign_report import report
        report(root)
        atomic(root/"COMPLETED.json", {"completed_utc": stamp(), "blocks": len(spec["block_seeds"]),
                    "report": "REPORT.md"})
        state(root, "completed", blocks=len(spec["block_seeds"]))
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if not any(p.poll() is None for p in running.values()):
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("campaign.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.worker is None:
            manager(args.config, args.out, args.resume)
        else:
            worker(args.out, args.worker)
    except Exception as exc:
        folder = args.out if args.worker is None else args.out/f"block_{args.worker:02d}"
        if folder.exists():
            atomic(folder/"LAST_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc(), "utc": stamp()})
        raise
