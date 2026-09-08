"""Controlled pilot: GNN ranking versus regression for robust graph design."""
import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F
import torch_geometric

from experiments.gendesign_rank.model import RepositoryGraphSurrogate, ranknet_loss, tensors
from experiments.gendesign_rank.experiment import save_csv, save_json
from .problem import NetworkDesign, Task, attack_priorities, graph_key, make_tasks, propose, robustness

LOSSES = ("mse", "log_mse", "ranknet")


def atomic_json(path, value):
    """Replace a progress/checkpoint JSON, tolerating transient Windows readers."""
    path = Path(path)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(100):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def features(problem, graphs):
    return tensors(tuple(np.stack(v) for v in zip(*(problem.features(g) for g in graphs))))


def model(config):
    return RepositoryGraphSurrogate(node_dim=6, context_dim=6, hidden=config["hidden"], layers=config["layers"])


def prepare(tasks, config, out):
    datasets, costs = {}, []
    folder = out/"offline"
    folder.mkdir()
    for task in tasks:
        if task.split == "test":
            continue
        tick = time.perf_counter()
        problem = NetworkDesign(task, config)
        rng = np.random.default_rng(task.graph_seed+11000)
        graphs = [problem.base.copy()]
        seen = {graph_key(problem.base)}
        while len(graphs) < config["designs_per_task"]:
            graph = problem.sample(rng)
            key = graph_key(graph)
            if key not in seen:
                seen.add(key)
                graphs.append(graph)
        values = np.array([problem.evaluate(g) for g in graphs])
        np.savez_compressed(folder/(task.task_id+".npz"), adjacency=np.stack(graphs), robustness=values,
                            priorities=problem.priorities)
        datasets[task.task_id] = {"features": features(problem, graphs), "values": values, "split": task.split}
        costs.append({"task_id": task.task_id, "split": task.split, "oracle_evaluations": len(graphs),
                      "attack_trajectories": len(graphs)*config["attack_scenarios"],
                      "seconds": time.perf_counter()-tick, "min_r": values.min(), "max_r": values.max()})
        save_csv(out/"offline_cost.csv", costs)
        print(f"offline {task.task_id}: R in [{values.min():.4f}, {values.max():.4f}]", flush=True)
    return datasets


@torch.no_grad()
def validate(net, datasets):
    net.eval()
    return float(np.mean([np.log(d["values"].max()/d["values"][int(net(*d["features"]).argmin())])
                          for d in datasets.values() if d["split"] == "validation"]))


def train(loss_name, seed, datasets, config, out):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = model(config)
    initial = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in net.parameters())).hexdigest()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    training = [v for v in datasets.values() if v["split"] == "train"]
    raw_targets = [np.log(1-d["values"]) if loss_name in ("log_mse", "global_log_mse")
                   else 1-d["values"] for d in training]
    global_values = np.concatenate(raw_targets)
    targets = [torch.tensor((y-global_values.mean())/max(global_values.std(), 1e-8)
                           if loss_name.startswith("global_") else (y-y.mean())/max(y.std(), 1e-8),
                           dtype=torch.float32) for y in raw_targets]
    history, best, best_state, best_epoch = [], float("inf"), None, 0
    tick, steps = time.perf_counter(), 0
    for epoch in range(config["epochs"]):
        net.train()
        for i in rng.permutation(len(training)):
            data = training[i]
            # Lower deficit is better. MSE on deficit equals MSE on R after sign reversal.
            y = targets[i]
            order = rng.permutation(len(y))
            for offset in range(0, len(y), config["batch_size"]):
                idx = order[offset:offset+config["batch_size"]]
                pred = net(*(x[idx] for x in data["features"]))
                loss = ranknet_loss(pred, y[idx]) if loss_name == "ranknet" else F.mse_loss(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
                steps += 1
        regret = validate(net, datasets)
        history.append({"epoch": epoch+1, "validation_log_regret": regret})
        if regret < best:
            best, best_epoch, best_state = regret, epoch+1, copy.deepcopy(net.state_dict())
        if config.get("training_heartbeat", False):
            atomic_json(out/"training_progress.json", {"loss": loss_name, "model_seed": seed,
                      "epoch": epoch+1, "epochs": config["epochs"], "validation_log_regret": regret,
                      "updated_unix": time.time()})
        if epoch == 0 or (epoch+1) % 20 == 0:
            print(f"train {loss_name} seed={seed} epoch={epoch+1} validation={regret:.5f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    torch.save(best_state, out/f"{loss_name}_{seed}.pt")
    save_csv(out/f"{loss_name}_{seed}_training.csv", history)
    return net, {"loss": loss_name, "model_seed": seed, "initial_weights_sha256": initial,
                 "optimizer_steps": steps, "parameters": sum(p.numel() for p in net.parameters()),
                 "validation_log_regret": best, "selected_epoch": best_epoch, "seconds": time.perf_counter()-tick}


def search(task, config, method, net, model_seed, search_seed):
    problem = NetworkDesign(task, config)
    rng_init = np.random.default_rng(search_seed)
    rng = np.random.default_rng(search_seed+10000)
    rng_select = np.random.default_rng(search_seed+20000)
    archive, seen, history, checkpoints = [], set(), [], []
    oracle_seconds, screen_seconds, proposals = 0., 0., 0
    tick = time.perf_counter()

    def evaluate(graph):
        nonlocal oracle_seconds
        key = graph_key(graph)
        assert key not in seen
        start = time.perf_counter()
        value = problem.evaluate(graph)
        oracle_seconds += time.perf_counter()-start
        seen.add(key)
        archive.append((value, graph.copy()))
        history.append({"task_id": task.task_id, "method": method, "model_seed": model_seed,
                        "search_seed": search_seed, "evaluations": len(archive),
                        "best_r": max(v for v, _ in archive), "elapsed_seconds": time.perf_counter()-tick})
        if len(archive) in config.get("evaluation_checkpoints", []):
            best_value, best_graph = max(archive, key=lambda a: a[0])
            checkpoints.append({"evaluations": len(archive), "best_r": best_value,
                                "edges": np.column_stack(np.where(np.triu(best_graph, 1))).tolist()})

    evaluate(problem.base)
    while len(archive) < config["initial_evaluations"]:
        g = problem.sample(rng_init)
        if graph_key(g) not in seen:
            evaluate(g)
    initial = max(v for v, _ in archive)
    initial_hash = hashlib.sha256(b"".join(graph_key(g) for _, g in archive)).hexdigest()
    while len(archive) < config["budget"]:
        parents = [g for _, g in sorted(archive, key=lambda a: -a[0])[:config["population"]]]
        if method == "hillclimb":
            parents = parents[:1]
        n = min(config["batch_evaluations"], config["budget"]-len(archive))
        pool_size = config["proposal_pool"] if net is not None or method == "pool_random" else n
        pool = propose(problem, parents, pool_size, rng, seen, restart=method == "random", greedy=method == "hillclimb")
        proposals += len(pool)
        if method == "pool_random":
            chosen = rng_select.choice(len(pool), n, replace=False)
            pool = [pool[i] for i in chosen]
        elif net is not None:
            start = time.perf_counter()
            with torch.no_grad():
                score = net(*features(problem, pool)).numpy()
            ordered = np.argsort(score, kind="stable")
            explore = min(config["random_exploration"], n-1)
            chosen = list(ordered[:n-explore])
            if explore:
                chosen.extend(rng_select.choice(ordered[n-explore:], explore, replace=False))
            pool = [pool[i] for i in chosen]
            screen_seconds += time.perf_counter()-start
        for graph in pool:
            evaluate(graph)
    elapsed = time.perf_counter()-tick
    best, graph = max(archive, key=lambda a: a[0])
    assert len(archive) == len(seen) == config["budget"]
    row = {"task_id": task.task_id, "family": task.family, "n": task.n, "method": method,
           "model_seed": model_seed, "search_seed": search_seed, "evaluations": len(archive),
           "attack_trajectories": len(archive)*config["attack_scenarios"], "initial_r": initial,
           "base_r": archive[0][0], "best_r": best, "gain_percent": 100*(best/initial-1),
           "elapsed_seconds": elapsed, "oracle_seconds": oracle_seconds, "screen_seconds": screen_seconds,
           "proposals": proposals, "new_edges": int(np.count_nonzero(graph & ~problem.base)//2),
           "initial_graphs_sha256": initial_hash}
    design = {"task": task.to_dict(), "method": method, "model_seed": model_seed, "search_seed": search_seed,
              "edges": np.column_stack(np.where(np.triu(graph, 1))).tolist(), "best_r": best}
    if "evaluation_checkpoints" in config:
        design["checkpoints"] = checkpoints
    return row, history, design


def save_sources(out):
    repo = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py"))
    files += [repo/"experiments/gendesign_rank/model.py", repo/"experiments/gendesign_rank/experiment.py"]
    files += [repo/"experiments/gendesign_rank/truss.py", repo/"experiments/gendesign_rank/__init__.py",
              repo/"experiments/__init__.py", repo/"gamlet/__init__.py", repo/"gamlet/surrogate/__init__.py"]
    files += list((Path(__file__).parent/"tests").glob("*.py"))
    files += list((repo/"gamlet/surrogate/encoders").glob("*.py"))
    hashes = {}
    for source in files:
        relative = source.relative_to(repo)
        dest = out/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def run(config_path, out):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    assert config["budget"] >= config["initial_evaluations"] >= 2
    assert config["designs_per_task"] % config["batch_size"] == 0
    assert config["proposal_pool"] >= config["batch_evaluations"] > config["random_exploration"] >= 0
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    save_json(out/"config.json", config)
    shutil.copy2(Path(__file__).parent/"README.md", out/"PROTOCOL.md")
    (out/"requirements-lock.txt").write_text(subprocess.check_output(
        [__import__("sys").executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    tasks = make_tasks(config)
    save_json(out/"tasks.json", [t.to_dict() for t in tasks])
    save_json(out/"provenance.json", {"source_sha256": save_sources(out), "python": platform.python_version(),
              "numpy": np.__version__, "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
              "networkx": nx.__version__, "device": "cpu",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    save_json(out/"state.json", {"status": "offline"})
    datasets = prepare(tasks, config, out)
    models, training = {}, []
    save_json(out/"state.json", {"status": "training"})
    for seed in config["model_seeds"]:
        for loss in LOSSES:
            models[loss, seed], row = train(loss, seed, datasets, config, out)
            training.append(row)
            save_csv(out/"training_summary.csv", training)
    for seed in config["model_seeds"]:
        rows = [r for r in training if r["model_seed"] == seed]
        assert len({r["initial_weights_sha256"] for r in rows}) == len({r["optimizer_steps"] for r in rows}) == 1
    val = {loss: float(np.mean([r["validation_log_regret"] for r in training if r["loss"] == loss])) for loss in LOSSES[:2]}
    baseline = min(val, key=val.get)
    save_json(out/"baseline_selection.json", {"baseline": baseline, "validation": val, "test_labels_used": 0})
    results, histories, designs = [], [], []
    order_rng = np.random.default_rng(config["seed"]+99000)
    for task in [t for t in tasks if t.split == "test"]:
        for ss in config["search_seeds"]:
            variants = [(name, None, -1) for name in ("ea", "pool_random", "hillclimb", "random")]
            variants += [(name, models[name, ms], ms) for ms in config["model_seeds"] for name in LOSSES]
            for i in order_rng.permutation(len(variants)):
                name, net, ms = variants[i]
                row, history, design = search(task, config, name, net, ms, ss)
                results.append(row)
                histories.extend(history)
                designs.append(design)
            save_csv(out/"results.csv", results)
            save_csv(out/"trajectories.csv", histories)
            save_json(out/"best_designs.json", designs)
            save_json(out/"state.json", {"status": "search", "task": task.task_id, "searches_completed": len(results)})
            print(f"search {task.task_id} seed={ss}: {len(results)} searches", flush=True)
    # No fresh evaluation influences candidate or model selection.
    save_json(out/"state.json", {"status": "independent_attack_scenarios"})
    for task in [t for t in tasks if t.split == "test"]:
        pri = attack_priorities(task.n, task.attack_seed+1000000, config["fresh_scenarios"])
        lookup = {(d["method"], d["model_seed"], d["search_seed"]): d for d in designs if d["task"]["task_id"] == task.task_id}
        for row in [r for r in results if r["task_id"] == task.task_id]:
            design = lookup[row["method"], row["model_seed"], row["search_seed"]]
            adj = np.zeros((task.n, task.n), dtype=bool)
            for a, b in design["edges"]:
                adj[a, b] = adj[b, a] = True
            row["fresh_r"] = robustness(adj, pri)
    save_csv(out/"results.csv", results)
    from .audit import audit
    from .report import report
    audit(out)
    report(out)
    save_json(out/"COMPLETED.json", {"searches": len(results), "online_evaluations": sum(r["evaluations"] for r in results),
              "online_attack_trajectories": sum(r["attack_trajectories"] for r in results),
              "fresh_final_attack_trajectories": len(results)*config["fresh_scenarios"]})
    save_json(out/"state.json", {"status": "completed", "searches_completed": len(results)})
    print(f"Completed: {out.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args.config, args.out)
    except Exception as exc:
        if not existed and args.out.exists() and not (args.out/"COMPLETED.json").exists():
            save_json(args.out/"FAILED.json", {"error": repr(exc)})
        raise
