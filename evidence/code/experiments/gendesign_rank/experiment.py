"""Run with python -m experiments.gendesign_rank.experiment --config ... --out ..."""

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import GraphSurrogate, RepositoryGraphSurrogate, ranknet_loss, tensors
from .truss import Truss, make_tasks


LOSSES = ("mse", "log_mse", "ranknet")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, records):
    if not records:
        raise ValueError("Cannot save empty table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def structure(task, config):
    return Truss(task, **{k: config[k] for k in ("nx", "ny", "volume", "young")})


def model(config, node_dim=17):
    cls = {"custom_gnn": GraphSurrogate, "repo_gnn": RepositoryGraphSurrogate}[config.get("architecture", "custom_gnn")]
    return cls(node_dim=node_dim, hidden=config["hidden"], layers=config["layers"])


def stack_features(truss, designs):
    return tuple(np.stack(items) for items in zip(*(truss.features(g) for g in designs)))


def prepare_data(tasks, config, out):
    """No test objectives are available during training or checkpoint selection."""
    start = time.perf_counter()
    rng = np.random.default_rng(config["seed"] + 100)
    seen = set()
    records, datasets, archive = [], {}, {}
    for task in tasks:
        if task.split == "test":
            continue
        truss = structure(task, config)
        count = config["train_designs_per_task" if task.split == "train" else "validation_designs_per_task"]
        genes, values = [], []
        for _ in range(count):
            for attempt in range(100000):
                g = truss.sample(rng)
                key = truss.key(g)
                if key not in seen:
                    break
            else:
                raise RuntimeError("Cannot create globally disjoint candidate sets")
            seen.add(key)
            result = truss.solve(g)
            genes.append(g)
            values.append(result["compliance"])
        genes, values = np.stack(genes), np.array(values)
        features = tensors(stack_features(truss, genes))
        datasets[task.task_id] = {"features": features, "values": values, "split": task.split}
        archive[task.task_id + "_genes"] = genes
        archive[task.task_id + "_compliance"] = values
        records.append({"task_id": task.task_id, "split": task.split, "oracle_evaluations": count,
                        "min_compliance": float(values.min()), "mean_compliance": float(values.mean()),
                        "max_compliance": float(values.max())})
    np.savez_compressed(out / "offline_data.npz", **archive)
    save_csv(out / "offline_tasks.csv", records)
    save_json(out / "offline_cost.json", {"seconds": time.perf_counter()-start,
               "oracle_evaluations": sum(r["oracle_evaluations"] for r in records),
               "distinct_canonical_genotypes": len(seen), "test_labels_used": 0,
               "data_sha256": hashlib.sha256((out / "offline_data.npz").read_bytes()).hexdigest()})
    return datasets


@torch.no_grad()
def validation_regret(net, datasets):
    net.eval()
    regrets = []
    for data in datasets.values():
        if data["split"] != "validation":
            continue
        selected = int(net(*data["features"]).argmin())
        values = data["values"]
        # Scale independent and common to all heads; candidate pool oracle is validation only.
        regrets.append(float(np.log(values[selected] / values.min())))
    return float(np.mean(regrets))


def train_one(loss_name, seed, datasets, config, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = model(config)
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    training = [v for v in datasets.values() if v["split"] == "train"]
    history, best_state, best_epoch, best_regret = [], None, -1, float("inf")
    steps = 0
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        net.train()
        losses = []
        # Identical task order and exact candidate batches for each loss at a model seed.
        for task_index in rng.permutation(len(training)):
            data = training[task_index]
            targets = np.log(data["values"]) if loss_name == "log_mse" else data["values"]
            # Statistics use only that training task's offline observations.
            targets = (targets - targets.mean()) / max(targets.std(), 1e-8)
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            indices = rng.permutation(len(targets))
            for offset in range(0, len(indices), config["batch_size"]):
                selected = indices[offset:offset+config["batch_size"]]
                if len(selected) < 2:
                    continue
                prediction = net(*(x[selected] for x in data["features"]))
                y = target_tensor[selected]
                loss = ranknet_loss(prediction, y) if loss_name == "ranknet" else F.mse_loss(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
        regret = validation_regret(net, datasets)
        history.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "validation_log_regret": regret})
        if regret < best_regret:
            best_regret, best_state, best_epoch = regret, copy.deepcopy(net.state_dict()), epoch+1
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"train {loss_name} seed={seed} epoch={epoch+1} val_log_regret={regret:.4f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    name = f"{loss_name}_{seed}"
    torch.save(best_state, out / f"{name}.pt")
    save_csv(out / f"{name}_training.csv", history)
    metadata = {"loss": loss_name, "model_seed": seed, "selected_epoch": best_epoch,
                "validation_log_regret": best_regret, "training_seconds": time.perf_counter()-start,
                "initial_weights_sha256": initial_hash, "optimizer_steps": steps,
                "trainable_parameters": sum(p.numel() for p in net.parameters()),
                "selection": "minimum validation task-mean log regret; earliest epoch on ties"}
    return net, metadata


def mutate(truss, parents, rng):
    genes = parents[int(rng.integers(len(parents)))].copy()
    if rng.random() < 0.25:
        other = parents[int(rng.integers(len(parents)))]
        mask = rng.random(len(genes)) < 0.5
        genes[mask] = other[mask]
    for member in rng.choice(len(genes), size=int(rng.integers(1, 4)), replace=False):
        low = 1 if member < truss.mandatory else 0
        candidates = [v for v in range(low, 5) if v != genes[member]]
        genes[member] = rng.choice(candidates)
    return genes


def propose(truss, parents, count, rng, evaluated, random_search=False):
    candidates, keys = [], set()
    for _ in range(100000):
        genes = truss.sample(rng) if random_search else mutate(truss, parents, rng)
        key = truss.key(genes)
        if key not in evaluated and key not in keys:
            candidates.append(genes)
            keys.add(key)
            if len(candidates) == count:
                return candidates
    raise RuntimeError("Candidate pool exhausted")


def search(task, method, net, model_seed, search_seed, config):
    truss = structure(task, config)
    # No training data or statistics enter this function.
    rng_init = np.random.default_rng(search_seed)
    rng_propose = np.random.default_rng(search_seed+10000)
    rng_explore = np.random.default_rng(search_seed+20000)
    rng_update = np.random.default_rng(search_seed+30000)
    online_loss = config.get("online_loss")
    if online_loss:
        assert net is not None and online_loss in LOSSES
        net = copy.deepcopy(net)
        online_optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=config["weight_decay"])
    history, archive, evaluated = [], [], set()
    evaluator_seconds, screening_seconds, proposal_count = 0.0, 0.0, 0
    online_training_seconds, online_steps, rounds = 0.0, 0, 0
    start = time.perf_counter()

    def evaluate(genes):
        nonlocal evaluator_seconds
        key = truss.key(genes)
        if key in evaluated:
            raise AssertionError("Repeated true evaluation")
        tick = time.perf_counter()
        result = truss.solve(genes)
        evaluator_seconds += time.perf_counter()-tick
        evaluated.add(key)
        archive.append((result["compliance"], genes.copy()))
        history.append({"task_id": task.task_id, "method": method, "model_seed": model_seed,
                        "search_seed": search_seed, "evaluations": len(archive),
                        "best_compliance": min(a[0] for a in archive),
                        "elapsed_seconds": time.perf_counter()-start})

    # Same full-reference candidate and same random initial population for all methods.
    evaluate(np.ones(truss.n_members, dtype=np.int8))
    while len(archive) < config["initial_evaluations"]:
        candidate = truss.sample(rng_init)
        if truss.key(candidate) not in evaluated:
            evaluate(candidate)
    initial_best = min(a[0] for a in archive)
    while len(archive) < config["budget"]:
        if "time_limit_seconds" in config and time.perf_counter()-start >= config["time_limit_seconds"]:
            break
        if online_loss and rounds > 0 and rounds % 5 == 0:
            tick = time.perf_counter()
            features = tensors(stack_features(truss, [g for _, g in archive]))
            targets = np.array([v for v, _ in archive])
            if online_loss == "log_mse":
                targets = np.log(targets)
            targets = torch.tensor((targets-targets.mean()) / max(targets.std(), 1e-8), dtype=torch.float32)
            net.train()
            for _ in range(10):
                idx = rng_update.choice(len(archive), size=min(64, len(archive)), replace=False)
                pred = net(*(x[idx] for x in features))
                loss = ranknet_loss(pred, targets[idx]) if online_loss == "ranknet" else F.mse_loss(pred, targets[idx])
                online_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                online_optimizer.step()
                online_steps += 1
            net.eval()
            online_training_seconds += time.perf_counter()-tick
        parents = [g for _, g in sorted(archive, key=lambda x: x[0])[:config["population"]]]
        n = min(config["batch_evaluations"], config["budget"]-len(archive))
        pool_selection = config.get("pool_selection", "model")
        pool_size = config["proposal_pool"] if net is not None or pool_selection == "random" else n
        candidates = propose(truss, parents, pool_size, rng_propose, evaluated, method == "random")
        proposal_count += len(candidates)
        if pool_selection == "random":
            chosen = rng_explore.choice(len(candidates), size=n, replace=False)
            candidates = [candidates[i] for i in chosen]
        elif net is not None:
            tick = time.perf_counter()
            with torch.no_grad():
                prediction = net(*tensors(stack_features(truss, candidates))).numpy()
            if pool_selection == "reverse":
                prediction = -prediction
            exploration = min(config["random_exploration"], n-1)
            ordered = np.argsort(prediction, kind="stable")
            chosen = ordered[:n-exploration].tolist()
            if exploration:
                chosen.extend(rng_explore.choice(ordered[n-exploration:], size=exploration, replace=False).tolist())
            screening_seconds += time.perf_counter()-tick
            candidates = [candidates[i] for i in chosen]
        for genes in candidates:
            evaluate(genes)
        rounds += 1
    elapsed = time.perf_counter()-start
    best, best_genes = min(archive, key=lambda a: a[0])
    assert len(evaluated) == len(history)
    assert len(evaluated) == config["budget"] or "time_limit_seconds" in config
    result = {"task_id": task.task_id, "method": method, "model_seed": model_seed,
              "search_seed": search_seed, "evaluations": len(evaluated),
              "initial_best_compliance": initial_best, "best_compliance": best,
              "improvement_percent": 100*(1-best/initial_best),
              "elapsed_seconds": elapsed, "evaluator_seconds": evaluator_seconds,
              "screening_seconds": screening_seconds, "proposed_candidates": proposal_count,
              "online_training_seconds": online_training_seconds, "online_optimizer_steps": online_steps,
              "active_members": int(np.count_nonzero(best_genes)),
              "volume": float(truss.areas(best_genes) @ truss.length)}
    return result, history, {"genes": best_genes.tolist(), "task": task.to_dict(),
                             "method": method, "model_seed": model_seed, "search_seed": search_seed,
                             "compliance": best}


def validate_config(config):
    assert 2 <= config["initial_evaluations"] <= config["budget"]
    assert 0 <= config["random_exploration"] < config["batch_evaluations"]
    assert config["proposal_pool"] >= config["batch_evaluations"]
    assert min(config[k] for k in ("train_tasks", "validation_tasks", "test_tasks", "epochs")) > 0
    assert config["nx"] >= 2 and config["ny"] >= 2


def run(config_path, out):
    config_path, out = Path(config_path), Path(out)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if out.exists():
        raise FileExistsError(f"Use a new output directory to preserve the previous run: {out}")
    out.mkdir(parents=True)
    save_json(out / "config.json", config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tasks = (make_tasks("train", config["train_tasks"], config["seed"])
             + make_tasks("validation", config["validation_tasks"], config["seed"]+1)
             + make_tasks("test", config["test_tasks"], config["seed"]+2))
    save_json(out / "tasks.json", [task.to_dict() for task in tasks])
    package_dir = Path(__file__).parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package_dir.glob("*.py")}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in package_dir.glob("*.py"):
        shutil.copy2(p, snapshot / p.name)
    external = {}
    if config.get("architecture") == "repo_gnn":
        import torch_geometric
        encoder_dir = package_dir.parents[1] / "gamlet/surrogate/encoders"
        for p in encoder_dir.glob("*.py"):
            relative = str(p.relative_to(package_dir.parents[1])).replace("\\", "/")
            external[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    save_json(out / "provenance.json", {"python": platform.python_version(), "numpy": np.__version__,
              "torch": torch.__version__, "platform": platform.platform(), "device": "cpu",
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "source_sha256": sources,
              "repository_source_sha256": external,
              "architecture": config.get("architecture", "custom_gnn"),
              "torch_geometric": torch_geometric.__version__ if external else None,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True).strip()})
    print("Generating offline train/validation data (no test labels)", flush=True)
    datasets = prepare_data(tasks, config, out)
    models, training_meta = {}, []
    for seed in config["model_seeds"]:
        for loss in LOSSES:
            models[loss, seed], metadata = train_one(loss, seed, datasets, config, out)
            training_meta.append(metadata)
    save_csv(out / "training_summary.csv", training_meta)
    for seed in config["model_seeds"]:
        rows = [row for row in training_meta if row["model_seed"] == seed]
        assert len({r["initial_weights_sha256"] for r in rows}) == 1
        assert len({r["optimizer_steps"] for r in rows}) == 1
    regression_validation = {loss: float(np.mean([r["validation_log_regret"] for r in training_meta
                             if r["loss"] == loss])) for loss in ("mse", "log_mse")}
    selected_regression = min(regression_validation, key=regression_validation.get)
    # Selection is recorded before any test evaluations; never choose a winner on test.
    save_json(out / "baseline_selection.json", {"selected_regression": selected_regression,
               "validation_log_regret": regression_validation, "test_labels_used": 0})
    print(f"Validation-selected regression baseline: {selected_regression}", flush=True)
    results, histories, designs = [], [], []
    test_tasks = [task for task in tasks if task.split == "test"]
    for task in test_tasks:
        for seed in config["search_seeds"]:
            combinations = [("ea", None, -1), ("random", None, -1)]
            combinations += [(loss, models[loss, ms], ms) for ms in config["model_seeds"] for loss in LOSSES]
            for method, net, model_seed in combinations:
                result, history, design = search(task, method, net, model_seed, seed, config)
                results.append(result)
                histories.extend(history)
                designs.append(design)
        print(f"Completed {task.task_id}; {len(results)} searches", flush=True)
        save_csv(out / "results.csv", results)
        save_csv(out / "trajectories.csv", histories)
        save_json(out / "best_designs.json", designs)
    from .report import create_report
    create_report(out)
    save_json(out / "COMPLETED.json", {"searches": len(results),
                "online_oracle_evaluations": sum(r["evaluations"] for r in results),
                "test_tasks": len(test_tasks), "selected_regression": selected_regression})
    print(f"Completed. Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
