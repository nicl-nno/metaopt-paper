"""Prespecified search ablations and separate, fully evaluated common pools."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .audit import independent_compliance
from .experiment import (LOSSES, model, propose, save_csv, save_json, search,
                         stack_features, structure, train_one)
from .model import tensors
from .truss import Task, make_tasks


def load_models(source, config, seeds):
    models = {}
    for seed in seeds:
        for loss in LOSSES:
            net = model(config)
            net.load_state_dict(torch.load(source/f"{loss}_{seed}.pt", weights_only=True))
            models[loss, seed] = net.eval()
        torch.manual_seed(seed)
        models["untrained", seed] = model(config).eval()
    return models


def shuffle_control(source, config, seeds, out):
    """Permute TRAIN labels within tasks; validation stays unchanged, as in main."""
    tasks = [Task(**v) for v in json.loads((source/"tasks.json").read_text())]
    datasets = {}
    with np.load(source/"offline_data.npz", allow_pickle=False) as data:
        for task in tasks:
            if task.split == "test":
                continue
            genes = data[task.task_id+"_genes"]
            datasets[task.task_id] = {"features": tensors(stack_features(structure(task, config), genes)),
                                     "values": data[task.task_id+"_compliance"].copy(), "split": task.split}
    controls, metadata = {}, []
    for seed in seeds:
        rng = np.random.default_rng(seed+88100)
        shuffled = {k: dict(v, values=rng.permutation(v["values"]) if v["split"] == "train" else v["values"])
                    for k, v in datasets.items()}
        dest = out/f"shuffled_{seed}"
        dest.mkdir()
        controls["shuffled", seed], meta = train_one("ranknet", seed, shuffled, config, dest)
        meta["training_label_permutation_seed"] = seed+88100
        metadata.append(meta)
    save_json(out/"shuffled_training.json", metadata)
    return controls


def selection_metrics(scores, values, k=3):
    i, j = np.triu_indices(len(values), 1)
    truth = np.sign(values[i]-values[j])
    pred = np.sign(scores[i]-scores[j])
    valid = truth != 0
    accuracy = np.mean((pred[valid] == truth[valid]) + 0.5*(pred[valid] == 0))
    selected = np.argsort(scores, kind="stable")[:k]
    top = set(np.argsort(values)[:k])
    return {"pair_accuracy": float(accuracy),
            "precision_at_3": len(set(selected) & top)/k,
            "selected_mean_vs_random_percent": float(100*(np.mean(values[selected])/values.mean()-1)),
            "selected_best_log_regret": float(np.log(values[selected].min()/values.min())),
            "oracle_mean_vs_random_percent": float(100*(np.sort(values)[:k].mean()/values.mean()-1))}


def report(out):
    from .report import table
    rows = table(out/"search_results.csv")
    lookup = {(r["regime"], r["task_id"], r["search_seed"], r["method"], r["model_seed"], r["pool"], r["exploration"]): r for r in rows}
    contrasts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] != "ranknet" or int(r["exploration"]) != 1:
            continue
        candidates = [("mse", r["model_seed"], r["pool"]), ("log_mse", r["model_seed"], r["pool"]),
                      ("pool_random", "-1", r["pool"]), ("ea", "-1", "4")]
        if int(r["pool"]) == 48:
            candidates += [(m, r["model_seed"], "48") for m in ("untrained", "shuffled", "reverse")]
        for baseline, ms, pool in candidates:
            other = lookup.get((r["regime"], r["task_id"], r["search_seed"], baseline, ms, pool, "1"))
            if other:
                name = f"{r['regime']}/pool{r['pool']}/ranknet_vs_{baseline}"
                contrasts[name][r["task_id"]].append(np.log(float(r["best_compliance"])/float(other["best_compliance"])))
    effects = {name: {"delta_percent": float(100*np.expm1(np.mean([np.mean(v) for v in tasks.values()]))),
                       "task_count": len(tasks), "task_log_ratios": {t: float(np.mean(v)) for t, v in tasks.items()}}
               for name, tasks in contrasts.items()}
    additional = defaultdict(lambda: defaultdict(list))
    for r in rows:
        comparisons = []
        if r["method"] == "online_ranknet":
            comparisons = [("online_mse", "1"), ("online_log_mse", "1"), ("ranknet", "1")]
        elif r["method"] == "ranknet" and r["exploration"] == "0":
            comparisons = [("ranknet", "1")]
        for baseline, explore in comparisons:
            other = lookup[r["regime"], r["task_id"], r["search_seed"], baseline, r["model_seed"], "48", explore]
            name = f"{r['regime']}/pool48/{r['method']}_explore{r['exploration']}_vs_{baseline}_explore{explore}"
            additional[name][r["task_id"]].append(np.log(float(r["best_compliance"])/float(other["best_compliance"])))
    effects.update({name: {"delta_percent": float(100*np.expm1(np.mean([np.mean(v) for v in tasks.values()]))),
                           "task_count": len(tasks), "task_log_ratios": {t: float(np.mean(v)) for t, v in tasks.items()}}
                   for name, tasks in additional.items()})
    pool_rows = table(out/"common_pool_metrics.csv") if (out/"common_pool_metrics.csv").exists() else []
    pool_groups = defaultdict(list)
    for row in pool_rows:
        pool_groups[row["stage"], row["method"]].append(row)
    pool_means = {"/".join(key): {field: float(np.mean([float(r[field]) for r in group]))
                                 for field in ("pair_accuracy", "precision_at_3", "selected_mean_vs_random_percent", "selected_best_log_regret")}
                  for key, group in pool_groups.items()}
    timing = []
    for r in rows:
        if r["method"] == "ea_time":
            ref = lookup[r["regime"], r["task_id"], r["search_seed"], "ranknet", r["time_reference_model_seed"], "48", "1"]
            timing.append({"regime": r["regime"], "task": r["task_id"], "search_seed": r["search_seed"],
                           "ranknet_vs_ea_time_delta_percent": 100*(float(ref["best_compliance"])/float(r["best_compliance"])-1),
                           "ea_evaluations": int(r["evaluations"]), "ranknet_evaluations": int(ref["evaluations"]),
                           "ea_seconds": float(r["elapsed_seconds"]), "ranknet_seconds": float(ref["elapsed_seconds"])})
    save_json(out/"mechanism_summary.json", {"effects": effects, "common_pools": pool_means, "equal_online_time": timing,
              "interpretation": "Exploratory mechanism checks. Negative deltas favor RankNet; no inference from seed counts alone."})
    lines = ["# Mechanism checks", "", "Negative compliance differences favor RankNet. All equal-budget comparisons use real FEM values.", "",
             "| Comparison | Difference | Tasks |", "|---|---:|---:|"]
    lines += [f"| {name} | {v['delta_percent']:+.2f}% | {v['task_count']} |" for name, v in effects.items()]
    lines += ["", "## Separate common-pool diagnostics", "", "These FEM calls occur after searches; their labels cannot influence search, model choice or checkpoints.", "",
              "| Stage / model | Pair accuracy | Precision@3 | Selected mean vs random |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {v['pair_accuracy']:.3f} | {v['precision_at_3']:.3f} | {v['selected_mean_vs_random_percent']:+.2f}% |" for name, v in pool_means.items()]
    if timing:
        lines += ["", f"Equal online time: mean RankNet/EA compliance difference {np.mean([v['ranknet_vs_ea_time_delta_percent'] for v in timing]):+.2f}%. "
                  "Descriptive timing comparison; offline training is excluded and EA may overshoot by one evaluation batch."]
    lines += ["", "Read effects jointly: a gain over random selection from the same pool supports useful screening; "
              "a gain over untrained/shuffled controls supports learning; pool-size sensitivity tests selection opportunity. "
              "A rank/regression difference on identical pools separates ordering quality from search-trajectory divergence. "
              "Larger-grid transfer remains within the rectangular truss family. This is not a proof of universal superiority."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(source, out, quick=False):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((source/"config.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    seeds = config["model_seeds"][:1 if quick else 3]
    search_seeds = [901] if quick else [901, 902, 903, 904, 905]
    tasks = make_tasks("test", 1 if quick else 8, config["seed"]+4001)
    save_json(out/"tasks.json", [t.to_dict() for t in tasks])
    save_json(out/"protocol.json", {"source": str(source.resolve()), "source_config": config,
              "model_seeds": seeds, "search_seeds": search_seeds, "quick": quick,
              "pools": [4, 12, 48, 192], "budget": 24 if quick else 256,
              "source_checkpoints": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.pt")}})
    models = load_models(source, config, seeds)
    models.update(shuffle_control(source, config, seeds, out))
    results, checks, designs, endpoints = [], [], [], []
    total_queries = 0
    for regime, nx, ny in [("in_distribution", config["nx"], config["ny"]), ("larger_grid", 6, 4)]:
        cfg = dict(config, nx=nx, ny=ny, budget=24 if quick else 256)
        for task in tasks:
            for ss in search_seeds:
                variants = [("ea", None, -1, 4, 1, "model")]
                pools = [4, 12, 48, 192] if regime == "in_distribution" else [48]
                for pool in pools:
                    variants.append(("pool_random", None, -1, pool, 1, "random"))
                    variants += [(loss, models[loss, ms], ms, pool, 1, "model") for ms in seeds for loss in LOSSES]
                variants += [(name, models["ranknet" if name == "reverse" else name, ms], ms, 48, 1,
                              "reverse" if name == "reverse" else "model") for ms in seeds for name in ("untrained", "shuffled", "reverse")]
                variants += [("ranknet", models["ranknet", ms], ms, 48, 0, "model") for ms in seeds]
                variants += [("online_"+loss, models[loss, ms], ms, 48, 1, "model") for ms in seeds for loss in LOSSES]
                for method, net, ms, pool, exploration, selection in variants:
                    local = dict(cfg, proposal_pool=pool, random_exploration=exploration, pool_selection=selection)
                    if method.startswith("online_"):
                        local["online_loss"] = method.removeprefix("online_")
                    row, history, design = search(task, method, net, ms, ss, local)
                    extra = {"regime": regime, "pool": pool, "exploration": exploration,
                             "time_reference_model_seed": "", "nx": nx, "ny": ny}
                    row.update(extra)
                    results.append(row)
                    design.update(extra)
                    designs.append(design)
                    total_queries += row["evaluations"]
                    for h in history:
                        if h["evaluations"] in (24, 32, 96, 256):
                            endpoints.append(dict(h, **extra))
                    actual = independent_compliance(structure(task, local), design["genes"])
                    assert np.isclose(actual, row["best_compliance"], rtol=1e-9)
                    checks.append(abs(actual/row["best_compliance"]-1))
                ref = next(r for r in reversed(results) if r["method"] == "ranknet" and r["model_seed"] == seeds[0] and r["pool"] == 48 and r["exploration"] == 1)
                timed = dict(cfg, budget=131072, time_limit_seconds=ref["elapsed_seconds"])
                row, _, design = search(task, "ea_time", None, -1, ss, timed)
                extra = {"regime": regime, "pool": 4, "exploration": 1, "time_reference_model_seed": seeds[0], "nx": nx, "ny": ny}
                row.update(extra)
                results.append(row)
                design.update(extra)
                designs.append(design)
                total_queries += row["evaluations"]
                actual = independent_compliance(structure(task, timed), design["genes"])
                assert np.isclose(actual, row["best_compliance"], rtol=1e-9)
                checks.append(abs(actual/row["best_compliance"]-1))
            save_csv(out/"search_results.csv", results)
            save_csv(out/"budget_endpoints.csv", endpoints)
            save_json(out/"best_designs.json", designs)
            report(out)
            print(f"mechanism {regime} {task.task_id}: {len(results)} searches", flush=True)
    # Common pools are generated and evaluated only AFTER every online search.
    metrics, pools_saved = [], {}
    diagnostic_queries = 0
    for task in tasks:
        truss = structure(task, config)
        for stage in ("initial", "near_ea_solution"):
            rng = np.random.default_rng(config["seed"]+5001+int(task.task_id.split("_")[-1]))
            if stage == "initial":
                parents = [truss.sample(rng) for _ in range(16)]
            else:
                best = next(d for d in designs if d["regime"] == "in_distribution" and d["task"]["task_id"] == task.task_id and d["method"] == "ea")
                parents = [np.array(best["genes"], dtype=np.int8)]
            for pool_id in range(1 if quick else 4):
                genes = propose(truss, parents, 48 if quick else 192, rng, set())
                values = np.array([truss.solve(g)["compliance"] for g in genes])
                diagnostic_queries += len(genes)
                features = tensors(stack_features(truss, genes))
                prefix = f"{task.task_id}_{stage}_{pool_id}"
                pools_saved[prefix+"_genes"] = np.stack(genes)
                pools_saved[prefix+"_compliance"] = values
                for (name, ms), net in models.items():
                    with torch.no_grad():
                        scores = net(*features).numpy()
                    metrics.append(dict(task_id=task.task_id, stage=stage, pool_id=pool_id, method=name, model_seed=ms,
                                        **selection_metrics(scores, values)))
                    pools_saved[prefix+f"_{name}_{ms}_scores"] = scores
    save_csv(out/"common_pool_metrics.csv", metrics)
    np.savez_compressed(out/"common_pools.npz", **pools_saved)
    save_json(out/"verification.json", {"passed": True, "searches_verified": len(results),
              "max_relative_compliance_discrepancy": max(checks), "search_oracle_evaluations": total_queries,
              "separate_diagnostic_oracle_evaluations": diagnostic_queries, "independent_final_checks": len(checks),
              "shuffled_control_new_oracle_evaluations": 0})
    report(out)
    save_json(out/"COMPLETED.json", {"searches": len(results), "oracle_evaluations": total_queries,
                                    "diagnostic_oracle_evaluations": diagnostic_queries})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.source, args.out, args.quick)
