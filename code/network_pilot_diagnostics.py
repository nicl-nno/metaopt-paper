"""Post-run diagnostics only: no checkpoint or candidate selection."""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent/"GAMLET"
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.network_robustness.experiment import model, features, save_csv
from experiments.network_robustness.problem import Task, NetworkDesign, attack_priorities, robustness


def diagnose(out):
    config = json.loads((out/"config.json").read_text())
    torch.set_num_threads(config["threads"])
    tasks = [Task(**t) for t in json.loads((out/"tasks.json").read_text())]
    destination = out/"posthoc_diagnostics"
    destination.mkdir(exist_ok=True)
    metrics = []
    for task in tasks:
        if task.split != "validation":
            continue
        problem = NetworkDesign(task, config)
        with np.load(out/"offline"/(task.task_id+".npz")) as data:
            inputs = features(problem, data["adjacency"])
            truth = data["robustness"]
        i, j = np.triu_indices(len(truth), 1)
        signs = np.sign(truth[i]-truth[j])
        valid = signs != 0
        for seed in config["model_seeds"]:
            for loss in ("mse", "log_mse", "ranknet"):
                net = model(config)
                net.load_state_dict(torch.load(out/f"{loss}_{seed}.pt", weights_only=True))
                net.eval()
                with torch.no_grad():
                    scores = net(*inputs).numpy()
                predicted = -np.sign(scores[i]-scores[j])
                accuracy = np.mean((predicted[valid] == signs[valid]) + .5*(predicted[valid] == 0))
                selected = scores.argmin()
                metrics.append({"task_id": task.task_id, "method": loss, "model_seed": seed,
                                "within_task_score_std": float(scores.std()), "robustness_std": float(truth.std()),
                                "pair_accuracy_non_ties": float(accuracy), "fraction_true_ties": float((~valid).mean()),
                                "validation_log_regret": float(np.log(truth.max()/truth[selected]))})
    save_csv(destination/"validation_ordering.csv", metrics)
    if not (out/"COMPLETED.json").exists():
        return
    designs = json.loads((out/"best_designs.json").read_text())
    first = sorted(t.task_id for t in tasks if t.split == "test")[0]
    task = next(t for t in tasks if t.task_id == first)
    problem = NetworkDesign(task, config)
    baseline = json.loads((out/"baseline_selection.json").read_text())["baseline"]
    chosen = [d for d in designs if d["task"]["task_id"] == first and d["search_seed"] == config["search_seeds"][0]
              and ((d["method"] in ("ranknet", baseline) and d["model_seed"] == config["model_seeds"][0]) or d["method"] == "ea")]
    graphs = [("Initial network", problem.base)]
    for d in chosen:
        adj = np.zeros_like(problem.base)
        for a, b in d["edges"]:
            adj[a, b] = adj[b, a] = True
        graphs.append((d["method"], adj))
    pri = attack_priorities(task.n, task.attack_seed+1000000, config["fresh_scenarios"])
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    archived = {}
    for label, adj in graphs:
        value, curves = robustness(adj, pri, True)
        archived[label] = curves
        means = np.r_[1., curves.mean(0)/task.n]
        ax.plot(np.arange(task.n+1)/task.n, means, label=f"{label}: R={value:.4f}")
    ax.set(xlabel="Fraction of deleted nodes", ylabel="Largest component / original node count",
           title="Adaptive degree attacks: independent tie scenarios")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(destination/"attack_curves.png", dpi=170)
    fig.savefig(destination/"attack_curves.pdf")
    plt.close(fig)
    np.savez_compressed(destination/"attack_curves.npz", **archived)
    (destination/"NOTE.txt").write_text("Posthoc model diagnostics and visualization. No model or candidate was selected using these results. "
                                       "Attack plot uses the fixed first task and seeds; repeats the already defined fresh-scenario bank. "
                                       "Additional plotting calls: "+str(len(graphs)*len(pri)), encoding="utf-8")
    print("Saved diagnostics:", destination)


if __name__ == "__main__":
    diagnose(Path(sys.argv[1]))
