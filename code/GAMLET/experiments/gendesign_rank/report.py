"""Descriptive, task-blocked pilot analysis and standalone scientific figures."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .truss import Task, Truss


LABELS = {"ranknet": "RankNet", "mse": "Regression (MSE)", "log_mse": "Regression (log MSE)",
          "ea": "Evolution without surrogate", "random": "Random search"}
COLORS = {"ranknet": "#196b9e", "mse": "#d47b28", "log_mse": "#9d499c",
          "ea": "#4f7952", "random": "#888888"}


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_effect(rows, baseline):
    lookup = {(r["task_id"], r["method"], r["model_seed"], r["search_seed"]): float(r["best_compliance"])
              for r in rows}
    task_logs = defaultdict(list)
    for (task, method, ms, ss), value in lookup.items():
        if method != "ranknet":
            continue
        baseline_seed = "-1" if baseline in ("ea", "random") else ms
        other = lookup[task, baseline, baseline_seed, ss]
        task_logs[task].append(float(np.log(value/other)))
    block_means = np.array([np.mean(v) for v in task_logs.values()])
    rng = np.random.default_rng(4417)
    boot = rng.choice(block_means, size=(10000, len(block_means)), replace=True).mean(axis=1)
    low, high = 100*(np.exp(np.quantile(boot, [0.025, 0.975]))-1)
    return {"baseline": baseline, "delta_percent": float(100*(np.exp(block_means.mean())-1)),
            "task_bootstrap_95_percent": [float(low), float(high)],
            "task_count": len(block_means), "paired_runs": sum(map(len, task_logs.values())),
            "tasks_ranknet_better": int(np.sum(block_means < 0)),
            "task_log_ratios": {k: float(np.mean(v)) for k, v in task_logs.items()},
            "interpretation": "negative delta favors RankNet; bootstrap across tasks only; exploratory"}


def create_report(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    rows = table(out/"results.csv")
    trajectories = table(out/"trajectories.csv")
    selection = json.loads((out/"baseline_selection.json").read_text())
    effects = {b: paired_effect(rows, b) for b in ("mse", "log_mse", "ea", "random")}
    methods = ("ranknet", "mse", "log_mse", "ea", "random")
    costs = {m: {"mean_seconds": float(np.mean([float(r["elapsed_seconds"]) for r in rows if r["method"] == m])),
                  "mean_improvement_percent": float(np.mean([float(r["improvement_percent"]) for r in rows if r["method"] == m]))}
             for m in methods}
    summary = {"primary_baseline": selection["selected_regression"], "effects": effects,
               "methods": costs, "test_tasks": config["test_tasks"], "true_evaluations_per_search": config["budget"],
               "global_optimum_known": False, "confirmatory": False}
    budget_effects = {}
    for budget in (32, 96, 256):
        if budget > config["budget"] or budget < config["initial_evaluations"]:
            continue
        prefix_rows = [r for r in trajectories if int(r["evaluations"]) == budget]
        budget_effects[budget] = {b: paired_effect(prefix_rows, b) for b in ("mse", "log_mse", "ea", "random")}
    (out/"budget_summary.json").write_text(json.dumps(budget_effects, indent=2), encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    initial = {(r["task_id"], r["method"], r["model_seed"], r["search_seed"]): float(r["initial_best_compliance"]) for r in rows}
    grouped = defaultdict(list)
    for r in trajectories:
        key = (r["task_id"], r["method"], r["model_seed"], r["search_seed"])
        if int(r["evaluations"]) >= config["initial_evaluations"]:
            grouped[r["method"], r["task_id"], int(r["evaluations"])].append(float(r["best_compliance"])/initial[key])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    for m in methods:
        xs = sorted({n for method, _, n in grouped if method == m})
        ys = [np.mean([np.mean(values) for (method, task, n), values in grouped.items() if method == m and n == x]) for x in xs]
        axes[0].plot(xs, ys, label=LABELS[m], color=COLORS[m], linewidth=2)
    axes[0].set(xlabel="True mechanical evaluations", ylabel="Best compliance / initial best (lower is better)",
                title=f"Equal evaluation budget; {config['test_tasks']} held-out tasks")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    regressions = ["mse", "log_mse"]
    for i, baseline in enumerate(regressions):
        effect = effects[baseline]
        values = 100*(np.exp(list(effect["task_log_ratios"].values()))-1)
        axes[1].scatter(values, np.full(len(values), i), color=COLORS[baseline], alpha=0.7, s=40)
        lo, hi = effect["task_bootstrap_95_percent"]
        axes[1].plot([lo, hi], [i+0.16, i+0.16], color="black", linewidth=2)
        axes[1].scatter([effect["delta_percent"]], [i+0.16], marker="D", color="black", s=36)
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set(yticks=[0, 1], yticklabels=[LABELS[m] for m in regressions],
                xlabel="RankNet compliance difference (%)\nNegative favors ranking", title="Task effects and exploratory 95% intervals")
    axes[1].set_ylim(-0.4, 1.6)
    axes[1].grid(axis="x", alpha=0.2)
    fig.savefig(out/"convergence.png", dpi=180)
    fig.savefig(out/"convergence.pdf")
    plt.close(fig)

    designs = json.loads((out/"best_designs.json").read_text())
    first_task = sorted({r["task_id"] for r in rows})[0]
    chosen = [d for d in designs if d["task"]["task_id"] == first_task
              and d["search_seed"] == config["search_seeds"][0]
              and (d["model_seed"] == config["model_seeds"][0] or d["method"] == "ea")
              and d["method"] != "random"]
    chosen.sort(key=lambda d: methods.index(d["method"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for ax, d in zip(axes.ravel(), chosen):
        truss = Truss(Task(**d["task"]), **{k: config[k] for k in ("nx", "ny", "volume", "young")})
        areas = truss.areas(d["genes"])
        for (a, b), area in zip(truss.edges, areas):
            ax.plot(truss.xy[[a, b], 0], truss.xy[[a, b], 1],
                    color=COLORS[d["method"]] if area else "#d8d8d8",
                    linewidth=0.7 + 3*area/0.003 if area else 0.6,
                    linestyle="-" if area else ":", zorder=2 if area else 1)
        ax.scatter(truss.xy[:, 0], truss.xy[:, 1], s=12, color="#333333", zorder=3)
        support_nodes = np.flatnonzero(truss.fixed[::2])
        ax.scatter(truss.xy[support_nodes, 0], truss.xy[support_nodes, 1], marker=">", color="black", s=70)
        forces = truss.force_vector.reshape(-1, 2)
        for node in np.flatnonzero(np.linalg.norm(forces, axis=1)):
            delta = forces[node]/truss.task.force*0.65
            ax.annotate("", xy=truss.xy[node]+delta, xytext=truss.xy[node],
                        arrowprops={"arrowstyle": "->", "color": "#a32929", "lw": 2})
        ax.set_title(f"{LABELS[d['method']]}\nCompliance {d['compliance']:.1f} J; {sum(areas > 0)} members")
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)", ylim=(-0.8, truss.task.height+0.2))
    fig.suptitle(f"Fixed first test task and seeds (not best-case selection); volume = {config['volume']} m³")
    fig.savefig(out/"designs.png", dpi=180)
    fig.savefig(out/"designs.pdf")
    plt.close(fig)

    primary = effects[selection["selected_regression"]]
    low, high = primary["task_bootstrap_95_percent"]
    lines = ["# Exploratory structural-design pilot", "",
             f"Primary comparator selected on validation: **{LABELS[selection['selected_regression']]}**.", "",
             f"RankNet compliance difference: **{primary['delta_percent']:+.2f}%**; "
             f"task-bootstrap interval [{low:+.2f}%, {high:+.2f}%]. Negative favors ranking.", "",
             f"Independent test tasks: {primary['task_count']}; paired model/search runs: {primary['paired_runs']}. "
             "Repeated seeds are averaged within tasks, not counted as independent tasks.", "",
             "| Method | Mean reduction from identical initial best | Mean online seconds |",
             "|---|---:|---:|"]
    for m in methods:
        lines.append(f"| {LABELS[m]} | {costs[m]['mean_improvement_percent']:.2f}% | {costs[m]['mean_seconds']:.3f} |")
    lines += ["", "| True evaluation budget | RankNet vs validation-selected regression |", "|---|---:|"]
    for budget, values in budget_effects.items():
        lines.append(f"| {budget} | {values[selection['selected_regression']]['delta_percent']:+.2f}% |")
    lines += ["", "## Scope", "",
              "This pilot compares losses in one matched architecture and one fixed-volume truss family. "
              "All search choices use predictions and already evaluated candidates only. Test labels never select checkpoints or the regression baseline. "
              "The primary endpoint is compliance after an equal number of true evaluations; the global optimum is unknown. "
              "A small linear solver can be faster than GNN screening, so evaluation savings are not wall-clock acceleration.", "",
              "Mandatory triangulation and area normalization guarantee the encoded stability/volume constraints; "
              "stress, buckling, fatigue, manufacturing and geometric nonlinearity are not modeled. "
              "Topology varies only in optional diagonals; unseen graph families and cross-domain transfer are not tested.", "",
              "Intervals are exploratory and based on few tasks. No hyperparameters were tuned on these test outcomes. "
              "Publication claims require a frozen protocol and a new independent task set after pilot-driven changes.", "",
              "![Convergence](convergence.png)", "", "![Designs](designs.png)", "",
              "Raw results: results.csv; per-evaluation trajectories: trajectories.csv; "
              "checkpoint selection: training_summary.csv; offline cost: offline_cost.json; "
              "configuration, task manifest, checkpoints and source hashes are included."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
