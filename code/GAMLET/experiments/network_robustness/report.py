"""Task-blocked effects and inspectable graph-design figures."""
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .audit import table
from .problem import Task, NetworkDesign, attack_priorities, robustness

LABELS = {"ranknet": "GNN RankNet", "mse": "GNN MSE", "log_mse": "GNN log-MSE", "ea": "EA",
          "pool_random": "Random pool selection", "hillclimb": "Hill climbing", "random": "Random restarts"}


def effect(rows, baseline, field):
    lookup = {(r["task_id"], r["method"], r["model_seed"], r["search_seed"]): float(r[field]) for r in rows}
    blocks = defaultdict(list)
    for (task, method, ms, ss), value in lookup.items():
        if method == "ranknet":
            bms = ms if baseline in ("mse", "log_mse") else "-1"
            blocks[task].append(np.log(value/lookup[task, baseline, bms, ss]))
    values = np.array([np.mean(v) for v in blocks.values()])
    rng = np.random.default_rng(8137)
    boot = rng.choice(values, size=(5000, len(values)), replace=True).mean(1)
    return {"gain_percent": float(100*np.expm1(values.mean())),
            "task_bootstrap_95_percent": (100*np.expm1(np.quantile(boot, [.025, .975]))).tolist(),
            "task_count": len(values), "tasks_better": int(np.sum(values > 0)),
            "task_log_ratios": {k: float(np.mean(v)) for k, v in blocks.items()}}


def report(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    rows = table(out/"results.csv")
    baseline = json.loads((out/"baseline_selection.json").read_text())["baseline"]
    effects = {field: {b: effect(rows, b, field) for b in LABELS if b != "ranknet"} for field in ("best_r", "fresh_r")}
    costs = {m: {"mean_online_seconds": float(np.mean([float(r["elapsed_seconds"]) for r in rows if r["method"] == m])),
                 "mean_gain_over_initial_percent": float(np.mean([float(r["gain_percent"]) for r in rows if r["method"] == m]))}
             for m in LABELS}
    summary = {"primary_baseline": baseline, "effects": effects, "costs": costs, "exploratory": True,
               "primary_budget": config["budget"], "primary_scenarios": config["attack_scenarios"],
               "fresh_scenarios": config["fresh_scenarios"], "note": "Positive gains favor RankNet. CIs conditional on one offline training set."}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trajectories = table(out/"trajectories.csv")
    starts = {(r["task_id"], r["search_seed"]): float(r["initial_r"]) for r in rows}
    grouped = defaultdict(list)
    for r in trajectories:
        if int(r["evaluations"]) >= config["initial_evaluations"]:
            grouped[r["method"], r["task_id"], int(r["evaluations"])].append(100*(float(r["best_r"])/starts[r["task_id"], r["search_seed"]]-1))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for m in LABELS:
        xs = sorted({e for method, _, e in grouped if method == m})
        ys = [np.mean([np.mean(v) for (method, _, e), v in grouped.items() if method == m and e == x]) for x in xs]
        axes[0].plot(xs, ys, label=LABELS[m])
    axes[0].set(xlabel="True evaluations (each averages attack scenarios)", ylabel="Robustness gain over shared initial best (%)", title="Equal evaluation budgets")
    axes[0].legend(fontsize=8)
    for i, field in enumerate(("best_r", "fresh_r")):
        values = effects[field][baseline]
        lo, hi = values["task_bootstrap_95_percent"]
        axes[1].plot([lo, hi], [i, i], color="black", lw=2)
        axes[1].scatter([values["gain_percent"]], [i], color="#176b91", s=50)
    axes[1].axvline(0, ls="--", color="gray")
    axes[1].set(yticks=[0, 1], yticklabels=["Search scenario bank", "Independent scenario bank"],
                xlabel=f"RankNet relative gain vs {baseline} (%)", title="Exploratory task-bootstrap intervals", ylim=(-.5, 1.5))
    fig.savefig(out/"convergence.png", dpi=170)
    fig.savefig(out/"convergence.pdf")
    plt.close(fig)
    designs = json.loads((out/"best_designs.json").read_text())
    first = sorted({r["task_id"] for r in rows})[0]
    chosen = [d for d in designs if d["task"]["task_id"] == first and d["search_seed"] == config["search_seeds"][0]
              and ((d["method"] in ("ranknet", baseline) and d["model_seed"] == config["model_seeds"][0]) or d["method"] == "ea")]
    problem = NetworkDesign(Task(**chosen[0]["task"]), config)
    pos = nx.spring_layout(nx.from_numpy_array(problem.base), seed=53)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    graphs = [("Initial network", nx.from_numpy_array(problem.base))] + [(LABELS[d["method"]], nx.Graph(d["edges"])) for d in sorted(chosen, key=lambda x: (x["method"] != "ranknet", x["method"]))]
    for ax, (label, graph) in zip(axes, graphs):
        colors = ["#ed8936" if not problem.base[a, b] else "#a8afb7" for a, b in graph.edges]
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False, node_size=12, node_color="#176b91", edge_color=colors, width=.65)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Fixed first task and seeds; orange = added edge; same vertex layout")
    fig.savefig(out/"networks.png", dpi=170)
    fig.savefig(out/"networks.pdf")
    plt.close(fig)
    primary = effects["best_r"][baseline]
    fresh = effects["fresh_r"][baseline]
    lo, hi = primary["task_bootstrap_95_percent"]
    lines = ["# Пилот: оптимизация устойчивости сетей", "",
             f"Основная GNN-регрессия выбрана по валидации: **{baseline}**. Чем выше R, тем лучше.", "",
             f"RankNet относительно неё: **{primary['gain_percent']:+.2f}%**; исследовательский интервал по исходным сетям [{lo:+.2f}%, {hi:+.2f}%].",
             f"Повторная оценка тех же найденных решений на независимых сценариях: **{fresh['gain_percent']:+.2f}%**.", "",
             "| Сравнение RankNet с | Основные сценарии | Независимые сценарии |", "|---|---:|---:|"]
    for b in LABELS:
        if b != "ranknet":
            lines.append(f"| {LABELS[b]} | {effects['best_r'][b]['gain_percent']:+.2f}% | {effects['fresh_r'][b]['gain_percent']:+.2f}% |")
    lines += ["", "| Метод | Среднее онлайн-время, с | Улучшение относительно общего начального лучшего, % |", "|---|---:|---:|"]
    lines += [f"| {LABELS[m]} | {v['mean_online_seconds']:.3f} | {v['mean_gain_over_initial_percent']:+.2f} |" for m, v in costs.items()]
    lines += ["", f"Тестовых исходных сетей: {primary['task_count']}. Бюджет каждого поиска: {config['budget']} оценок по {config['attack_scenarios']} сценариям. "
              "Это пилот на одной обучающей выборке; seed не считаются независимыми сетями. Глобальный оптимум неизвестен.", "",
              "При равных степенях порядок удаления определяется фиксированными случайными приоритетами. "
              "Независимые сценарии не выбирают модели или решения. Бюджеты истинных оценок включают начальную популяцию; "
              "обучающие, валидационные и проверочные траектории учитываются отдельно. Онлайн-время исключает предварительное обучение.", "",
              "Сохраняются степени вершин, связность и число рёбер; число новых рёбер ограничено 20% исходного числа (с округлением вверх). "
              "Оптимизируется устойчивость к адаптивному удалению узлов по степени. Другие виды отказов и реальные инфраструктурные ограничения не тестируются.", "",
              "![Convergence](convergence.png)", "", "![Networks](networks.png)", "",
              "Проверка: verification.json; исходные результаты: results.csv; конфигурация: config.json; сохранённые конструкции: best_designs.json."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
