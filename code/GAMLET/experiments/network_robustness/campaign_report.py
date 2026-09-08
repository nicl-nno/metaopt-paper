"""Nested offline-block/source-task analysis, with fixed evaluation budgets."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .campaign import read, atomic


LABELS = {"ranknet": "GNN RankNet", "selected_regression": "Selected GNN regression",
          "mse": "GNN MSE", "log_mse": "GNN log-MSE", "global_mse": "GNN global MSE",
          "global_log_mse": "GNN global log-MSE", "ea": "EA", "pool_random": "Random pool selection",
          "hillclimb": "Hill climbing", "random": "Random restarts"}


def size_label(cfg, scope):
    return "/".join(map(str, cfg["sizes"] if scope == "id" else cfg["extrapolation_sizes"]))


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def hierarchical_effect(rows, selections, comparator, field, budget, scope, nboot=10000):
    subset = [r for r in rows if int(r["budget"]) == budget and r["scope"] == scope]
    lookup = {(int(r["block"]), r["task_id"], r["method"], str(r["model_seed"]), str(r["search_seed"])):
              float(r[field]) for r in subset}
    tasks = defaultdict(list)
    cells = {}
    for r in subset:
        if r["method"] != "ranknet":
            continue
        block = int(r["block"])
        baseline = selections[block] if comparator == "selected_regression" else comparator
        ms = str(r["model_seed"]) if baseline not in ("ea", "pool_random", "hillclimb", "random") else "-1"
        base = lookup[block, r["task_id"], baseline, ms, str(r["search_seed"])]
        tasks[block, r["task_id"]].append(np.log(float(r[field])/base))
        cells[block, r["task_id"]] = (r["family"], int(r["n"]))
    block_ids = sorted({b for b, _ in tasks})
    cell_ids = sorted(set(cells.values()))
    matrix = np.array([[[np.mean(tasks[b, task]) for bb, task in sorted(tasks)
                        if bb == b and cells[b, task] == cell] for cell in cell_ids] for b in block_ids])
    # Balanced cells by construction; indexing resamples tasks separately for each sampled block occurrence.
    assert matrix.ndim == 3
    rng = np.random.default_rng(20260907)
    nb, nc, nt = matrix.shape
    blocks = rng.integers(nb, size=(nboot, nb))
    task_indices = rng.integers(nt, size=(nboot, nb, nc, nt))
    boot = matrix[blocks[:, :, None, None], np.arange(nc)[None, None, :, None], task_indices].mean((1, 2, 3))
    return {"gain_percent": float(100*np.expm1(matrix.mean())),
            "hierarchical_bootstrap_95_percent": (100*np.expm1(np.quantile(boot, [.025, .975]))).tolist(),
            "blocks": nb, "source_tasks": len(tasks), "source_tasks_better": int(sum(np.mean(v) > 0 for v in tasks.values())),
            "block_gain_percent": {str(b): float(100*np.expm1(matrix[i].mean())) for i, b in enumerate(block_ids)},
            "task_log_ratios": {f"{b}/{task}": float(np.mean(v)) for (b, task), v in tasks.items()},
            "bootstrap_replicates": nboot}


def report(root):
    root = Path(root)
    spec = read(root/"campaign.json")
    cfg = spec["config"]
    rows, selections, counts, diagnostics = [], {}, [], {}
    for b in range(len(spec["block_seeds"])):
        folder = root/f"block_{b:02d}"
        assert (folder/"COMPLETED.json").exists(), "Aggregate report requires every block"
        verification = read(folder/"verification.json")
        assert verification["passed"]
        counts.append(verification)
        rows += [dict(r, block=b) for r in table(folder/"endpoint_results.csv")]
        selections[b] = read(folder/"baseline_selection.json")["baseline"]
        if (folder/"validation_diagnostics.csv").exists():
            diagnostic_rows = table(folder/"validation_diagnostics.csv")
            diagnostics[b] = {m: float(np.mean([float(r["pair_accuracy"]) for r in diagnostic_rows
                               if r["method"] == m and r["pair_accuracy"] != ""])) for m in cfg["losses"]}
    comparators = ["selected_regression"] + [m for m in LABELS if m not in ("selected_regression", "ranknet")]
    effects = {scope: {str(budget): {field: {m: hierarchical_effect(rows, selections, m, field, budget, scope)
                             for m in comparators} for field in ("best_r", "fresh_r")}
                      for budget in cfg["evaluation_checkpoints"]}
               for scope in sorted({r["scope"] for r in rows})}
    summary = {"primary": {"scope": "id", "budget": cfg["primary_budget"], "field": "best_r",
                           "comparator": "selected_regression"},
               "regression_selection_by_block": selections, "effects": effects,
               "counts": {k: sum(c[k] for c in counts) for k in (
                   "verified_searches", "verified_endpoints", "offline_evaluations", "online_evaluations",
                   "fresh_attack_trajectories", "reference_attack_trajectories", "audit_production_attack_trajectories")},
               "offline_blocks": len(selections), "all_blocks_audited": True,
               "inference": f"Hierarchical bootstrap of offline blocks and source tasks within family/size strata; {len(selections)} top-level blocks limit precision.",
               "validation_pair_accuracy_by_block": diagnostics,
               "wall_time_is_an_endpoint": False}
    atomic(root/"summary.json", summary)
    primary = effects["id"][str(cfg["primary_budget"])]["best_r"]["selected_regression"]
    lo, hi = primary["hierarchical_bootstrap_95_percent"]
    lines = ["# Устойчивость сетей: независимые повторы с фиксированным бюджетом", "",
             f"Завершены и проверены все {len(selections)} независимых блоков. Основной бюджет: {cfg['primary_budget']} точных оценок, включая начальную популяцию.", "",
             f"RankNet относительно GNN-регрессии, выбранной по валидации: **{primary['gain_percent']:+.2f}%** по устойчивости R; иерархический bootstrap-интервал **[{lo:+.2f}%; {hi:+.2f}%]**.",
             f"Среднее преимущество по seed есть на {primary['source_tasks_better']} из {primary['source_tasks']} исходных сетей основного теста. Это относительные проценты, не процентные пункты.", "",
             "## Основное сравнение по бюджетам и перенос на другие размеры", "",
             "| Набор | Бюджет | Основные сценарии | Новые сценарии |", "|---|---:|---:|---:|"]
    for scope in effects:
        for budget in cfg["evaluation_checkpoints"]:
            values = effects[scope][str(budget)]
            name = f"{size_label(cfg, scope)} вершин" if scope == "id" else f"Перенос на {size_label(cfg, scope)} вершин"
            lines.append(f"| {name} | {budget} | {values['best_r']['selected_regression']['gain_percent']:+.2f}% | {values['fresh_r']['selected_regression']['gain_percent']:+.2f}% |")
    lines += ["", "## Все сравнения на основном наборе при основном бюджете", "",
              "| С чем сравниваем RankNet | Прирост R | 95% интервал |", "|---|---:|---:|"]
    for m in comparators:
        v = effects["id"][str(cfg["primary_budget"])]["best_r"][m]
        a, z = v["hierarchical_bootstrap_95_percent"]
        lines.append(f"| {LABELS[m]} | {v['gain_percent']:+.2f}% | [{a:+.2f}%; {z:+.2f}%] |")
    lines += ["", "## Результаты отдельных независимых блоков", "",
              "| Блок | Регрессия, выбранная по валидации | Прирост R |", "|---|---|---:|"]
    for b, selected in selections.items():
        lines.append(f"| {b:02d} | {selected} | {primary['block_gain_percent'][str(b)]:+.2f}% |")
    lines += ["", "Каждый блок использует новую обучающую и валидационную выборку и новые тестовые исходные сети. "
              "Пилот в анализ не включён. Повторы с разными seed сначала усредняются внутри исходной сети. "
              "Интервалы повторно выбирают целые блоки и сети внутри комбинаций семейства/размера. "
              f"Число независимых блоков ({len(selections)}) ограничивает точность оценки вариативности обучения; интервалы вторичных сравнений описательные и не скорректированы за множественные проверки.", "",
              "## Бюджет и проверка", "",
              f"Поисков: {summary['counts']['verified_searches']}; точных онлайн-оценок: {summary['counts']['online_evaluations']}; "
              f"отдельных обучающих/валидационных оценок: {summary['counts']['offline_evaluations']}. "
              "Одна оценка усредняет четыре фиксированных сценария. Сравнение при меньшем бюджете использует префикс того же поиска.", "",
              f"Проверены {summary['counts']['verified_endpoints']} графов в бюджетных точках. "
              "Независимая реализация NetworkX воспроизвела все основные траектории атаки без расхождений. "
              "Проверены степени, связность, ограничения на новые рёбра, равенство начальных популяций и числа оценок. "
              "32 новых сценария на каждом найденном решении не влияли на поиск или выбор модели.", "",
              "Время не является критерием эксперимента. Предварительные обучающие оценки и контрольные пересчёты учтены отдельно: "
              "результат при фиксированном бюджете поиска не означает превосходства по общему числу оценок с учётом создания обучающей базы.", "",
              "![Сходимость по числу точных оценок](convergence.png)", "",
              "Полные эффекты и интервалы: summary.json. Протокол: PROTOCOL.md. "
              "Данные по конечным точкам: block_*/endpoint_results.csv. Проверки: block_*/verification.json. "
              "Графы и траектории каждого поиска: block_*/runs/*.json."]
    if diagnostics:
        rank_acc = np.mean([v["ranknet"] for v in diagnostics.values()])
        reg_acc = np.mean([v[selections[b]] for b, v in diagnostics.items()])
        lines += ["", "## Диагностика порядка кандидатов", "",
                  f"Средняя точность порядка нетождественных пар на валидационных исходных графах: "
                  f"RankNet — {100*rank_acc:.2f}%, выбранная регрессия — {100*reg_acc:.2f}%. "
                  "Это диагностика выбранных моделей на валидации, а не независимая тестовая оценка или доказательство причинного механизма. "
                  "Дополнительный контроль качества отбора — случайный выбор из такого же пула кандидатов, включённый в основное сравнение."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot_convergence(root, spec, selections)


def plot_convergence(root, spec, selections):
    accumulated = defaultdict(lambda: [0., 0])
    methods = ("ranknet", "selected_regression", "ea", "hillclimb", "pool_random")
    for block in selections:
        for path in (root/f"block_{block:02d}/runs").glob("*.json"):
            obj = read(path)
            row = obj["row"]
            method = "selected_regression" if row["method"] == selections[block] else row["method"]
            if method not in methods:
                continue
            scope = "id" if row["n"] in spec["config"]["sizes"] else "extrapolation"
            for r in obj["history"]:
                if r["evaluations"] < spec["config"]["initial_evaluations"]:
                    continue
                slot = accumulated[block, scope, method, r["evaluations"]]
                slot[0] += np.log(r["best_r"]/row["initial_r"])
                slot[1] += 1
    scopes = [s for s in ("id", "extrapolation") if any(scope == s for _, scope, _, _ in accumulated)]
    fig, axes = plt.subplots(1, len(scopes), figsize=(6*len(scopes), 4.5), squeeze=False, layout="constrained")
    for ax, scope in zip(axes[0], scopes):
        for method in methods:
            xs = sorted({e for _, s, m, e in accumulated if s == scope and m == method})
            ys = [100*np.expm1(np.mean([accumulated[b, scope, method, x][0]/accumulated[b, scope, method, x][1]
                                       for b in selections])) for x in xs]
            ax.plot(xs, ys, label=LABELS[method])
        ax.axvline(spec["config"]["primary_budget"], color="gray", ls=":")
        ax.set(xlabel="Exact evaluations (including initialization)", ylabel="Geometric mean gain over shared initial best (%)",
               title=f"In-distribution: {size_label(spec['config'], scope)} nodes" if scope == "id"
                     else f"Size extrapolation: {size_label(spec['config'], scope)} nodes")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(root/"convergence.png", dpi=170)
    fig.savefig(root/"convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    report(parser.parse_args().out)
