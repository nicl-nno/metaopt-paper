"""Exploratory tests on source-graph pairs, with offline-block sensitivity."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.network_robustness.audit import reference_robustness
from experiments.network_robustness.problem import Task, NetworkDesign, attack_priorities
from experiments.network_robustness.fast_oracle import fast_robustness

COMPARATORS = ("selected_regression", "mse", "log_mse", "global_mse", "global_log_mse",
               "ea", "pool_random", "hillclimb", "random")
CONTROLS = {"ea", "pool_random", "hillclimb", "random"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps, kind="stable")
    adjusted = np.empty(len(ps))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    return adjusted


def sign_test(values):
    d = np.round(np.asarray(values, float), 12)
    wins, losses = int(np.sum(d > 0)), int(np.sum(d < 0))
    result = stats.binomtest(wins, wins+losses, .5, alternative="two-sided") if wins+losses else None
    return {"wins": wins, "losses": losses, "ties": int(np.sum(d == 0)),
            "p_two_sided": float(result.pvalue) if result else 1.,
            "win_fraction_non_tied": wins/(wins+losses) if wins+losses else None}


def signed_rank_exact(values):
    # Exact distribution of signed rank sum conditional on |d|, including tied average ranks.
    d = np.round(np.asarray(values, float), 12)
    d = d[d != 0]
    if not len(d):
        return {"W": 0., "p_two_sided": 1., "rank_biserial": 0., "n_nonzero": 0}
    ranks = stats.rankdata(abs(d), method="average")
    doubled = np.rint(2*ranks).astype(int)
    total = int(doubled.sum())
    counts = [0]*(total+1)
    counts[0] = 1
    reached = 0
    for rank in doubled:
        for s in range(reached, -1, -1):
            counts[s+rank] += counts[s]
        reached += int(rank)
    pos = int(doubled[d > 0].sum())
    tail = min(sum(counts[:pos+1]), sum(counts[pos:]))
    return {"W": min(pos, total-pos)/2,
            "p_two_sided": min(1., 2*tail/(2**len(d))),
            "rank_biserial": (2*pos-total)/total, "n_nonzero": len(d),
            "calibration": "Exact independent fair signs conditional on ranks; symmetry required"}


def block_sensitivity(values):
    d = np.asarray(values, float)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    null = (signs*d).mean(1)
    p_flip = float(np.mean(abs(null) >= abs(d.mean())-1e-14))
    if len(d) >= 2 and d.std(ddof=1) > 0:
        tt = stats.ttest_1samp(d, 0)
        ci = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=stats.sem(d))
        parametric = {"t": float(tt.statistic), "df": len(d)-1, "p_two_sided": float(tt.pvalue),
                      "gain_ci95_percent": (100*np.expm1(ci)).tolist(),
                      "assumption": "Independent approximately normal offline-block mean effects; very few blocks cannot validate this"}
    else:
        parametric = None
    return {"n_blocks": len(d), "sign_test": sign_test(d), "sign_flip_p_two_sided": p_flip,
            "sign_flip_assignments": len(signs), "minimum_possible_exact_two_sided_p": min(1., 2/(2**len(d))),
            "parametric_t_sensitivity": parametric}


def collect(root, out):
    spec = read(root/"campaign.json")
    cfg = spec["config"]
    selections, tasks, objs, hashes, statuses = {}, {}, [], {}, {}
    for block in range(len(spec["block_seeds"])):
        folder = root/f"block_{block:02d}"
        manifest = [t for t in read(folder/"tasks.json") if t["split"] == "test" and t["n"] in cfg["sizes"]]
        variants = [(m, -1) for m in sorted(CONTROLS)] + [(m, ms) for m in cfg["losses"] for ms in cfg["model_seeds"]]
        expected = {f"{t['task_id']}__{m}__{ms}__{ss}.json" for t in manifest
                    for m, ms in variants for ss in cfg["search_seeds"]}
        found = {p.name for p in (folder/"runs").glob("*.json")}
        statuses[block] = {"complete_id_runs": len(expected & found), "expected_id_runs": len(expected),
                           "campaign_block_audited": (folder/"COMPLETED.json").exists()}
        if not expected <= found:
            continue
        selections[block] = read(folder/"baseline_selection.json")["baseline"]
        for t in manifest:
            tasks[block, t["task_id"]] = t
        for name in sorted(expected):
            path = folder/"runs"/name
            raw = path.read_bytes()
            hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            obj = json.loads(raw)
            row, h, design = obj["row"], obj["history"], obj["design"]
            assert row["evaluations"] == len(h) == cfg["budget"]
            assert [r["evaluations"] for r in h] == list(range(1, cfg["budget"]+1))
            assert np.all(np.diff([r["best_r"] for r in h]) >= 0)
            assert row["best_r"] == design["best_r"] == h[-1]["best_r"]
            assert [cp["evaluations"] for cp in design["checkpoints"]] == cfg["evaluation_checkpoints"]
            for cp in design["checkpoints"]:
                assert cp["best_r"] == h[cp["evaluations"]-1]["best_r"]
                assert 0 < cp["best_r"] < 1 and 0 < cp["fresh_r"] < 1
            objs.append((block, obj))
    assert selections, "No complete in-distribution offline-data block yet"
    starts = defaultdict(set)
    snapshot = []
    for b, obj in objs:
        r = obj["row"]
        starts[b, r["task_id"], r["search_seed"]].add(r["initial_graphs_sha256"])
        for cp in obj["design"]["checkpoints"]:
            snapshot.append({"block": b, "task_id": r["task_id"], "family": r["family"], "n": r["n"],
                             "method": r["method"], "model_seed": r["model_seed"], "search_seed": r["search_seed"],
                             "budget": cp["evaluations"], "best_r": cp["best_r"], "fresh_r": cp["fresh_r"]})
    assert all(len(v) == 1 for v in starts.values())
    write_csv(out/"endpoint_snapshot.csv", snapshot)
    save(out/"input_manifest.json", {"campaign": str(root.resolve()), "input_sha256": hashes,
         "included_blocks": list(selections), "selection": selections, "block_completion_at_snapshot": statuses,
         "included_source_tasks": len(tasks), "included_search_runs": len(objs),
         "snapshot_utc": datetime.now(timezone.utc).isoformat(), "scope": "50/100 nodes only",
         "complete_campaign": len(selections) == len(spec["block_seeds"])})
    return cfg, selections, tasks, objs, snapshot


def paired_values(cfg, selections, tasks, snapshot, comparator, budget, field):
    lookup = {(r["block"], r["task_id"], r["method"], r["model_seed"], r["search_seed"]): r[field]
              for r in snapshot if r["budget"] == budget}
    results = []
    for (b, task), info in sorted(tasks.items()):
        baseline = selections[b] if comparator == "selected_regression" else comparator
        pairs = [(lookup[b, task, "ranknet", ms, ss], lookup[b, task, baseline, -1 if baseline in CONTROLS else ms, ss])
                 for ms in cfg["model_seeds"] for ss in cfg["search_seeds"]]
        results.append({"block": b, "task_id": task, "family": info["family"], "n": info["n"],
                        "comparator": comparator, "budget": budget, "field": field,
                        "log_ratio": float(np.mean([math.log(a/z) for a, z in pairs])),
                        "absolute_r_difference": float(np.mean([a-z for a, z in pairs])),
                        "rank_r": float(np.mean([a for a, z in pairs])),
                        "baseline_r": float(np.mean([z for a, z in pairs])), "matched_seed_pairs": len(pairs)})
    return results


def matrix_from_rows(rows, field="log_ratio"):
    blocks = sorted({r["block"] for r in rows})
    cells = sorted({(r["family"], r["n"]) for r in rows})
    matrix = np.array([[[r[field] for r in rows if r["block"] == b and (r["family"], r["n"]) == cell]
                       for cell in cells] for b in blocks])
    assert matrix.ndim == 3
    return matrix, blocks, cells


def bootstrap_intervals(matrix, nboot=10000):
    nb, nc, nt = matrix.shape
    rng = np.random.default_rng(26090817)
    ti = rng.integers(nt, size=(nboot, nb, nc, nt))
    fixed = matrix[np.arange(nb)[None, :, None, None], np.arange(nc)[None, None, :, None], ti].mean((1, 2, 3))
    bi = rng.integers(nb, size=(nboot, nb))
    ti = rng.integers(nt, size=(nboot, nb, nc, nt))
    hierarchical = matrix[bi[:, :, None, None], np.arange(nc)[None, None, :, None], ti].mean((1, 2, 3))
    return {"fixed_models_stratified_ci95_percent": (100*np.expm1(np.quantile(fixed, [.025, .975]))).tolist(),
            "hierarchical_descriptive_ci95_percent": (100*np.expm1(np.quantile(hierarchical, [.025, .975]))).tolist()}


def analyze_comparison(rows):
    matrix, blocks, cells = matrix_from_rows(rows)
    x = matrix.ravel()
    block_values = matrix.mean((1, 2))
    per_block = {str(b): {"sign_test": sign_test(matrix[i].ravel()),
                         "signed_rank_sensitivity": signed_rank_exact(matrix[i].ravel()),
                         "gain_percent": float(100*np.expm1(block_values[i]))}
                 for i, b in enumerate(blocks)}
    strata = {f"{b}/{family}/{n}": sign_test(matrix[i, j]) for i, b in enumerate(blocks)
               for j, (family, n) in enumerate(cells)}
    return {"gain_percent": float(100*np.expm1(x.mean())), "median_task_gain_percent": float(100*np.expm1(np.median(x))),
            "mean_absolute_r_difference": float(np.mean([r["absolute_r_difference"] for r in rows])),
            "task_count": len(rows), "block_count": len(blocks),
            "rank_r_task_mean": float(np.mean([r["rank_r"] for r in rows])),
            "rank_r_task_sd": float(np.std([r["rank_r"] for r in rows], ddof=1)),
            "baseline_r_task_mean": float(np.mean([r["baseline_r"] for r in rows])),
            "baseline_r_task_sd": float(np.std([r["baseline_r"] for r in rows], ddof=1)),
            "conditional_sign_test": sign_test(x), "conditional_signed_rank_sensitivity": signed_rank_exact(x),
            "offline_block_tests": block_sensitivity(block_values), "by_block": per_block, "by_stratum_sign_test": strata,
            **bootstrap_intervals(matrix)}


def diagnostics(rows, out, helper_path):
    sys.path.insert(0, str(helper_path.parent))
    from assumption_checks import check_normality, detect_outliers
    shutil.copy2(helper_path, out/"assumption_checks.py")
    data, groups = {}, sorted({r["block"] for r in rows})
    fig, axes = plt.subplots(len(groups), 2, figsize=(10, 3.3*len(groups)), squeeze=False, layout="constrained")
    for i, block in enumerate(groups):
        group = [r for r in rows if r["block"] == block]
        x = np.array([r["log_ratio"] for r in group])
        normal = check_normality(x, name=f"offline block {block}", plot=False)
        outliers = detect_outliers(x, plot=False)
        data[block] = {"normality": normal, "iqr_outliers": outliers,
                       "skewness": float(stats.skew(x, bias=False)),
                       "median_log_ratio": float(np.median(x)), "no_outliers_removed": True}
        colors = ["#176b91" if r["family"] == "er" else "#c66a2e" for r in group]
        axes[i, 0].scatter(np.arange(len(x)), 100*np.expm1(x), c=colors)
        axes[i, 0].axhline(0, color="gray", ls="--")
        axes[i, 0].set(title=f"Offline block {block}: all source tasks", xlabel="Source task (ER blue, BA orange)", ylabel="RankNet relative gain (%)")
        stats.probplot(x, dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f"Block {block}: paired log-ratio Q-Q")
    fig.savefig(out/"diagnostics.png", dpi=170)
    fig.savefig(out/"diagnostics.pdf")
    plt.close(fig)
    save(out/"assumption_diagnostics.json", data)
    return data


def audit_primary(cfg, selections, tasks, objs, out):
    checked, problems = 0, {}
    chosen = [(b, obj) for b, obj in objs if obj["row"]["method"] in ("ranknet", selections[b])]
    total = len(chosen)*len(cfg["evaluation_checkpoints"])
    for b, obj in chosen:
        task_id = obj["row"]["task_id"]
        if (b, task_id) not in problems:
            problems[b, task_id] = NetworkDesign(Task(**tasks[b, task_id]), cfg)
        problem = problems[b, task_id]
        pri = attack_priorities(problem.task.n, problem.task.attack_seed+1000000, cfg["fresh_scenarios"])
        for cp in obj["design"]["checkpoints"]:
            adj = np.zeros((problem.task.n, problem.task.n), bool)
            for a, z in cp["edges"]: adj[a, z] = adj[z, a] = True
            assert problem.valid(adj)
            expected, curves = reference_robustness(adj, problem.priorities)
            fast, fast_curves = fast_robustness(adj, problem.priorities)
            np.testing.assert_array_equal(curves, fast_curves)
            assert expected == fast == cp["best_r"]
            fresh, _ = fast_robustness(adj, pri)
            assert fresh == cp["fresh_r"]
            ref_one, ref_curve = reference_robustness(adj, pri[:1])
            fast_one, fast_curve = fast_robustness(adj, pri[:1])
            assert ref_one == fast_one
            np.testing.assert_array_equal(ref_curve, fast_curve)
            checked += 1
            if checked % 100 == 0:
                print(f"Independent audit: {checked}/{total} primary-comparison endpoints", flush=True)
    result = {"passed": True, "primary_comparison_endpoints_verified": checked, "max_absolute_discrepancy": 0,
              "reference_attack_trajectories": checked*(cfg["attack_scenarios"]+1),
              "production_recheck_attack_trajectories": checked*(cfg["attack_scenarios"]+cfg["fresh_scenarios"]+1),
              "scope": "RankNet and validation-selected regression, both budgets, all included ID tasks; other comparisons have structural/trajectory checks only"}
    save(out/"primary_comparison_audit.json", result)
    return result


def make_report(out, cfg, selections, results, audit):
    primary = results["96/best_r/selected_regression"]
    s = primary["conditional_sign_test"]
    block = primary["offline_block_tests"]
    lo, hi = primary["fixed_models_stratified_ci95_percent"]
    complete = read(out/"input_manifest.json")["complete_campaign"]
    title = "исследовательский анализ завершённой серии" if complete else "исследовательский промежуточный анализ"
    lines = [f"# Статистические тесты: {title}", "",
             f"Данные: {primary['task_count']} исходных сетей 50/100 вершин, {len(selections)} независимых обучающих блока. "
             "Все seed сначала усреднены внутри исходной сети. Тесты добавлены после просмотра первых эффектов; это не заранее зарегистрированная проверка.", "",
             "## Основной результат: 96 реальных оценок", "",
             f"RankNet относительно выбранной по валидации GNN-регрессии: **{primary['gain_percent']:+.3f}%**; "
             f"95% bootstrap-интервал при фиксированных обученных моделях **[{lo:+.3f}%; {hi:+.3f}%]**. "
             f"Средняя абсолютная разница R: {primary['mean_absolute_r_difference']:.6f}.", "",
             f"Точный двусторонний тест знаков по исходным сетям: {s['wins']} побед, {s['losses']} поражений, {s['ties']} ничьих; "
             f"p = {s['p_two_sided']:.6g}; после Holm по 36 сравнениям p = {s['p_holm_36']:.6g}. "
             "Этот тест относится к вероятности выигрыша на новых сетях при данных обученных моделях; он не тестирует напрямую величину среднего процентного улучшения.", "",
             f"Точный тест знаков по независимым обучающим блокам: **p = {block['sign_test']['p_two_sided']:.6g}**; "
             f"перестановка знака целого блока: p = {block['sign_flip_p_two_sided']:.6g}. "
             f"Всего {block['n_blocks']} независимых блоков. Минимально возможный двусторонний exact p при таком числе блоков: "
             f"{block['minimum_possible_exact_two_sided_p']:.6g}.", "",
             "Значимость по исходным сетям условна на уже обученных моделях. Она не заменяет проверку воспроизводимости при создании новой обучающей выборки. "
             "Даже для пяти блоков минимальный двусторонний exact p равен 0,0625. Не следует переключаться на односторонний тест после просмотра результата или добавлять блоки только до достижения p < 0,05.", "",
             "## Бюджеты и независимые сценарии: выбранная регрессия", "",
             "| Бюджет | Сценарии | Прирост R | Победы/сети | Условный p знаков | Holm, 36 | p по блокам |", "|---|---|---:|---:|---:|---:|---:|"]
    for budget in cfg["evaluation_checkpoints"]:
        for field in ("best_r", "fresh_r"):
            r = results[f"{budget}/{field}/selected_regression"]
            ss = r["conditional_sign_test"]
            lines.append(f"| {budget} | {'Основные' if field == 'best_r' else '32 новых'} | {r['gain_percent']:+.3f}% | "
                         f"{ss['wins']}/{r['task_count']} | {ss['p_two_sided']:.4g} | {ss['p_holm_36']:.4g} | {r['offline_block_tests']['sign_test']['p_two_sided']:.4g} |")
    lines += ["", "## Все сравнения при 96 оценках, основные сценарии", "",
              "| Сравнение RankNet с | Прирост | Условный p знаков | Holm, 36 | Wilcoxon, чувствительность |", "|---|---:|---:|---:|---:|"]
    for c in COMPARATORS:
        r = results[f"96/best_r/{c}"]
        lines.append(f"| {c} | {r['gain_percent']:+.3f}% | {r['conditional_sign_test']['p_two_sided']:.4g} | "
                     f"{r['conditional_sign_test']['p_holm_36']:.4g} | {r['conditional_signed_rank_sensitivity']['p_two_sided']:.4g} |")
    lines += ["", "## Предпосылки и ограничения", "",
              "Тест знаков требует независимых равновероятных положительных/отрицательных знаков при нулевой гипотезе. "
              "Условная независимость исходных сетей обоснована генерацией отдельных графов; обученные модели, сценарии и повторные seed общие внутри соответствующих блоков и не создают новых наблюдений. "
              "Результаты по отдельным блокам и комбинациям семейства/размера сохранены в JSON.", "",
              "Wilcoxon здесь приведён как анализ чувствительности: его калибровка требует симметрии разностей при нулевой гипотезе. "
              "Проверка Shapiro–Wilk, Q-Q и IQR-флаги сохранены; ни одна сеть не исключалась. "
              "Нормальность не доказывает независимость и не обосновывает обобщение с двух обучающих блоков.", "",
              "Параметрический t-тест по средним блоков и его интервал сохранены как отдельный анализ чувствительности. "
              "При малом числе блоков нормальность их эффектов нельзя надёжно проверить. "
              "Иерархический bootstrap также сохранён как описательный; положительный интервал при двух блоках не отменяет ограничение точных тестов.", "",
              "Поправка Холма контролирует множественность для явно перечисленной семьи из 36 сравнений "
              "(9 компараторов × 2 бюджета × 2 банка сценариев) отдельно для условных тестов знаков и для тестов знаков по блокам. "
              "Она не исправляет повторный просмотр промежуточных данных или предшествующий выбор исследовательской постановки. "
              "По результатам этого анализа вычислительная серия не останавливается и не меняется.", "",
              f"Независимо проверены {audit['primary_comparison_endpoints_verified']} бюджетных конечных графов основного сравнения, "
              "все четыре основные траектории воспроизведены через NetworkX; расхождение равно нулю. "
              "Полный банк новых сценариев пересчитан. Это отдельный аудит основного сравнения на сетях 50/100 вершин.", "",
              "![Диагностика разностей](diagnostics.png)", "",
              "Данные: endpoint_snapshot.csv; пары после усреднения seed: task_effects.csv; все тесты: statistics.json; "
              "хеши исходных файлов: input_manifest.json; заранее записанное для этого добавления описание тестов: PLAN.md. "
              "Тесты остаются исследовательскими, поскольку первые эффекты были просмотрены до составления добавления.", "",
              "Навык statistical-analysis использован для выбора единицы анализа и диагностики; атрибуция и ссылки на методы приведены в PLAN.md."]
    diag = read(out/"assumption_diagnostics.json")
    lines += ["", "Результаты диагностики парных логарифмов отношений: " + "; ".join(
        f"блок {b}: Shapiro–Wilk p={d['normality']['p_value']:.4g}, асимметрия={d['skewness']:.3f}"
        for b, d in diag.items()) + ". Тест знаков выбран до этой проверки и не требует нормальности разностей. "
        "Wilcoxon остаётся анализом чувствительности; его маленький p не используется для утверждения симметрии или нормальности."]
    (out/"REPORT_RU.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(root, out, helper_path):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("PLAN.md"), out/"PLAN.md")
    shutil.copy2(Path(__file__), out/"analyze.py")
    shutil.copy2(Path(__file__).with_name("requirements-lock.txt"), out/"requirements-lock.txt")
    cfg, selections, tasks, objs, snapshot = collect(root, out)
    print(f"Snapshot: {len(tasks)} source networks, {len(selections)} offline blocks", flush=True)
    primary = paired_values(cfg, selections, tasks, snapshot, "selected_regression", 96, "best_r")
    diagnostics(primary, out, helper_path)
    audit = audit_primary(cfg, selections, tasks, objs, out)
    results, effects = {}, []
    for budget in cfg["evaluation_checkpoints"]:
        for field in ("best_r", "fresh_r"):
            for comparator in COMPARATORS:
                rows = paired_values(cfg, selections, tasks, snapshot, comparator, budget, field)
                effects.extend(rows)
                results[f"{budget}/{field}/{comparator}"] = analyze_comparison(rows)
    assert len(results) == 36
    for target in ("conditional", "block"):
        ps = [r["conditional_sign_test"]["p_two_sided"] if target == "conditional"
              else r["offline_block_tests"]["sign_test"]["p_two_sided"] for r in results.values()]
        for r, adjusted in zip(results.values(), holm(ps)):
            dest = r["conditional_sign_test"] if target == "conditional" else r["offline_block_tests"]["sign_test"]
            dest["p_holm_36"] = float(adjusted)
    write_csv(out/"task_effects.csv", effects)
    save(out/"statistics.json", {"exploratory": True, "comparisons": results, "selections": selections,
         "scipy": scipy.__version__, "numpy": np.__version__, "primary": "96/best_r/selected_regression",
         "rounding_for_tests": 12, "bootstrap_seed": 26090817, "bootstrap_replicates": 10000,
         "primary_comparison_audited": True})
    make_report(out, cfg, selections, results, audit)
    save(out/"COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "source_tasks": len(tasks),
         "offline_blocks": len(selections), "primary_comparison_audited": True, "report": "REPORT_RU.md"})
    print(json.dumps(results["96/best_r/selected_regression"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assumption-helper", type=Path,
                        default=Path("C:/Users/Николай/.codex/skills/statistical-analysis/scripts/assumption_checks.py"))
    args = parser.parse_args()
    main(args.campaign, args.out, args.assumption_helper)
