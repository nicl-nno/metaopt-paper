"""Run a frozen, finite experiment queue unattended with a hard time limit."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def stop_child(child):
    """Windows venv launchers may create a second Python process: stop both."""
    import psutil
    try:
        parent = psutil.Process(child.pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        processes = []
    pids = [p.pid for p in processes]
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    child.wait(timeout=10)
    return pids


def write_json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def aggregate(out, state):
    """Only completed, independently audited main runs enter the synthesis."""
    runs = []
    for folder in sorted(out.glob("main_*")):
        if not (folder/"verification.json").exists() or not (folder/"COMPLETED.json").exists():
            continue
        verification = json.loads((folder/"verification.json").read_text())
        if not verification["passed"]:
            continue
        runs.append((folder.name, json.loads((folder/"summary.json").read_text())))
    effects = {}
    for baseline in ("primary", "mse", "log_mse", "ea", "random"):
        blocks = [np.array(list(s["effects"][s["primary_baseline"] if baseline == "primary" else baseline]["task_log_ratios"].values())) for _, s in runs]
        if not blocks:
            continue
        mean = np.mean([v.mean() for v in blocks])
        info = {"delta_percent": float(100*np.expm1(mean)), "training_data_replicates": len(blocks),
                "held_out_tasks": sum(len(v) for v in blocks), "replicate_delta_percent": [float(100*np.expm1(v.mean())) for v in blocks]}
        if len(blocks) >= 3:
            rng = np.random.default_rng(98012)
            boot = []
            for _ in range(4000):
                chosen = rng.integers(len(blocks), size=len(blocks))
                boot.append(np.mean([rng.choice(blocks[i], size=len(blocks[i]), replace=True).mean() for i in chosen]))
            info["exploratory_hierarchical_95_percent"] = (100*np.expm1(np.quantile(boot, [.025, .975]))).tolist()
        effects[baseline] = info
    mechanisms = {}
    for folder in sorted(out.glob("mechanism_*")):
        if (folder/"COMPLETED.json").exists():
            mechanisms[folder.name] = json.loads((folder/"mechanism_summary.json").read_text())
    save = {"state": state["status"], "main_runs": [name for name, _ in runs], "effects": effects,
            "mechanism_runs": list(mechanisms), "note": "Partial results if the queue is running/stopped. Seeds are not independent task replicates."}
    write_json(out/"SUMMARY.json", save)
    lines = ["# Ночная серия: ранжирование и регрессия GNN", "", f"Состояние: **{state['status']}**.", "",
             f"Завершено и проверено основных повторов: {len(runs)} из 5. Завершено блоков диагностики: {len(mechanisms)} из 5.", "",
             "В регрессии и ранжировании используется один SimpleGNNEncoder из GAMLET (GraphSAGE); отличаются функции потерь. "
             "Отрицательная разница податливости означает преимущество RankNet. Основной бюджет — 256 истинных оценок.", "",
             "| Сравнение RankNet с | Разница | Новые обучающие выборки | Тестовые задачи |", "|---|---:|---:|---:|"]
    for name, v in effects.items():
        lines.append(f"| {name} | {v['delta_percent']:+.2f}% | {v['training_data_replicates']} | {v['held_out_tasks']} |")
    lines += ["", "`primary`: MSE или log-MSE, выбранная только на валидации отдельно для каждого повтора. "
              "Сначала усредняются парные логарифмы отношений по seed внутри задачи, затем по задачам и обучающим выборкам. "
              "При наличии хотя бы трёх повторов JSON содержит исследовательский иерархический bootstrap по обучающим выборкам и задачам."]
    if mechanisms:
        groups = defaultdict(list)
        common = defaultdict(list)
        times = []
        for summary in mechanisms.values():
            for name, value in summary["effects"].items():
                groups[name].append(np.mean(list(value["task_log_ratios"].values())))
            for name, value in summary["common_pools"].items():
                common[name].append(value)
            times.extend(summary["equal_online_time"])
        lines += ["", "## Что объясняет результат", "", "| Контроль | Разница RankNet |", "|---|---:|"]
        for name in ("in_distribution/pool48/ranknet_vs_pool_random", "in_distribution/pool48/ranknet_vs_untrained",
                     "in_distribution/pool48/ranknet_vs_shuffled", "in_distribution/pool48/ranknet_vs_ea",
                     "in_distribution/pool4/ranknet_vs_ea", "larger_grid/pool48/ranknet_vs_mse"):
            if name in groups:
                effect = 100*np.expm1(np.mean(groups[name]))
                lines.append(f"| {name} | {effect:+.2f}% |")
        key = "in_distribution/pool48/ranknet_vs_pool_random"
        if key in groups:
            delta = 100*np.expm1(np.mean(groups[key]))
            lines += ["", ("При одинаковом размере пула обученный отбор даёт более низкую среднюю податливость, чем случайный: "
                          "это согласуется с полезным отбором кандидатов по предсказанному порядку." if delta < 0 else
                          "Преимущество над случайным отбором из такого же пула пока не получено; объяснять выигрыш качеством ранжирования преждевременно.")]
        lines += ["", "| Общий пул: стадия / модель | Точность пар | Precision@3 | Средняя податливость выбранных против случайных |", "|---|---:|---:|---:|"]
        for name, values in common.items():
            avg = {k: np.mean([v[k] for v in values]) for k in values[0]}
            lines.append(f"| {name} | {avg['pair_accuracy']:.3f} | {avg['precision_at_3']:.3f} | {avg['selected_mean_vs_random_percent']:+.2f}% |")
        if times:
            delta = np.mean([v["ranknet_vs_ea_time_delta_percent"] for v in times])
            lines += ["", f"При одинаковом онлайн-времени средняя разница RankNet относительно EA: {delta:+.2f}%. "
                      "Это описательное сравнение по измеренному времени; оно не включает предварительное обучение."]
        lines += ["", "Все диагностические оценки общих пулов выполнены отдельно после поиска и не использовались для выбора моделей. "
                  "Остальные контроли, бюджеты 32/96/256, онлайн-дообучение и перенос на сетку 6×4 сохранены в отчётах блоков."]
    lines += ["", "## Файлы", "", "- `state.json`: текущий процесс, шаг очереди, срок остановки и ошибки.",
              "- `queue.json`: зафиксированные задания; `configs/`: параметры.",
              "- `main_*/REPORT.md`: основные результаты; `mechanism_*/REPORT.md`: диагностика.",
              "- `logs/`: логи каждого задания; `code_snapshot/`: исходники запуска.",
              "", "Выводы относятся к линейным шарнирным фермам с постоянным объёмом. Ограничения напряжений, потери устойчивости и изготовления не проверяются. "
              "Отдельно проверяется экономия истинных оценок и времени; преимущество по одному критерию не означает преимущество по другому. "
              "Результаты пилотов в эту сводку не включаются."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def freeze(out):
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    root = out/"code_snapshot"
    for relative in ("experiments", "experiments/gendesign_rank", "gamlet", "gamlet/surrogate", "gamlet/surrogate/encoders"):
        target = root/relative
        target.mkdir(parents=True, exist_ok=True)
        paths = (repo/relative).glob("*.py") if relative in ("experiments/gendesign_rank", "gamlet/surrogate/encoders") else [(repo/relative)/"__init__.py"]
        for path in paths:
            shutil.copy2(path, target/path.name)
    write_json(out/"source_manifest.json", {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")})
    return root


def run(out, hours):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"logs").mkdir()
    (out/"configs").mkdir()
    root = freeze(out)
    package = Path(__file__).parent
    for name in ("NIGHT_PROTOCOL.md", "requirements-repo-lock.txt"):
        shutil.copy2(package/name, out/name)
    base = json.loads((package/"pilot_repo.json").read_text())
    base.update(train_tasks=24, validation_tasks=8, test_tasks=24, train_designs_per_task=512,
                validation_designs_per_task=512, epochs=80, model_seeds=[17, 23, 37, 53, 71],
                search_seeds=list(range(1001, 1011)), budget=256, threads=2)
    queue = []
    for replicate in range(5):
        cfg = dict(base, seed=260907+10000*replicate, name=f"overnight_main_{replicate:02d}",
                   purpose="Frozen extended experiment after repository-GNN pilot; fresh train/validation/test tasks")
        path = out/"configs"/f"main_{replicate:02d}.json"
        write_json(path, cfg)
        dest = out/f"main_{replicate:02d}"
        queue += [{"name": f"train_search_{replicate:02d}", "args": ["-m", "experiments.gendesign_rank.experiment", "--config", str(path), "--out", str(dest)]},
                  {"name": f"audit_{replicate:02d}", "args": ["-m", "experiments.gendesign_rank.audit", str(dest)]},
                  {"name": f"mechanism_{replicate:02d}", "args": ["-m", "experiments.gendesign_rank.mechanism", str(dest), str(out/f"mechanism_{replicate:02d}")]}]
    write_json(out/"queue.json", queue)
    start = time.monotonic()
    now = datetime.now(timezone.utc)
    state = {"status": "running", "pid": os.getpid(), "started_utc": now.isoformat(),
             "deadline_utc": (now+timedelta(hours=hours)).isoformat(), "maximum_hours": hours,
             "finished_jobs": [], "active_job": None, "child_pid": None, "threads": 2,
             "stop_instructions": f"Create the file {out/'STOP'} to stop the queue and current child."}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if os.name == "nt":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # scoped system-sleep inhibition, screen unaffected
    child = None
    try:
        aggregate(out, state)
        for job in queue:
            if (out/"STOP").exists() or time.monotonic()-start >= hours*3600:
                state["status"] = "stopped_by_user" if (out/"STOP").exists() else "time_limit"
                break
            state["active_job"] = job["name"]
            log_path = out/"logs"/(job["name"]+".log")
            with log_path.open("w", encoding="utf-8") as log:
                env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                child = subprocess.Popen([sys.executable, *job["args"]], cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                state["child_pid"] = child.pid
                while child.poll() is None:
                    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                    state["elapsed_hours"] = (time.monotonic()-start)/3600
                    write_json(out/"state.json", state)
                    if (out/"STOP").exists() or state["elapsed_hours"] >= hours:
                        state["status"] = "stopped_by_user" if (out/"STOP").exists() else "time_limit"
                        state["terminated_worker_pids"] = stop_child(child)
                        break
                    time.sleep(min(10, max(0.05, hours*3600-(time.monotonic()-start))))
                returncode = child.returncode
            state["child_pid"] = None
            if state["status"] != "running":
                break
            if returncode:
                state.update(status="failed", failed_job=job["name"], returncode=returncode, error_log=str(log_path))
                break
            state["finished_jobs"].append(job["name"])
            write_json(out/"state.json", state)
            aggregate(out, state)
        else:
            state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        if child is not None and child.poll() is None:
            state["terminated_worker_pids"] = stop_child(child)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        state.update(active_job=None, child_pid=None, ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/"state.json", state)
        aggregate(out, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.hours <= 24:
        parser.error("hours must be positive and at most 24")
    run(args.out, args.hours)
