"""Additional regression normalization check, selected on validation only.

Added after observing early validation checkpoint selection; original pilot is
preserved. This is an exploratory extension, not an independent confirmation.
"""
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent/"GAMLET"
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.nn import functional as F
from experiments.network_robustness.experiment import features, model, validate, search, save_csv, save_json
from experiments.network_robustness.problem import Task, NetworkDesign, attack_priorities, robustness
from experiments.network_robustness.audit import table, reference_robustness
from experiments.network_robustness.report import effect


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((source/"config.json").read_text())
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    tasks = [Task(**t) for t in json.loads((source/"tasks.json").read_text())]
    save_json(out/"protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "purpose": __doc__, "config": cfg, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "selection": "Best mean validation regret across original and globally normalized regressions; no test outcomes used for selection",
              "timing": "May overlap original pilot; do not use extension timings as controlled wall-time comparisons"})
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    datasets = {}
    for task in tasks:
        if task.split == "test":
            continue
        with np.load(source/"offline"/(task.task_id+".npz")) as data:
            datasets[task.task_id] = {"features": features(NetworkDesign(task, cfg), data["adjacency"]),
                                     "values": data["robustness"].copy(), "split": task.split}
    training = [d for d in datasets.values() if d["split"] == "train"]
    models, records = {}, []
    old = table(source/"training_summary.csv")
    for seed in cfg["model_seeds"]:
        for method in ("global_mse", "global_log_mse"):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = model(cfg)
            initial = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
            assert initial == next(r["initial_weights_sha256"] for r in old if r["loss"] == "mse" and int(r["model_seed"]) == seed)
            optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
            raw = [np.log(1-d["values"]) if method == "global_log_mse" else 1-d["values"] for d in training]
            joined = np.concatenate(raw)
            mu, sigma = joined.mean(), max(joined.std(), 1e-8)
            targets = [torch.tensor((v-mu)/sigma, dtype=torch.float32) for v in raw]
            start = time.perf_counter()
            best, best_state, best_epoch, history = float("inf"), None, 0, []
            steps = 0
            for epoch in range(cfg["epochs"]):
                net.train()
                for i in rng.permutation(len(training)):
                    order = rng.permutation(len(targets[i]))
                    for offset in range(0, len(order), cfg["batch_size"]):
                        idx = order[offset:offset+cfg["batch_size"]]
                        pred = net(*(v[idx] for v in training[i]["features"]))
                        loss = F.mse_loss(pred, targets[i][idx])
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                        optimizer.step()
                        steps += 1
                regret = validate(net, datasets)
                history.append({"epoch": epoch+1, "validation_log_regret": regret})
                if regret < best:
                    best, best_epoch, best_state = regret, epoch+1, copy.deepcopy(net.state_dict())
                if epoch == 0 or (epoch+1) % 20 == 0:
                    print(method, seed, epoch+1, regret, flush=True)
            net.load_state_dict(best_state)
            models[method, seed] = net.eval()
            torch.save(best_state, out/f"{method}_{seed}.pt")
            save_csv(out/f"{method}_{seed}_training.csv", history)
            records.append({"loss": method, "model_seed": seed, "initial_weights_sha256": initial,
                            "optimizer_steps": steps, "validation_log_regret": best, "selected_epoch": best_epoch,
                            "global_training_mean": mu, "global_training_std": sigma, "training_seconds": time.perf_counter()-start})
            save_csv(out/"training_summary.csv", records)
    val = {m: float(np.mean([float(r["validation_log_regret"]) for r in old+records if r["loss"] == m]))
           for m in ("mse", "log_mse", "global_mse", "global_log_mse")}
    selected = min(val, key=val.get)
    save_json(out/"baseline_selection.json", {"baseline": selected, "validation": val, "test_labels_used": 0})
    print("Validation selection:", selected, val, flush=True)
    if selected not in ("global_mse", "global_log_mse"):
        save_json(out/"COMPLETED.json", {"selected": selected, "additional_test_searches": 0,
                  "note": "Original regression remains the validation-selected comparator; no extra test search needed."})
        return
    rows, histories, designs = [], [], []
    for task in tasks:
        if task.split != "test":
            continue
        for ss in cfg["search_seeds"]:
            for ms in cfg["model_seeds"]:
                row, history, design = search(task, cfg, selected, models[selected, ms], ms, ss)
                rows.append(row)
                histories.extend(history)
                designs.append(design)
        save_csv(out/"results.csv", rows)
        save_csv(out/"trajectories.csv", histories)
        save_json(out/"best_designs.json", designs)
        print("Extension searches", task.task_id, len(rows), flush=True)
    for row, design in zip(rows, designs):
        task = Task(**design["task"])
        problem = NetworkDesign(task, cfg)
        adj = np.zeros_like(problem.base)
        for a, b in design["edges"]:
            adj[a, b] = adj[b, a] = True
        ref, _ = reference_robustness(adj, problem.priorities)
        assert ref == row["best_r"] and problem.valid(adj)
        pri = attack_priorities(task.n, task.attack_seed+1000000, cfg["fresh_scenarios"])
        row["fresh_r"] = robustness(adj, pri)
    save_csv(out/"results.csv", rows)
    save_json(out/"verification.json", {"passed": True, "independent_final_network_checks": len(rows)})
    # Original run may still be completing its independent-scenario evaluation.
    wait_start = time.monotonic()
    while not (source/"COMPLETED.json").exists():
        if (source/"FAILED.json").exists() or time.monotonic()-wait_start > 3600:
            raise RuntimeError("Original pilot did not complete successfully")
        time.sleep(10)
    original_rows = table(source/"results.csv")
    initial = {(r["task_id"], r["search_seed"]): r["initial_graphs_sha256"] for r in original_rows}
    for row in rows:
        assert row["initial_graphs_sha256"] == initial[row["task_id"], str(row["search_seed"])]
    combined = [r for r in original_rows if r["method"] == "ranknet"]
    combined += [dict(r, method="mse", model_seed=str(r["model_seed"]), search_seed=str(r["search_seed"])) for r in rows]
    effects = {field: effect(combined, "mse", field) for field in ("best_r", "fresh_r")}
    save_json(out/"summary.json", {"baseline": selected, "effects": effects, "exploratory_extension": True})
    text = ["# Дополнительная проверка нормировки регрессии", "",
            "Добавлена после наблюдения ранних валидационных чекпойнтов; исходный пилот сохранён. "
            "Выбор нормировки использует только валидационные данные. Это расширение пилота на тех же тестовых сетях, не независимое подтверждение.", "",
            f"Выбранный по валидации бейзлайн: **{selected}**. Одинаковые начальные веса, батчи, число шагов, начальные популяции и поисковые бюджеты проверены.", "",
            f"RankNet относительно выбранной регрессии: **{effects['best_r']['gain_percent']:+.2f}%** по основному банку атак; "
            f"**{effects['fresh_r']['gain_percent']:+.2f}%** на независимых сценариях. Положительная разница означает преимущество RankNet.", "",
            "Сырые результаты: results.csv; параметры выбора: baseline_selection.json; интервалы по исходным сетям: summary.json. "
            "Время расширения могло пересекаться с первоначальным прогоном; его не следует использовать как контролируемое сравнение скорости."]
    (out/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    save_json(out/"COMPLETED.json", {"selected": selected, "additional_test_searches": len(rows)})


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
