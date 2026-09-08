"""Independently verify saved pilot results without changing search decisions."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .truss import Task, Truss


def independent_compliance(truss, genes):
    """Conventional 4x4 element assembly, separate from production B assembly."""
    stiffness = np.zeros((truss.xy.size, truss.xy.size))
    for (a, b), area in zip(truss.edges, truss.areas(genes)):
        dx, dy = truss.xy[b]-truss.xy[a]
        length = float(np.hypot(dx, dy))
        c, s = dx/length, dy/length
        element = truss.young*area/length*np.array([
            [c*c, c*s, -c*c, -c*s], [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s], [-c*s, -s*s, c*s, s*s]])
        indices = [2*a, 2*a+1, 2*b, 2*b+1]
        stiffness[np.ix_(indices, indices)] += element
    reduced = stiffness[np.ix_(truss.free, truss.free)]
    u = np.linalg.solve(reduced, truss.f_free)
    return float(truss.f_free @ u)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    provenance = json.loads((out/"provenance.json").read_text())
    package = Path(__file__).parent
    if (out/"source_snapshot").exists():
        package = out/"source_snapshot"
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == digest, name
    for name, digest in provenance.get("repository_source_sha256", {}).items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == digest, name
    rows = read_csv(out/"results.csv")
    expected = config["test_tasks"] * len(config["search_seeds"]) * (2+3*len(config["model_seeds"]))
    assert len(rows) == expected
    trajectories = defaultdict(list)
    key = lambda r: (r["task_id"], r["method"], str(r["model_seed"]), str(r["search_seed"]))
    for row in read_csv(out/"trajectories.csv"):
        trajectories[key(row)].append(row)
    initial = defaultdict(set)
    for row in rows:
        records = trajectories[key(row)]
        assert len(records) == config["budget"] == int(row["evaluations"])
        assert [int(r["evaluations"]) for r in records] == list(range(1, config["budget"]+1))
        assert np.all(np.diff([float(r["best_compliance"]) for r in records]) <= 0)
        assert float(records[-1]["best_compliance"]) == float(row["best_compliance"])
        initial[row["task_id"], row["search_seed"]].add(tuple(float(r["best_compliance"]) for r in records[:config["initial_evaluations"]]))
    assert all(len(values) == 1 for values in initial.values())
    designs = json.loads((out/"best_designs.json").read_text())
    assert len(designs) == len(rows)
    lookup = {key(r): r for r in rows}
    errors, volume_errors = [], []
    for design in designs:
        task = Task(**design["task"])
        assert task.split == "test"
        truss = Truss(task, **{k: config[k] for k in ("nx", "ny", "volume", "young")})
        genes = np.array(design["genes"])
        assert np.all((genes >= 0) & (genes <= 4))
        assert np.all(genes[:truss.mandatory] >= 1)
        true = independent_compliance(truss, genes)
        saved = float(lookup[task.task_id, design["method"], str(design["model_seed"]), str(design["search_seed"])]["best_compliance"])
        assert np.isclose(true, saved, rtol=1e-9, atol=1e-10)
        errors.append(abs(true-saved)/saved)
        volume_errors.append(abs(truss.areas(genes) @ truss.length-config["volume"]))
    training = read_csv(out/"training_summary.csv")
    for seed in config["model_seeds"]:
        group = [r for r in training if int(r["model_seed"]) == seed]
        assert len(group) == 3
        assert len({r["initial_weights_sha256"] for r in group}) == 1
        assert len({r["optimizer_steps"] for r in group}) == 1
    baseline = json.loads((out/"baseline_selection.json").read_text())
    validation = {m: np.mean([float(r["validation_log_regret"]) for r in training if r["loss"] == m])
                  for m in ("mse", "log_mse")}
    assert baseline["selected_regression"] == min(validation, key=validation.get)
    tasks = json.loads((out/"tasks.json").read_text())
    train_ids = {t["task_id"] for t in tasks if t["split"] == "train"}
    validation_ids = {t["task_id"] for t in tasks if t["split"] == "validation"}
    test_ids = {t["task_id"] for t in tasks if t["split"] == "test"}
    assert not train_ids & validation_ids and not train_ids & test_ids and not validation_ids & test_ids
    canonical = set()
    with np.load(out/"offline_data.npz", allow_pickle=False) as data:
        for name in data.files:
            assert not name.startswith("test_")
            if name.endswith("_genes"):
                for genes in data[name]:
                    canonical_key = Truss.key(genes)
                    assert canonical_key not in canonical
                    canonical.add(canonical_key)
    result = {"passed": True, "searches_verified": len(rows), "independent_final_mechanical_checks": len(designs),
              "max_relative_compliance_discrepancy": max(errors), "max_absolute_volume_error_m3": max(volume_errors),
              "distinct_offline_genotypes": len(canonical), "recorded_source_hashes_match": True,
              "equal_initial_populations": True, "exact_evaluation_budgets": True,
              "checkpoint_and_baseline_selection_validation_only": True,
              "note": "Post-run independent checks are not search queries and did not select candidates or models."}
    (out/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    audit(parser.parse_args().out)
