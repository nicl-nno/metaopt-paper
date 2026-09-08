import json
import unittest
import os
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from experiments.network_robustness.problem import make_tasks, Task
from experiments.network_robustness.experiment import search, model, atomic_json
from experiments.network_robustness.campaign_report import hierarchical_effect


class CampaignTests(unittest.TestCase):
    def test_atomic_checkpoint_retries_temporary_windows_lock(self):
        real_replace = os.replace
        attempts = []
        def transient_replace(source, destination):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("temporary reader")
            return real_replace(source, destination)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.json"
            path.write_text('{"value": 0}', encoding="utf-8")
            with patch("experiments.network_robustness.experiment.os.replace", side_effect=transient_replace):
                atomic_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertEqual(len(attempts), 3)

    def test_extrapolation_is_only_in_test(self):
        cfg = {"seed": 51, "sizes": [50, 100], "tasks_per_cell": {"train": 2, "validation": 1, "test": 2},
               "extrapolation_sizes": [200], "extrapolation_tasks_per_cell": 3}
        tasks = make_tasks(cfg)
        self.assertEqual(sum(t.n == 200 for t in tasks), 6)
        self.assertTrue(all(t.split == "test" for t in tasks if t.n == 200))
        self.assertEqual(len({t.task_id for t in tasks}), len(tasks))

    def test_checkpoint_matches_shorter_budget_without_new_search(self):
        cfg = json.loads((Path(__file__).parents[1]/"smoke.json").read_text())
        cfg.update(oracle="union_find", evaluation_checkpoints=[8, 12])
        torch.set_num_threads(2)
        net = model(cfg).eval()
        task = Task("unit", "test", "er", 12, 701, 702)
        for method, predictor in [("ea", None), ("ranknet", net)]:
            full, h1, d1 = search(task, cfg, method, predictor, 17, 81)
            short, h2, d2 = search(task, dict(cfg, budget=8, evaluation_checkpoints=[8]), method, predictor, 17, 81)
            self.assertEqual(d1["checkpoints"][0]["edges"], d2["edges"])
            self.assertEqual(d1["checkpoints"][0]["best_r"], short["best_r"])
            self.assertEqual([r["best_r"] for r in h1[:8]], [r["best_r"] for r in h2])
            self.assertEqual(full["initial_graphs_sha256"], short["initial_graphs_sha256"])

    def test_hierarchical_effect_averages_within_tasks_and_blocks(self):
        rows = []
        selections = {0: "mse", 1: "global_mse"}
        for block in range(2):
            for family in ("er", "ba"):
                for task in range(3):
                    for ms in (17, 23):
                        for ss in (101, 202):
                            for method, value in [("ranknet", .22), (selections[block], .2)]:
                                rows.append({"block": block, "task_id": f"{family}_{task}", "family": family,
                                             "n": 50, "scope": "id", "budget": 96, "method": method,
                                             "model_seed": ms, "search_seed": ss, "best_r": value})
        result = hierarchical_effect(rows, selections, "selected_regression", "best_r", 96, "id", nboot=100)
        self.assertAlmostEqual(result["gain_percent"], 10)
        np.testing.assert_allclose(result["hierarchical_bootstrap_95_percent"], [10, 10])
        self.assertEqual(result["source_tasks"], 12)
        self.assertEqual(result["blocks"], 2)
        # Duplicating seed observations must not increase source-task count or effect.
        duplicate = hierarchical_effect(rows+rows, selections, "selected_regression", "best_r", 96, "id", nboot=100)
        self.assertEqual(duplicate["source_tasks"], result["source_tasks"])
        self.assertAlmostEqual(duplicate["gain_percent"], result["gain_percent"])


if __name__ == "__main__":
    unittest.main()
