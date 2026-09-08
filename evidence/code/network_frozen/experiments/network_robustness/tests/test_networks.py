import json
import unittest
from pathlib import Path
import networkx as nx
import numpy as np
import torch

from experiments.network_robustness.problem import (Task, NetworkDesign, attack_priorities, robustness,
                                                   largest_component, bit_rows, graph_key, make_tasks)
from experiments.network_robustness.audit import reference_robustness
from experiments.network_robustness.experiment import search, features, model


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1]/"smoke.json").read_text())
        self.task = Task("unit", "test", "ba", 12, 11, 13)
        self.problem = NetworkDesign(self.task, self.config)

    def test_complete_and_star_analytical(self):
        from experiments.network_robustness.fast_oracle import fast_robustness
        n = 9
        pri = attack_priorities(n, 42, 3)
        clique = nx.to_numpy_array(nx.complete_graph(n), dtype=bool)
        star = nx.to_numpy_array(nx.star_graph(n-1), dtype=bool)
        self.assertAlmostEqual(robustness(clique, pri), (n-1)/(2*n))
        self.assertAlmostEqual(robustness(star, pri), (n-1)/n**2)
        self.assertAlmostEqual(fast_robustness(clique, pri)[0], (n-1)/(2*n))
        self.assertAlmostEqual(fast_robustness(star, pri)[0], (n-1)/n**2)

    def test_union_find_oracle_same_trajectories_and_configuration(self):
        from experiments.network_robustness.fast_oracle import fast_robustness
        problem = NetworkDesign(self.task, dict(self.config, oracle="union_find"))
        rng = np.random.default_rng(129)
        for _ in range(8):
            adj = problem.sample(rng)
            r, curves = fast_robustness(adj, problem.priorities)
            expected, reference = reference_robustness(adj, problem.priorities)
            np.testing.assert_array_equal(curves, reference)
            self.assertEqual(r, expected)
            self.assertEqual(problem.evaluate(adj), expected)

    def test_adaptive_attack_matches_independent_networkx(self):
        rng = np.random.default_rng(7)
        for _ in range(6):
            adj = self.problem.sample(rng)
            actual, curves = robustness(adj, self.problem.priorities, True)
            expected, reference = reference_robustness(adj, self.problem.priorities)
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(curves, reference)
            self.assertTrue(np.all(np.diff(curves, axis=1) <= 0))

    def test_swap_preserves_degree_connectivity_edit_budget(self):
        rng = np.random.default_rng(19)
        adj = self.problem.base.copy()
        for _ in range(80):
            before = adj.copy()
            adj = self.problem.swap(adj, rng)
            self.assertEqual(np.count_nonzero(adj != before), 8)
            self.assertTrue(self.problem.valid(adj))
            self.assertTrue(nx.is_connected(nx.from_numpy_array(adj)))
            np.testing.assert_array_equal(adj.sum(1), self.problem.degree)
            self.assertLessEqual(np.count_nonzero(adj & ~self.problem.base)//2, self.problem.max_new_edges)

    def test_lcc_disconnected_and_relabeling(self):
        adj = nx.to_numpy_array(nx.disjoint_union(nx.path_graph(5), nx.complete_graph(3)), dtype=bool)
        self.assertEqual(largest_component(bit_rows(adj), (1 << 8)-1), 5)
        p = np.random.default_rng(41).permutation(len(adj))
        pri = attack_priorities(len(adj), 37, 3)
        self.assertEqual(robustness(adj, pri), robustness(adj[np.ix_(p, p)], pri[:, p]))

    def test_gnn_features_permutation_and_batching(self):
        torch.set_num_threads(2)
        net = model(self.config).eval()
        adj = self.problem.base
        f = features(self.problem, [adj, adj])
        nodes, matrix, mask, ctx = f
        p = torch.randperm(len(adj))
        torch.testing.assert_close(net(*f), net(nodes[:, p], matrix[:, p][:, :, p], mask[:, p], ctx))
        torch.testing.assert_close(net(*f)[:1], net(*(v[:1] for v in f)))

    def test_search_same_initial_budget_and_no_screening_control(self):
        config = dict(self.config, proposal_pool=4)
        net = model(config).eval()
        ea, history, de = search(self.task, config, "ea", None, -1, 53)
        ranked, _, dr = search(self.task, config, "ranknet", net, 17, 53)
        self.assertEqual(ea["evaluations"], config["budget"])
        self.assertEqual(ea["initial_graphs_sha256"], ranked["initial_graphs_sha256"])
        self.assertEqual(ea["best_r"], ranked["best_r"])
        self.assertTrue(np.all(np.diff([r["best_r"] for r in history]) >= 0))


if __name__ == "__main__":
    unittest.main()
