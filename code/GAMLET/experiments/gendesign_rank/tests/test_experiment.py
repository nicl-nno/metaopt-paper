import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.gendesign_rank.truss import Task, Truss, make_tasks, solve_axial_system
from experiments.gendesign_rank.model import GraphSurrogate, RepositoryGraphSurrogate, ranknet_loss, tensors
from experiments.gendesign_rank.experiment import search, stack_features, validate_config


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        self.task = Task("unit", "test", 4.0, 2.0, 100000, 0.1, 0.6)
        self.truss = Truss(self.task)
        self.genes = self.truss.sample(np.random.default_rng(19))

    def test_single_bar_analytical(self):
        force, length, young, area = 1700., 2.3, 210e9, 0.002
        u, compliance, residual = solve_axial_system(np.ones((1, 1)), np.array([young*area/length]), np.array([force]))
        self.assertAlmostEqual(u[0], force*length/(young*area), places=12)
        self.assertAlmostEqual(compliance, force**2*length/(young*area), places=12)
        self.assertLess(residual, 1e-12)

    def test_independent_element_assembly_and_energy(self):
        t = self.truss
        k = np.zeros((t.xy.size, t.xy.size))
        # Independent conventional 4x4 element matrices, not the production B assembly.
        for (a, b), area in zip(t.edges, t.areas(self.genes)):
            dx, dy = t.xy[b]-t.xy[a]
            length = np.hypot(dx, dy)
            c, s = dx/length, dy/length
            element = t.young*area/length*np.array([[c*c, c*s, -c*c, -c*s],
                         [c*s, s*s, -c*s, -s*s], [-c*c, -c*s, c*c, c*s], [-c*s, -s*s, c*s, s*s]])
            dofs = [2*a, 2*a+1, 2*b, 2*b+1]
            k[np.ix_(dofs, dofs)] += element
        result = t.solve(self.genes)
        expected_u = np.linalg.solve(k[np.ix_(t.free, t.free)], t.f_free)
        np.testing.assert_allclose(result["displacement"][t.free], expected_u, rtol=1e-10, atol=1e-12)
        u = result["displacement"]
        self.assertAlmostEqual(float(u @ k @ u)/result["compliance"], 1, places=10)

    def test_physical_scaling_laws(self):
        value = self.truss.solve(self.genes)["compliance"]
        stiffer = Truss(self.task, young=2*self.truss.young).solve(self.genes)["compliance"]
        heavier = Truss(self.task, volume=2*self.truss.volume).solve(self.genes)["compliance"]
        loaded = Truss(dataclasses.replace(self.task, force=2*self.task.force)).solve(self.genes)["compliance"]
        self.assertAlmostEqual(stiffer/value, 0.5, places=10)
        self.assertAlmostEqual(heavier/value, 0.5, places=10)
        self.assertAlmostEqual(loaded/value, 4, places=10)

    def test_volume_stability_and_topology(self):
        rng = np.random.default_rng(42)
        counts = set()
        for _ in range(30):
            result = self.truss.solve(self.truss.sample(rng))
            self.assertAlmostEqual(result["volume"], self.truss.volume, places=13)
            self.assertLess(result["residual"], 1e-9)
            counts.add(result["active_members"])
        self.assertGreater(len(counts), 1)
        self.assertEqual(self.truss.key(self.genes), self.truss.key(2*self.genes))


class LearningAndSearchTests(unittest.TestCase):
    def test_pool_without_screening_opportunity_and_online_isolation(self):
        config = json.loads((Path(__file__).parents[1]/"smoke_repo.json").read_text())
        config.update(proposal_pool=4, budget=32)
        task = make_tasks("test", 1, 177)[0]
        torch.manual_seed(42)
        net = RepositoryGraphSurrogate().eval()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        ea, _, de = search(task, "ea", None, -1, 19, config)
        ranked, _, dr = search(task, "ranknet", net, 42, 19, config)
        self.assertEqual(de["genes"], dr["genes"])
        self.assertEqual(ea["best_compliance"], ranked["best_compliance"])
        trained, _, _ = search(task, "online_ranknet", net, 42, 19, dict(config, online_loss="ranknet"))
        self.assertEqual(trained["online_optimizer_steps"], 10)
        for k, v in net.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_common_pool_metrics_orientation(self):
        from experiments.gendesign_rank.mechanism import selection_metrics
        truth = np.arange(1., 8.)
        good = selection_metrics(truth, truth)
        bad = selection_metrics(-truth, truth)
        self.assertEqual(good["pair_accuracy"], 1)
        self.assertEqual(good["precision_at_3"], 1)
        self.assertEqual(bad["pair_accuracy"], 0)
        self.assertEqual(bad["precision_at_3"], 0)
        self.assertLess(good["selected_mean_vs_random_percent"], 0)

    def test_actual_repository_encoder_batching_and_mask(self):
        from gamlet.surrogate.encoders.simple_graph_encoder import SimpleGNNEncoder
        truss = Truss(make_tasks("test", 1, 91)[0])
        genes = np.ones(truss.n_members, dtype=np.int8)
        genes[-1] = 0
        f = tensors(stack_features(truss, [genes, genes]))
        net = RepositoryGraphSurrogate().eval()
        self.assertIsInstance(net.encoder, SimpleGNNEncoder)
        nodes, adj, mask, context = f
        graph = net.graph_batch(nodes, adj, mask)
        self.assertEqual(len(graph.x), 2*(truss.n_members-1))
        self.assertTrue(torch.equal(graph.batch[graph.edge_index[0]], graph.batch[graph.edge_index[1]]))
        expected = net(*f)
        torch.testing.assert_close(expected[:1], net(*(x[:1] for x in f)))
        p = torch.randperm(truss.n_members)
        torch.testing.assert_close(expected, net(nodes[:, p], adj[:, p][:, :, p], mask[:, p], context))
        nodes = nodes.clone()
        nodes[:, -1] = 1e6
        torch.testing.assert_close(expected, net(nodes, adj, mask, context))
        net.train()
        net(*f).sum().backward()
        self.assertGreater(float(net.encoder.embedding.weight.grad.abs().sum()), 0)

    def test_ranknet_orientation_and_ties(self):
        target = torch.tensor([1., 2., 4.])
        self.assertLess(ranknet_loss(torch.tensor([-2., 0., 2.]), target),
                        ranknet_loss(torch.tensor([2., 0., -2.]), target))
        scores = torch.zeros(2, requires_grad=True)
        ranknet_loss(scores, torch.ones(2)).backward()
        torch.testing.assert_close(scores.grad, torch.zeros(2))

    def test_permutation_invariant_and_absent_member_mask(self):
        truss = Truss(make_tasks("test", 1, 91)[0])
        genes = np.ones(truss.n_members, dtype=np.int8)
        genes[-1] = 0
        features = tensors(stack_features(truss, [genes]))
        torch.manual_seed(7)
        net = GraphSurrogate().eval()
        expected = net(*features)
        p = torch.randperm(truss.n_members)
        nodes, adj, mask, context = features
        got = net(nodes[:, p], adj[:, p][:, :, p], mask[:, p], context)
        torch.testing.assert_close(expected, got)
        altered = nodes.clone()
        altered[:, -1] = 100000
        torch.testing.assert_close(expected, net(altered, adj, mask, context))

    def test_search_budget_same_initialization_and_determinism(self):
        config = json.loads((Path(__file__).parents[1]/"smoke.json").read_text())
        validate_config(config)
        task = make_tasks("test", 1, 7)[0]
        torch.manual_seed(4)
        net = GraphSurrogate().eval()
        base, base_history, _ = search(task, "ea", None, -1, 19, config)
        first, history, design = search(task, "ranknet", net, 4, 19, config)
        second, history2, design2 = search(task, "ranknet", net, 4, 19, config)
        self.assertEqual(first["evaluations"], config["budget"])
        self.assertEqual(first["initial_best_compliance"], base["initial_best_compliance"])
        self.assertEqual(design, design2)
        self.assertEqual([r["best_compliance"] for r in history], [r["best_compliance"] for r in history2])
        self.assertTrue(np.all(np.diff([r["best_compliance"] for r in history]) <= 0))
        self.assertEqual([r["best_compliance"] for r in history[:8]], [r["best_compliance"] for r in base_history[:8]])


if __name__ == "__main__":
    unittest.main()
