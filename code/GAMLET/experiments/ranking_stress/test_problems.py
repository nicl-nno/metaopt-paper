import unittest
import networkx as nx
import numpy as np
from .problems import CoverageDesign,CascadeDesign,cascade,visible

class OracleTests(unittest.TestCase):
    def test_cascade_shortest_path_reference(self):
        for seed in range(8):
            adj=nx.to_numpy_array(nx.gnm_random_graph(9,16,seed=seed),dtype=bool)
            for alpha in [.1,.3,.5]:
                a=cascade(adj,alpha,np.arange(9)); b=cascade(adj,alpha,np.arange(9),reference=True)
                self.assertEqual(a,b)
    def test_coverage_reference_and_constraints(self):
        for seed in range(4):
            p=CoverageDesign(seed,0,'test'); rng=np.random.default_rng(seed)
            x=p.sample(rng)
            for _ in range(10):
                x=p.mutate(x,rng)
                self.assertEqual(np.isfinite(x).sum(),6)
                self.assertAlmostEqual(p.evaluate(x),p.reference(x),places=12)
    def test_visibility(self):
        r=np.array([.4,.4,.6,.6])
        self.assertFalse(visible([0,.5],[1,.5],r))
        self.assertTrue(visible([0,.2],[1,.2],r))
        self.assertFalse(visible([0,0],[1,1],r))
    def test_cascade_feasibility(self):
        p=CascadeDesign(42,0,'test'); rng=np.random.default_rng(5)
        for _ in range(10):
            x=p.mutate(p.sample(rng),rng)
            self.assertTrue(p.valid(x))
            self.assertGreaterEqual(p.evaluate(x),0)
if __name__=='__main__': unittest.main()
