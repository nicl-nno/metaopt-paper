import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1]/"analyze.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StatisticalChecks(unittest.TestCase):
    def test_signed_ranks_match_scipy_exact_without_ties(self):
        values = np.array([.03, -.01, .08, .06, -.04, .02])
        actual = module.signed_rank_exact(values)
        expected = stats.wilcoxon(values, method="exact")
        self.assertEqual(actual["W"], expected.statistic)
        self.assertAlmostEqual(actual["p_two_sided"], expected.pvalue)

    def test_tied_rank_distribution_matches_exhaustive_enumeration(self):
        values = np.array([1., 1., -2., 0., 3.])
        d = values[values != 0]
        ranks = stats.rankdata(abs(d))
        sums = np.array([np.dot(np.array(signs) > 0, ranks) for signs in itertools.product((-1, 1), repeat=len(d))])
        observed = ranks[d > 0].sum()
        p = min(1., 2*min(np.mean(sums <= observed), np.mean(sums >= observed)))
        self.assertEqual(module.signed_rank_exact(values)["p_two_sided"], p)

    def test_exact_block_resolution_and_zero_handling(self):
        for n in (2, 5):
            result = module.block_sensitivity(np.arange(1, n+1)/100)
            self.assertEqual(result["sign_test"]["p_two_sided"], 2/2**n)
            self.assertEqual(result["sign_flip_p_two_sided"], 2/2**n)
        self.assertEqual(module.sign_test([0, 0])["p_two_sided"], 1)

    def test_holm_preserves_order_and_caps_at_one(self):
        np.testing.assert_allclose(module.holm([.04, .01, .03]), [.06, .03, .06])


if __name__ == "__main__":
    unittest.main()
