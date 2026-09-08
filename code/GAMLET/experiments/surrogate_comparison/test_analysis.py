import unittest
import numpy as np
from .analyze import summarize
class AnalysisTests(unittest.TestCase):
    def test_sign_convention_and_task_units(self):
        records=[dict(block=b,task_id=str(i),stratum='all',log_ratio=float(np.log(.95)),rank_mean=.95,base_mean=1.,absolute_advantage=.05) for b in range(5) for i in range(24)]
        v=summarize(records,'truss');self.assertAlmostEqual(v['ranknet_advantage_percent'],5.)
        self.assertEqual(v['task_count'],120);self.assertEqual(v['conditional_sign_test']['wins'],120)
        self.assertEqual(v['block_sign_test']['p_two_sided'],.0625)
        for r in records:r.update(log_ratio=float(np.log(1.05)),rank_mean=1.05)
        v=summarize(records,'coverage');self.assertAlmostEqual(v['ranknet_advantage_percent'],5.)
if __name__=='__main__':unittest.main()
