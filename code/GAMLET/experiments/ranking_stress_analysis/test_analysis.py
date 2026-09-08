"""Statistical-unit checks on a known proportional-effect synthetic experiment."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from .analyze import analyze,sign

class AnalysisTests(unittest.TestCase):
    def test_sign_ties_and_two_sided_floor(self):
        self.assertEqual(sign([1,1,1])['p_two_sided'],.25)
        self.assertEqual(sign([0,0])['p_two_sided'],1.)
        self.assertEqual(sign([1,-1,0])['ties'],1)
    def test_technical_repeats_do_not_inflate_task_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for kind in ['coverage','cascade']:
                folder=root/f'{kind}_20260908_v1';folder.mkdir()
                (folder/'protocol.json').write_text(json.dumps(dict(config=dict(model_seeds=[17,23],search_seeds=[101,202]))))
                (folder/'COMPLETED.json').write_text(json.dumps(dict(searches=648,online_evaluations=62208)))
                rows=[]
                for b in range(3):
                    block=folder/f'block_{b:02d}';block.mkdir();(block/'selection.json').write_text(json.dumps(dict(selected_regression='mse')))
                    for t in range(12):
                        for ss in [101,202]:
                            for method,seeds in [('ranknet',[17,23]),('mse',[17,23]),('log_mse',[17,23]),('ea',[None]),('pool_random',[None]),('random',[None])]:
                                for ms in seeds:
                                    best=.525 if method=='ranknet' else .5
                                    rows.append(dict(block=b,task_id=f'test_{t:02d}',method=method,model_seed=ms,search_seed=ss,best=best,trajectory=[best]*96,evaluations=96,initial_sha256='common'))
                (folder/'rows.json').write_text(json.dumps(rows))
            with contextlib.redirect_stdout(io.StringIO()): analyze(root)
            result=json.loads((root/'analysis_20260908_v1/statistics.json').read_text())
            for kind in ['coverage','cascade']:
                for v in result['results'][kind]['comparisons'].values():
                    self.assertAlmostEqual(v['gain_percent'],5.,places=10)
                    self.assertAlmostEqual(v['mean_absolute_difference'],.025,places=10)
                    self.assertEqual(v['conditional_sign_test']['wins'],36)
                    self.assertEqual(v['block_sign_test']['wins'],3)
                    self.assertEqual(v['block_sign_test']['p_two_sided'],.25)
if __name__=='__main__': unittest.main()
