"""Analyze all preselected stress benchmarks; independent of frozen worker sources."""
import argparse
from collections import defaultdict
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sign(values):
    a=np.round(values,12); w=int((a>0).sum()); l=int((a<0).sum())
    return dict(wins=w,losses=l,ties=len(a)-w-l,p_two_sided=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.)
def analyze(root):
    results={}; hashes={}; correction=[]
    for name in ['coverage','cascade']:
        folder=root/f'{name}_20260908_v1'
        complete=load(folder/'COMPLETED.json'); protocol=load(folder/'protocol.json')
        assert complete['searches']==648 and complete['online_evaluations']==62208
        rows=load(folder/'rows.json'); cfg=protocol['config']; selections={}
        for b in range(3): selections[b]=load(folder/f'block_{b:02d}/selection.json')['selected_regression']
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.npz','.pt']:
                hashes[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        lookup={(r['block'],r['task_id'],r['method'],r['model_seed'],r['search_seed']):r for r in rows}
        assert len(lookup)==len(rows)
        tasks=sorted({(r['block'],r['task_id']) for r in rows}); assert len(tasks)==36
        for row in rows:
            assert row['evaluations']==96 and len(row['trajectory'])==96
            assert abs(max(row['trajectory'])-row['best'])<1e-12 and np.diff(row['trajectory']).min()>=-1e-12
        results[name]=dict(counts=complete,selections=selections,comparisons={})
        for comp in ['selected_regression','mse','log_mse','ea','pool_random','random']:
            effects=[]; absolute=[]; records=[]; rankmeans=[]; basemeans=[]
            for b,tid in tasks:
                diffs=[]; absdiff=[]; ranks=[]; bases=[]
                method=selections[b] if comp=='selected_regression' else comp
                for ms in cfg['model_seeds']:
                    for ss in cfg['search_seeds']:
                        rank=lookup[b,tid,'ranknet',ms,ss]
                        base=lookup[b,tid,method,ms if method in ['mse','log_mse'] else None,ss]
                        assert rank['initial_sha256']==base['initial_sha256']
                        assert rank['best']>0 and base['best']>0,'Zero best: require explicit additive analysis'
                        diffs.append(math.log(rank['best']/base['best']))
                        absdiff.append(rank['best']-base['best']); ranks.append(rank['best']); bases.append(base['best'])
                effect=float(np.mean(diffs)); effects.append(effect); absolute.append(float(np.mean(absdiff)))
                rankmeans.append(np.mean(ranks)); basemeans.append(np.mean(bases))
                records.append(dict(block=b,task_id=tid,log_ratio=effect,absolute_difference=float(np.mean(absdiff))))
            effects=np.array(effects); byblock=np.array([np.mean([r['log_ratio'] for r in records if r['block']==b]) for b in range(3)])
            # Conditional bootstrap: retain fitted blocks, sample tasks within each
            # graph family/size stratum (coverage has one iid stratum per block).
            groups=defaultdict(list)
            for i,(b,tid) in enumerate(tasks): groups[b,int(tid[-2:])%4 if name=='cascade' else 0].append(i)
            rng=np.random.default_rng(26090899); draws=[]
            for _ in range(10000):
                sampled=np.concatenate([rng.choice(idx,len(idx),replace=True) for idx in groups.values()])
                draws.append(effects[sampled].mean())
            entry=dict(gain_percent=float(100*np.expm1(effects.mean())),mean_absolute_difference=float(np.mean(absolute)),rank_mean=float(np.mean(rankmeans)),baseline_mean=float(np.mean(basemeans)),
                       conditional_ci95_percent=(100*np.expm1(np.quantile(draws,[.025,.975]))).tolist(),
                       conditional_sign_test=sign(effects),block_sign_test=sign(byblock),block_gain_percent=(100*np.expm1(byblock)).tolist(),task_effects=records)
            results[name]['comparisons'][comp]=entry
            if comp not in ['mse','log_mse']: correction.append(entry['conditional_sign_test'])
    assert len(correction)==8
    previous=0.
    for i,test in enumerate(sorted(correction,key=lambda v:v['p_two_sided'])):
        previous=max(previous,min(1.,(8-i)*test['p_two_sided'])); test['p_holm_8']=previous
    out=root/'analysis_20260908_v1'; out.mkdir(exist_ok=True)
    (out/'statistics.json').write_text(json.dumps(dict(exploratory=True,results=results),indent=2),encoding='utf-8')
    (out/'input_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    lines=['# Exploratory coverage and cascade experiments','',
           'All preselected benchmarks are included. Each has 3 offline blocks,36 test tasks,648 searches,62208 online evaluations and6144 offline evaluations. Seeds are averaged within tasks; conditional tests do not establish generalization across new fitted blocks. No zero final objective was excluded.','',
           '| Benchmark | RankNet vs | Gain % | Conditional 95% CI | Wins/36 | Holm8 p | Block p |',
           '|---|---|---:|---|---:|---:|---:|']
    for name,result in results.items():
        for comp,v in result['comparisons'].items():
            t=v['conditional_sign_test']; ci=v['conditional_ci95_percent']; p=f"{t['p_holm_8']:.3g}" if 'p_holm_8' in t else 'descriptive'
            lines.append(f"|{name}|{comp}|{v['gain_percent']:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|{t['wins']}/36|{p}|{v['block_sign_test']['p_two_sided']:.3g}|")
    lines+=['','Positive effects favor ranking; negative effects favor the comparator. Model-free controls reuse the same run across model seeds without adding independent replicates. These synthetic adaptations are not reproductions of the cited papers or claims of real deployment safety.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'COMPLETED.json').write_text(json.dumps(dict(benchmarks=2,test_tasks=72,searches=1296,exploratory=True)),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
