"""Descriptive pooled-regression comparisons and independent endpoint audit."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from experiments.ranking_stress.problems import CoverageDesign,CascadeDesign,cascade
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def analyze(root,watch=False):
    results={};hashes={}
    for kind,klass in [('coverage',CoverageDesign),('cascade',CascadeDesign)]:
        main=root/f'{kind}_20260908_v1';add=root/f'{kind}_regression_addendum_v1'
        if watch:
            while not (add/'COMPLETED.json').exists() or not (main/'COMPLETED.json').exists():time.sleep(5)
        assert load(add/'COMPLETED.json')['searches']==288
        primary=load(main/'rows.json');extra=load(add/'rows.json');rows=primary+extra
        lookup={(r['block'],r['task_id'],r['method'],r['model_seed'],r['search_seed']):r for r in rows}
        assert len(lookup)==len(rows)
        selected={b:load(add/f'block_{b:02d}/selection.json')['selected_regression'] for b in range(3)}
        spec=load(add/'PROTOCOL.json')
        assert spec['source_sha256']==hashlib.sha256((Path(__file__).parent/'regression_addendum.py').read_bytes()).hexdigest()
        for b in range(3):
            training=load(add/f'block_{b:02d}/selection.json')['training']
            for ms in [17,23]:
                group=[r for r in training if r['model_seed']==ms]
                assert len(group)==5
                for key in ['initial_weights_sha256','optimizer_steps','parameters']:assert len({r[key] for r in group})==1
        tasks=sorted({(r['block'],r['task_id']) for r in primary});assert len(tasks)==36
        comparisons={};audited=0
        for b,tid in tasks:
            t=load(main/f'block_{b:02d}/tasks/{tid}.json');p=klass(t['seed'],t['index'],t['split'])
            for r in [r for r in extra if r['block']==b and r['task_id']==tid]:
                x=np.array(r['design'],dtype=float if kind=='coverage' else bool)
                if kind=='coverage':x[x==10]=np.nan;value=p.reference(x)
                else:
                    assert p.valid(x);value=cascade(x,p.alpha,p.priorities[0],reference=True)[0]
                assert abs(value-r['best'])<1e-10 and r['evaluations']==96
                assert len(r['trajectory'])==96 and np.diff(r['trajectory']).min()>=-1e-12
                audited+=1
            print(kind,'additional regression audit',b,tid,audited,flush=True)
        for comp in ['global_mse','global_log_mse','selected_among_four']:
            effects=[];absolute=[]
            for b,tid in tasks:
                method=selected[b] if comp=='selected_among_four' else comp
                delta=[];absdelta=[]
                for ms in [17,23]:
                    for ss in [101,202]:
                        a=lookup[b,tid,'ranknet',ms,ss];d=lookup[b,tid,method,ms,ss]
                        assert a['initial_sha256']==d['initial_sha256'] and min(a['best'],d['best'])>0
                        delta.append(math.log(a['best']/d['best']));absdelta.append(a['best']-d['best'])
                effects.append(dict(block=b,task_id=tid,log_ratio=float(np.mean(delta))))
                absolute.append(np.mean(absdelta))
            comparisons[comp]=dict(gain_percent=float(100*np.expm1(np.mean([r['log_ratio'] for r in effects]))),
                mean_absolute_difference=float(np.mean(absolute)),task_wins=sum(r['log_ratio']>1e-12 for r in effects),task_count=36,
                block_gain_percent=[float(100*np.expm1(np.mean([r['log_ratio'] for r in effects if r['block']==b]))) for b in range(3)],task_effects=effects)
        assert audited==288
        results[kind]=dict(selected_regression_by_block=selected,comparisons=comparisons,independently_audited_endpoints=audited,additional_online_evaluations=27648)
        for folder in [main,add]:
            for p in folder.rglob('*.json'):hashes[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=root/'analysis_20260908_v1';out.mkdir(exist_ok=True)
    (out/'regression_addendum.json').write_text(json.dumps(dict(posthoc=True,passed=True,results=results,input_hashes=hashes),indent=2),encoding='utf-8')
    lines=['# Pooled-normalization robustness addendum','',
        'Added after the coverage outcome was seen. Descriptive analysis; primary comparisons and Holm family remain unchanged. Same offline labels, GNN architecture, initial weights, minibatches, optimizer steps,80 epochs and96 true evaluations; these extra fits use one CPU thread rather than the primary runs’ two. No new offline labels.','',
        '| Benchmark | RankNet vs | Gain % | Wins/36 | Block gains % |','|---|---|---:|---:|---|']
    for kind,result in results.items():
        for comp,v in result['comparisons'].items():lines.append(f"|{kind}|{comp}|{v['gain_percent']:+.3f}|{v['task_wins']}/36|{', '.join(f'{x:+.3f}' for x in v['block_gain_percent'])}|")
    lines+=['','All576 additional endpoint designs independently recomputed. Selection among four regressors uses validation only. This addendum cannot be represented as preregistered confirmatory evidence.']
    (out/'REGRESSION_ADDENDUM.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args();analyze(a.root,a.watch)
