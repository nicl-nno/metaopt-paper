"""Independent endpoint and source-split audit of completed stress experiments."""
import argparse,hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import networkx as nx
import numpy as np
from experiments.ranking_stress.problems import CascadeDesign,CoverageDesign,cascade

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def audit(root,benchmark=None,watch=False):
    source=Path(__file__).resolve().parents[2]
    summary={}; graph_groups=defaultdict(list); graph_duplicates=[]
    for kind,klass in [('coverage',CoverageDesign),('cascade',CascadeDesign)]:
        if benchmark is not None and kind!=benchmark: continue
        folder=root/f'{kind}_20260908_v1'
        if not watch: assert (folder/'COMPLETED.json').exists()
        frozen=load(folder/'protocol.json'); cfg=frozen['config']
        for relative,digest in frozen['source_hashes'].items():
            assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==digest,relative
            assert hashlib.sha256((folder/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
        audited=0; offline=0; exact=0; parameter_counts=set(); seed_sets=[]
        for b in range(3):
            block=folder/f'block_{b:02d}'
            if watch:
                while not (block/'COMPLETED.json').exists(): time.sleep(5)
            assert (block/'COMPLETED.json').exists()
            selections=load(block/'selection.json'); records=selections['training']
            chosen=min(['mse','log_mse'],key=lambda loss:np.mean([r['validation_log_regret'] for r in records if r['loss']==loss]))
            assert chosen==selections['selected_regression']
            for ms in cfg['model_seeds']:
                group=[r for r in records if r['model_seed']==ms]
                assert {r['loss'] for r in group}=={'mse','log_mse','ranknet'}
                for key in ['initial_weights_sha256','optimizer_steps','parameters']: assert len({r[key] for r in group})==1
                assert group[0]['optimizer_steps']==cfg['epochs']*cfg['train_tasks']*4
                parameter_counts.add(group[0]['parameters'])
            rows=load(block/'rows.json'); assert len(rows)==216
            for p in (block/'tasks').glob('*.json'):
                spec=load(p); problem=klass(spec['seed'],spec['index'],spec['split']); tid=p.stem
                seed_sets.append(spec['seed'])
                if kind=='cascade':
                    g=nx.from_numpy_array(problem.base)
                    key=(len(g),tuple(sorted(dict(g.degree()).values())))
                    for other,other_id in graph_groups[key]:
                        if nx.is_isomorphic(g,other): graph_duplicates.append([other_id,f'{b}/{tid}'])
                    graph_groups[key].append((g,f'{b}/{tid}'))
                if spec['split']!='test':
                    arr=np.load(block/'offline'/f'{tid}.npz'); values=arr['values']; xs=arr['designs']
                    assert len(xs)==128 and np.isfinite(values).all()
                    assert len({problem.key(x) for x in xs})==128
                    # Independent sample of offline labels, selected by fixed indices.
                    for i in [0,31,63,95,127]:
                        ref=problem.reference(xs[i]) if kind=='coverage' else cascade(xs[i],problem.alpha,problem.priorities[0],reference=True)[0]
                        assert abs(ref-values[i])<1e-10
                        exact+=1
                    offline+=len(xs); continue
                for row in [r for r in rows if r['task_id']==tid]:
                    x=np.array(row['design'],dtype=float if kind=='coverage' else bool)
                    if kind=='coverage': x[x==10.]=np.nan; ref=problem.reference(x)
                    else:
                        assert problem.valid(x)
                        ref=cascade(x,problem.alpha,problem.priorities[0],reference=True)[0]
                    assert abs(ref-row['best'])<1e-10
                    audited+=1; exact+=1
                print(kind,b,tid,'audited endpoints',audited,flush=True)
                progress=root/'analysis_20260908_v1';progress.mkdir(exist_ok=True)
                (progress/f'audit_progress_{kind}.json').write_text(json.dumps(dict(block=b,task=tid,endpoints_verified=audited,updated_unix=time.time())),encoding='utf-8')
        assert len(set(seed_sets))==len(seed_sets)
        assert audited==648 and offline==6144
        if watch:
            while not (folder/'COMPLETED.json').exists(): time.sleep(5)
        summary[kind]=dict(endpoint_designs_independently_verified=audited,offline_labels=offline,
                           independent_oracle_checks=exact,source_tasks=84,parameters=sorted(parameter_counts),source_hashes_passed=True)
    assert not graph_duplicates,graph_duplicates
    summary.update(passed=True,cascade_isomorphic_source_duplicates=graph_duplicates,completed_unix=time.time())
    dest=root/'analysis_20260908_v1'; dest.mkdir(exist_ok=True)
    name=f'verification_{benchmark}.json' if benchmark else 'verification.json'
    (dest/name).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if benchmark and all((dest/f'verification_{k}.json').exists() for k in ['coverage','cascade']):
        merged={'passed':True,'cascade_isomorphic_source_duplicates':[]}
        for k in ['coverage','cascade']:
            checked=load(dest/f'verification_{k}.json'); assert checked['passed']; merged[k]=checked[k]
        merged['completed_unix']=time.time()
        (dest/'verification.json').write_text(json.dumps(merged,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--benchmark',choices=['coverage','cascade']);p.add_argument('--watch',action='store_true');a=p.parse_args(); audit(a.root,a.benchmark,a.watch)
