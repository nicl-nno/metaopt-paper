"""Audit new endpoints against independent mechanical/graph/coverage oracles."""
import argparse,csv,hashlib,time
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
from threadpoolctl import threadpool_limits
from .data import ROOT,load,source,TrussTask,NetworkTask,NetworkDesign,CoverageDesign,CascadeDesign,attack_priorities,fast_robustness
from .worker import save
from experiments.gendesign_rank.audit import independent_compliance
from experiments.gendesign_rank.experiment import structure
from experiments.network_robustness.audit import reference_robustness
from experiments.ranking_stress.problems import cascade
def audit_block(folder,kind,b):
    complete=load(folder/'COMPLETED.json');cfg=load(folder/'config.json');base=source(kind,b)
    for filename in ['input_hashes.json','source_hashes.json']:
        for path,digest in load(folder/filename).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    smoke=complete['smoke'];selected=load(folder/'selection.json');model_hashes={}
    for method,item in selected.items():
        scores=defaultdict(list);recipe_lookup={}
        for row in item['records']:
            key=str(row['recipe']);scores[key].append(row['validation_log_regret']);recipe_lookup[key]=row['recipe']
        chosen=min(scores,key=lambda key:np.mean(scores[key]))
        assert recipe_lookup[chosen]==item['selected_recipe'] and all(len(v)==len(cfg['model_seeds']) for v in scores.values())
        for ms in cfg['model_seeds']:
            path=folder/f'{method}_{ms}.joblib';model_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    starts={};problems={};rank_rows={}
    if kind=='truss':
        for row in csv.DictReader((base/'results.csv').open()):
            if row['method']=='ranknet':rank_rows[row['task_id'],int(row['model_seed']),int(row['search_seed'])]=row
        # Only initial trajectories are loaded; full historical records are not duplicated.
        initial=defaultdict(list)
        for row in csv.DictReader((base/'trajectories.csv').open()):
            if row['method']=='ranknet' and int(row['model_seed'])==17 and int(row['search_seed']) in cfg['search_seeds'] and int(row['evaluations'])<=16:
                initial[row['task_id'],int(row['search_seed'])].append(float(row['best_compliance']))
    if kind in ['coverage','cascade']:
        for row in load(base/'rows.json'):
            if row['method']=='ranknet':rank_rows[row['task_id'],row['model_seed'],row['search_seed']]=row
    paths=sorted((folder/'runs').glob('*.json'));assert len(paths)==complete['searches']
    audited=0;fresh_checks=0;endpoints=0;identities=set()
    for path in paths:
        obj=load(path);row=obj['row'];tid=row['task_id'];ms=row['model_seed'];ss=row['search_seed'];method=row['method']
        identity=tid,method,ms,ss;assert identity not in identities;identities.add(identity)
        assert row['evaluations']==cfg['budget']
        if kind=='truss':
            if tid not in problems:problems[tid]=structure(TrussTask(**obj['design']['task']),cfg)
            problem=problems[tid];g=np.array(obj['design']['genes']);history=obj['history'];reference=independent_compliance(problem,g)
            assert np.all((g>=0)&(g<=4)) and np.all(g[:problem.mandatory]>=1)
            assert np.isclose(problem.areas(g)@problem.length,cfg['volume'],atol=1e-12)
            assert np.isclose(reference,row['best_compliance'],rtol=1e-9,atol=1e-10)
            trace=[r['best_compliance'] for r in history];assert len(trace)==cfg['budget'] and np.all(np.diff(trace)<=0)
            assert trace[-1]==row['best_compliance'] and trace[:16]==initial[tid,ss]
            assert row['initial_best_compliance']==float(rank_rows[tid,ms,ss]['initial_best_compliance']);endpoints+=1
        elif kind=='network':
            task=NetworkTask(**obj['design']['task'])
            if tid not in problems:problems[tid]=NetworkDesign(task,cfg)
            problem=problems[tid];original=load(base/'runs'/f'{tid}__ranknet__{ms}__{ss}.json')
            assert row['initial_graphs_sha256']==original['row']['initial_graphs_sha256']
            assert [r['best_r'] for r in obj['history'][:16]]==[r['best_r'] for r in original['history'][:16]]
            assert len(obj['history'])==cfg['budget'] and np.all(np.diff([r['best_r'] for r in obj['history']])>=0)
            assert [p['evaluations'] for p in obj['design']['checkpoints']]==[96,192]
            fresh=attack_priorities(task.n,task.attack_seed+1000000,cfg['fresh_scenarios'])
            for cp in obj['design']['checkpoints']:
                adj=np.zeros((task.n,task.n),bool)
                for u,v in cp['edges']:adj[u,v]=adj[v,u]=True
                graph=nx.from_numpy_array(adj);assert nx.is_connected(graph) and nx.number_of_selfloops(graph)==0
                np.testing.assert_array_equal(adj.sum(1),problem.degree)
                assert np.count_nonzero(adj&~problem.base)//2<=problem.max_new_edges
                ref,curves=reference_robustness(adj,problem.priorities)
                assert ref==cp['best_r']==obj['history'][cp['evaluations']-1]['best_r']
                actual,fast_curves=fast_robustness(adj,problem.priorities);assert actual==ref;np.testing.assert_array_equal(curves,fast_curves)
                assert fast_robustness(adj,fresh)[0]==cp['fresh_r']
                assert reference_robustness(adj,fresh[:1])[0]==fast_robustness(adj,fresh[:1])[0]
                endpoints+=1;fresh_checks+=1
        else:
            if tid not in problems:
                spec=load(base/'tasks'/f'{tid}.json');klass=CoverageDesign if kind=='coverage' else CascadeDesign
                problems[tid]=klass(spec['seed'],spec['index'],spec['split'])
            problem=problems[tid];x=np.array(row['design'],dtype=float if kind=='coverage' else bool)
            if kind=='coverage':x[x==10.]=np.nan;assert np.isfinite(x).sum()==6;ref=problem.reference(x)
            else:assert problem.valid(x);ref=cascade(x,problem.alpha,problem.priorities[0],reference=True)[0]
            assert abs(ref-row['best'])<1e-10;assert len(row['trajectory'])==96 and np.all(np.diff(row['trajectory'])>=0)
            assert row['trajectory'][-1]==row['best']
            original=rank_rows[tid,ms,ss];assert row['initial_sha256']==original['initial_sha256'] and row['trajectory'][:16]==original['trajectory'][:16];endpoints+=1
        audited+=1
        if audited%16==0:save(folder/'audit_progress.json',dict(searches=audited,total=len(paths),updated=time.time()))
    expected={(tid,m,ms,ss) for tid in problems for m in selected for ms in cfg['model_seeds'] for ss in cfg['search_seeds']}
    assert identities==expected
    result=dict(passed=True,searches=audited,independent_endpoints=endpoints,fresh_endpoint_checks=fresh_checks,input_source_hashes_verified=True,matched_initial_populations=True,validation_only_selection_verified=True,selected_model_hashes=model_hashes,smoke=smoke)
    save(folder/'AUDIT.json',result);return result
def run(root,watch=False,kind_filter=None):
    threadpool_limits(1);root=Path(root)
    pending=[(k,b) for b in range(5) for k in ['coverage','cascade','network','truss'] if (kind_filter is None or k==kind_filter) and b<(3 if k in ['coverage','cascade'] else 5)]
    if 'smoke' in root.name:pending=[(k,0) for k in ['coverage','cascade','network','truss'] if kind_filter is None or k==kind_filter]
    results={}
    while pending:
        progressed=False
        for kind,b in pending[:]:
            folder=root/kind/f'block_{b:02d}'
            if not (folder/'COMPLETED.json').exists():continue
            result=load(folder/'AUDIT.json') if (folder/'AUDIT.json').exists() else audit_block(folder,kind,b)
            assert result['passed'];results[f'{kind}/{b}']=result;pending.remove((kind,b));progressed=True
            print('AUDITED',kind,b,result['searches'],flush=True)
        if pending:
            if not watch:raise RuntimeError('Incomplete blocks: '+str(pending))
            if not progressed:time.sleep(10)
    save(root/(f'AUDIT_{kind_filter}.json' if kind_filter else 'AUDIT.json'),dict(passed=True,blocks=results,searches=sum(r['searches'] for r in results.values()),endpoints=sum(r['independent_endpoints'] for r in results.values())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--watch',action='store_true');p.add_argument('--kind');a=p.parse_args();run(a.root,a.watch,a.kind)
