"""Paired task-level analysis of the fixed 2-model x 2-search seed slice."""
import argparse,csv,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .data import ROOT,source,load
from .worker import METHODS,save
def sign(values):
    v=np.round(values,12);w=int((v>0).sum());l=int((v<0).sum())
    return dict(wins=w,losses=l,ties=len(v)-w-l,p_two_sided=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.)
def summarize(records,kind,seed=26090911):
    oriented=np.array([r['log_ratio']*(-1 if kind=='truss' else 1) for r in records])
    groups=defaultdict(list);blocks=defaultdict(list)
    for i,r in enumerate(records):groups[r['block'],r['stratum']].append(i);blocks[r['block']].append(i)
    rng=np.random.default_rng(seed)
    # Within-stratum resampling conditions on the already fitted models/blocks.
    draws=np.mean([oriented[rng.choice(idx,size=(10000,len(idx)),replace=True)].mean(1) for idx in groups.values()],axis=0)
    means=np.array([oriented[idx].mean() for idx in blocks.values()]);mean=float(np.mean([oriented[idx].mean() for idx in groups.values()]))
    advantage=lambda z:100*(-np.expm1(-np.asarray(z)) if kind=='truss' else np.expm1(z))
    return dict(task_count=len(records),block_count=len(blocks),ranknet_advantage_percent=float(advantage(mean)),objective_ratio_change_percent=float(100*np.expm1(-mean if kind=='truss' else mean)),conditional_ci95_advantage_percent=advantage(np.quantile(draws,[.025,.975])).tolist(),conditional_sign_test=sign(oriented),block_sign_test=sign(means),block_advantage_percent=advantage(means).tolist(),ranknet_objective_mean=float(np.mean([r['rank_mean'] for r in records])),comparator_objective_mean=float(np.mean([r['base_mean'] for r in records])),mean_absolute_advantage=float(np.mean([r['absolute_advantage'] for r in records])),task_effects=records)
def analyze(root,watch=False):
    root=Path(root)
    if watch:
        while not all((root/p).exists() for p in ['COMPLETED.json','AUDIT.json']):time.sleep(10)
    complete=load(root/'COMPLETED.json');audit=load(root/'AUDIT.json')
    assert complete['searches']==audit['searches']==6272 and complete['online_evaluations']==1216512 and audit['passed']
    result={};family=[];manifest={}
    for kind in ['truss','network','coverage','cascade']:
        paired=defaultdict(lambda:defaultdict(list))
        for b in range(3 if kind in ['coverage','cascade'] else 5):
            folder=root/kind/f'block_{b:02d}';base=source(kind,b);cfg=load(folder/'config.json')
            assert not load(folder/'COMPLETED.json')['smoke']
            old={}
            if kind=='truss':
                oldpath=base/'results.csv'
                for r in csv.DictReader(oldpath.open()):
                    if r['method']=='ranknet':old[r['task_id'],int(r['model_seed']),int(r['search_seed'])]=r
            elif kind=='network':
                oldpath=base/'endpoint_results.csv'
                for r in csv.DictReader(oldpath.open()):
                    if r['method']=='ranknet':old[r['task_id'],int(r['model_seed']),int(r['search_seed']),int(r['budget'])]=r
            else:
                oldpath=base/'rows.json'
                for r in load(oldpath):
                    if r['method']=='ranknet':old[r['task_id'],r['model_seed'],r['search_seed']]=r
            manifest[oldpath.relative_to(ROOT).as_posix()]=hashlib.sha256(oldpath.read_bytes()).hexdigest()
            for path in sorted((folder/'runs').glob('*.json')):
                obj=load(path);r=obj['row'];tid=r['task_id'];ms=r['model_seed'];ss=r['search_seed'];method=r['method'];key=(b,tid)
                assert ms in [17,23] and ss in cfg['search_seeds']
                manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
                if kind=='network':
                    family_name=r['family'];n=int(r['n']);scope='id' if n in [50,100] else 'ood'
                    values=[(f"{cp['evaluations']}/{scope}/{bank}",float(old[tid,ms,ss,cp['evaluations']][bank]),cp[bank],family_name+'_'+str(n)) for cp in obj['design']['checkpoints'] for bank in ['best_r','fresh_r']]
                else:
                    bank='best_compliance' if kind=='truss' else 'best';scope='primary';original=old[tid,ms,ss]
                    stratum=str(int(tid[-2:])%4) if kind=='cascade' else 'all'
                    values=[(scope,float(original[bank]),float(r[bank]),stratum)]
                for endpoint,rank,base_value,stratum in values:
                    assert rank>0 and base_value>0,'Zero objective requires explicit additive analysis; do not omit'
                    paired[endpoint,method][key].append(dict(log_ratio=math.log(rank/base_value),rank=rank,base=base_value,stratum=stratum,ms=ms,ss=ss))
        result[kind]={}
        for (endpoint,method),tasks in paired.items():
            records=[]
            for (b,tid),vals in sorted(tasks.items()):
                assert len(vals)==4 and len({(v['ms'],v['ss']) for v in vals})==4
                records.append(dict(block=b,task_id=tid,stratum=vals[0]['stratum'],log_ratio=float(np.mean([v['log_ratio'] for v in vals])),rank_mean=float(np.mean([v['rank'] for v in vals])),base_mean=float(np.mean([v['base'] for v in vals])),absolute_advantage=float(np.mean([(v['base']-v['rank']) if kind=='truss' else (v['rank']-v['base']) for v in vals]))))
            v=summarize(records,kind);result[kind][endpoint+'/'+method]=v
            if endpoint==('96/id/best_r' if kind=='network' else 'primary'):family.append(v['conditional_sign_test'])
    assert len(family)==16;previous=0
    for i,test in enumerate(sorted(family,key=lambda t:t['p_two_sided'])):
        previous=max(previous,min(1.,(16-i)*test['p_two_sided']));test['p_holm_16']=previous
    save(root/'statistics.json',dict(exploratory=True,counts=complete,results=result,primary_holm_family=16,sign_convention='Positive advantage favors RankNet. Compliance advantage is relative reduction; utility advantage is relative increase.'))
    save(root/'analysis_input_hashes.json',manifest)
    lines=['# Other surrogate methods on the same benchmarks','',
           'Complete paired slice: two model seeds and two search seeds, all original offline blocks and test tasks. Positive advantage favors RankNet. No new offline labels. GNN fits use matched architecture/training steps; RF/GP use deterministic graph descriptors and validation-selected grids. Conditional task tests do not establish retraining generalization.','',
           '| Benchmark | Comparator | RankNet advantage % | Conditional 95% CI | Wins / tasks | Holm16 p | Block p |','|---|---|---:|---|---:|---:|---:|']
    for kind in ['truss','network','coverage','cascade']:
        for method in METHODS:
            endpoint='96/id/best_r' if kind=='network' else 'primary';v=result[kind][endpoint+'/'+method];t=v['conditional_sign_test'];ci=v['conditional_ci95_advantage_percent']
            lines.append(f"|{kind}|{method}|{v['ranknet_advantage_percent']:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|{t['wins']}/{v['task_count']}|{t['p_holm_16']:.3g}|{v['block_sign_test']['p_two_sided']:.4g}|")
    lines+=['','The wider historical RankNet aggregates are not used as comparators for this reduced technical-seed slice. All null and negative outcomes remain. Network192/fresh-bank/OOD estimates are descriptive and available in statistics.json. RFF-GP is a finite-feature posterior mean, not full GP/BO; RF is not a reproduction of SMAC. Few training blocks, prior benchmark inspection and graph-descriptor information loss limit interpretation.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(root/'ANALYSIS_COMPLETED.json',dict(primary_comparisons=16,searches=6272,audited=True));print('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--watch',action='store_true');a=p.parse_args();analyze(a.root,a.watch)
