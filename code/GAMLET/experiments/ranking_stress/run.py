"""Run the frozen exploratory protocol; no online surrogate retraining."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.gendesign_rank.model import RepositoryGraphSurrogate, tensors
from experiments.network_robustness import experiment as training
from .problems import CascadeDesign, CoverageDesign, cascade

ROOT=Path(__file__).resolve().parents[2]
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)
def features(problem,xs):
    return tensors(tuple(np.stack(v) for v in zip(*(problem.features(x) for x in xs))))
def state_array(x): return np.nan_to_num(x,nan=10.).tolist()

def search(problem,method,net,seed,config):
    init=np.random.default_rng(seed); mutation=np.random.default_rng(seed+10000)
    selection=np.random.default_rng(seed+20000)
    archive=[]; seen=set(); trace=[]; hashes=[]
    def evaluate(x):
        key=problem.key(x); assert key not in seen
        value=problem.evaluate(x); assert np.isfinite(value) and 0<=value<=1
        archive.append((value,x.copy())); seen.add(key)
        trace.append(max(v for v,_ in archive))
        if len(archive)<=16: hashes.append(key)
    while len(archive)<16:
        x=problem.sample(init)
        if problem.key(x) not in seen: evaluate(x)
    while len(archive)<config['budget']:
        parents=[x for _,x in sorted(archive,key=lambda a:-a[0])[:16]]
        size=48 if net is not None or method=='pool_random' else 4
        pool=[]; pool_keys=set(seen)
        for attempt in range(20000):
            x=problem.sample(mutation) if method=='random' else problem.mutate(parents[int(mutation.integers(len(parents)))],mutation)
            key=problem.key(x)
            if key not in pool_keys: pool.append(x); pool_keys.add(key)
            if len(pool)==size: break
        assert len(pool)==size
        if net is not None:
            with torch.no_grad(): scores=net(*features(problem,pool)).numpy()
            order=np.argsort(scores,kind='stable'); chosen=list(order[:3])+[int(selection.choice(order[3:]))]
        elif method=='pool_random': chosen=selection.choice(size,4,replace=False)
        else: chosen=range(4)
        for i in chosen: evaluate(pool[i])
    value,x=max(archive,key=lambda a:a[0])
    reference=(problem.reference(x) if isinstance(problem,CoverageDesign)
               else cascade(x,problem.alpha,problem.priorities[0])[0])
    assert abs(reference-value)<1e-12
    assert len(seen)==config['budget']==len(trace)
    return dict(best=value,initial=trace[15],trajectory=trace,design=state_array(x),
                evaluations=len(trace),initial_sha256=hashlib.sha256(b''.join(hashes)).hexdigest(),
                endpoint_recomputed=True)


def run(kind,out,smoke=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    klass=CoverageDesign if kind=='coverage' else CascadeDesign
    cfg=dict(block_seeds=[26090871,26090872,26090873],train_tasks=12,validation_tasks=4,test_tasks=12,
             designs_per_task=128,epochs=80,batch_size=32,learning_rate=.001,hidden=32,layers=3,
             model_seeds=[17,23],search_seeds=[101,202],budget=96,training_heartbeat=True)
    if smoke: cfg.update(block_seeds=[91911],train_tasks=2,validation_tasks=1,test_tasks=1,designs_per_task=16,epochs=2,model_seeds=[17],search_seeds=[101],budget=20)
    source_files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'PROTOCOL.md',ROOT/'experiments/gendesign_rank/model.py',ROOT/'experiments/network_robustness/experiment.py',ROOT/'experiments/network_robustness/problem.py']+list((ROOT/'gamlet/surrogate/encoders').glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    frozen=dict(benchmark=kind,config=cfg,source_hashes=hashes,exploratory=True)
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text())==frozen,'Protocol/source changed; use a new run'
    else:
        save(out/'protocol.json',frozen)
        for p in source_files:
            target=out/'source_snapshot'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(p.read_bytes())
    # Scoped adapter of existing, identical training implementation for input dimension.
    training.model=lambda c:RepositoryGraphSurrogate(node_dim=klass.node_dim,context_dim=6,hidden=c['hidden'],layers=c['layers'])
    allrows=[]
    for block,blockseed in enumerate(cfg['block_seeds']):
        folder=out/f'block_{block:02d}'; folder.mkdir(exist_ok=True)
        if (folder/'COMPLETED.json').exists():
            allrows+=json.loads((folder/'rows.json').read_text()); continue
        save(out/'state.json',dict(phase='offline',block=block,updated=time.time()))
        datasets={}; problems=[]
        rng=np.random.default_rng(blockseed+(100000 if kind=='coverage' else 0))
        for split,count in [('train',cfg['train_tasks']),('validation',cfg['validation_tasks']),('test',cfg['test_tasks'])]:
            for i in range(count):
                problem=klass(int(rng.integers(2**30)),i,split); tid=f'{split}_{i:02d}'
                save(folder/'tasks'/f'{tid}.json',problem.spec)
                if split=='test': problems.append((tid,problem)); continue
                offline=folder/'offline'/f'{tid}.npz'; offline.parent.mkdir(exist_ok=True)
                if offline.exists():
                    data=np.load(offline); xs=data['designs']; values=data['values']
                else:
                    sampler=np.random.default_rng(problem.spec['seed']+11000)
                    xs=[]; keys=set()
                    while len(xs)<cfg['designs_per_task']:
                        x=problem.sample(sampler); key=problem.key(x)
                        if key not in keys: xs.append(x); keys.add(key)
                    values=np.array([problem.evaluate(x) for x in xs])
                    np.savez_compressed(offline,designs=np.stack(xs),values=values)
                assert np.isfinite(values).all() and (values>=0).all() and (values<1).all()
                datasets[tid]=dict(features=features(problem,xs),values=values,split=split)
                print(kind,block,tid,float(values.min()),float(values.max()),flush=True)
        models={}; records=[]
        for seed in cfg['model_seeds']:
            for loss in ['mse','log_mse','ranknet']:
                save(out/'state.json',dict(phase='training',block=block,loss=loss,model_seed=seed,updated=time.time()))
                info=folder/f'{loss}_{seed}_info.json'
                if info.exists():
                    net=training.model(cfg); net.load_state_dict(torch.load(folder/f'{loss}_{seed}.pt',weights_only=True)); net.eval()
                    record=json.loads(info.read_text())
                else:
                    net,record=training.train(loss,seed,datasets,cfg,folder); save(info,record)
                models[loss,seed]=net; records.append(record)
        for seed in cfg['model_seeds']:
            group=[r for r in records if r['model_seed']==seed]
            for k in ['initial_weights_sha256','optimizer_steps','parameters']: assert len({r[k] for r in group})==1
        selected=min(['mse','log_mse'],key=lambda l:np.mean([r['validation_log_regret'] for r in records if r['loss']==l]))
        save(folder/'selection.json',dict(selected_regression=selected,training=records))
        variants=[(loss,seed,net) for (loss,seed),net in models.items()]+[(m,None,None) for m in ['ea','pool_random','random']]
        rows=[]
        for tid,problem in problems:
            for searchseed in cfg['search_seeds']:
                init_hashes=set()
                for method,modelseed,net in variants:
                    dest=folder/'search'/f'{tid}_{method}_{modelseed}_{searchseed}.json'
                    if dest.exists(): row=json.loads(dest.read_text())
                    else:
                        row=search(problem,method,net,searchseed,cfg)
                        row.update(block=block,task_id=tid,method=method,model_seed=modelseed,search_seed=searchseed)
                        save(dest,row)
                    rows.append(row); init_hashes.add(row['initial_sha256'])
                    save(out/'state.json',dict(phase='search',block=block,completed_in_block=len(rows),expected_in_block=len(problems)*len(cfg['search_seeds'])*len(variants),updated=time.time()))
                assert len(init_hashes)==1
            print(kind,block,'searched',tid,len(rows),flush=True)
        save(folder/'rows.json',rows); save(folder/'COMPLETED.json',dict(searches=len(rows),selected_regression=selected))
        allrows+=rows
    save(out/'rows.json',allrows)
    save(out/'COMPLETED.json',dict(searches=len(allrows),online_evaluations=sum(r['evaluations'] for r in allrows),offline_evaluations=len(cfg['block_seeds'])*(cfg['train_tasks']+cfg['validation_tasks'])*cfg['designs_per_task']))
    save(out/'state.json',dict(phase='completed',updated=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--benchmark',choices=['coverage','cascade'],required=True); parser.add_argument('--out',required=True); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); run(args.benchmark,args.out,args.smoke)
