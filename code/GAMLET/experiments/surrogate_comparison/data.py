"""Read the exact archived task splits, offline labels and search implementations."""
from pathlib import Path
import json,csv
import numpy as np
from experiments.gendesign_rank import experiment as tr
from experiments.gendesign_rank.truss import Task as TrussTask
from experiments.gendesign_rank.model import tensors
from experiments.network_robustness import experiment as nw
from experiments.network_robustness.problem import Task as NetworkTask,NetworkDesign,attack_priorities
from experiments.network_robustness.fast_oracle import fast_robustness
from experiments.ranking_stress.run import features,search as stress_search
from experiments.ranking_stress.problems import CoverageDesign,CascadeDesign
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def source(kind,b):
    if kind=='truss':return ROOT/f'experiments/gendesign_rank/results/overnight_20260906_v2/main_{b:02d}'
    if kind=='network':return ROOT/f'experiments/network_robustness/results/replication_20260907_v1/block_{b:02d}'
    return ROOT/f'experiments/ranking_stress/results/{kind}_20260908_v1/block_{b:02d}'
def read_data(kind,b,smoke=False):
    base=source(kind,b);data={};tests=[];inputs=[]
    if kind in ['truss','network']:
        cfg=load(base/'config.json');specs=load(base/'tasks.json');inputs=[base/'config.json',base/'tasks.json']
        for spec in specs:
            task=TrussTask(**spec) if kind=='truss' else NetworkTask(**spec)
            if task.split=='test':tests.append(task);continue
            if smoke and sum(d['split']==task.split for d in data.values())>=(2 if task.split=='train' else 1):continue
            if kind=='truss':
                path=base/'offline_data.npz';inputs.append(path)
                with np.load(path) as arr:xs=arr[task.task_id+'_genes'];values=arr[task.task_id+'_compliance']
                fs=tensors(tr.stack_features(tr.structure(task,cfg),xs))
            else:
                path=base/'offline'/f'{task.task_id}.npz';inputs.append(path)
                with np.load(path) as arr:xs=arr['adjacency'];values=arr['robustness']
                fs=nw.features(NetworkDesign(task,cfg),xs)
            data[task.task_id]=dict(features=fs,values=values,split=task.split)
    else:
        cfg=load(base.parent/'protocol.json')['config'];inputs=[base.parent/'protocol.json']
        klass=CoverageDesign if kind=='coverage' else CascadeDesign
        for path in sorted((base/'tasks').glob('*.json')):
            spec=load(path);inputs.append(path);problem=klass(spec['seed'],spec['index'],spec['split']);tid=path.stem
            if spec['split']=='test':tests.append((tid,problem));continue
            if smoke and sum(d['split']==spec['split'] for d in data.values())>=(2 if spec['split']=='train' else 1):continue
            path=base/'offline'/f'{tid}.npz';inputs.append(path)
            with np.load(path) as arr:xs=arr['designs'];values=arr['values']
            data[tid]=dict(features=features(problem,xs),values=values,split=spec['split'])
    cfg=dict(cfg,model_seeds=[17,23],search_seeds=[1001,1002] if kind=='truss' else [101,202],threads=1)
    if smoke:
        cfg.update(epochs=2,model_seeds=[17],search_seeds=cfg['search_seeds'][:1]);tests=tests[:1]
    return data,tests,cfg,list(dict.fromkeys(inputs))
def run_search(kind,task,method,net,ms,ss,cfg):
    if kind=='truss':
        row,history,design=tr.search(task,method,net,ms,ss,cfg);return dict(row=row,history=history,design=design)
    if kind=='network':
        row,history,design=nw.search(task,cfg,method,net,ms,ss)
        pri=attack_priorities(task.n,task.attack_seed+1000000,cfg['fresh_scenarios'])
        for cp in design['checkpoints']:
            adj=np.zeros((task.n,task.n),bool)
            for u,v in cp['edges']:adj[u,v]=adj[v,u]=True
            cp['fresh_r']=fast_robustness(adj,pri)[0]
        return dict(row=row,history=history,design=design)
    tid,problem=task;row=stress_search(problem,method,net,ss,cfg);row.update(task_id=tid,method=method,model_seed=ms,search_seed=ss);return dict(row=row)
def original_training(kind,b,ms):
    base=source(kind,b)
    if kind in ['truss','network']:
        records=list(csv.DictReader((base/'training_summary.csv').open(encoding='utf-8')))
    else:records=load(base/'selection.json')['training']
    return next(r for r in records if r['loss']=='ranknet' and int(r['model_seed'])==ms)
