"""Check archived RankNet replay through the adapters before a new campaign."""
import argparse,csv
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
from .data import load,read_data,source,run_search
from .models import RepositoryGraphSurrogate
from .worker import save
def run(kind,out):
    torch.set_num_threads(1);threadpool_limits(1)
    data,tasks,cfg,_=read_data(kind,0,smoke=True);task=tasks[0];ss=cfg['search_seeds'][0];ms=17;base=source(kind,0)
    net=RepositoryGraphSurrogate(node_dim=next(iter(data.values()))['features'][0].shape[-1],context_dim=6,hidden=32,layers=3)
    net.load_state_dict(torch.load(base/f'ranknet_{ms}.pt',weights_only=True));net.eval()
    obj=run_search(kind,task,'ranknet',net,ms,ss,cfg)
    if kind=='truss':
        original=next(r for r in csv.DictReader((base/'results.csv').open()) if r['task_id']==task.task_id and r['method']=='ranknet' and int(r['model_seed'])==ms and int(r['search_seed'])==ss)
        assert obj['row']['best_compliance']==float(original['best_compliance'])
        value=obj['row']['best_compliance']
    elif kind=='network':
        original=load(base/'runs'/f'{task.task_id}__ranknet__{ms}__{ss}.json')
        assert obj['row']['best_r']==original['row']['best_r']
        assert obj['design']['checkpoints']==original['design']['checkpoints']
        assert obj['row']['initial_graphs_sha256']==original['row']['initial_graphs_sha256'];value=obj['row']['best_r']
    else:
        original=load(base/'search'/f'{task[0]}_ranknet_{ms}_{ss}.json')
        assert obj['row']['best']==original['best'] and obj['row']['trajectory']==original['trajectory']
        assert obj['row']['initial_sha256']==original['initial_sha256'];value=obj['row']['best']
    save(Path(out)/f'replay_{kind}.json',dict(passed=True,benchmark=kind,block=0,model_seed=ms,search_seed=ss,objective=value,evaluations=cfg['budget'],note='Exact archived RankNet endpoint replay; technical validation, excluded from scientific sample.'))
    print(kind,'exact archived replay PASS',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.kind,a.out)
