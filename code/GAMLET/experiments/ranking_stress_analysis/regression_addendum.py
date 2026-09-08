"""Post-hoc robustness to pooled target normalization; primary protocol unchanged."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.network_robustness import experiment as training
from experiments.gendesign_rank.model import RepositoryGraphSurrogate
from experiments.ranking_stress.run import features,search,save
from experiments.ranking_stress.problems import CascadeDesign,CoverageDesign

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def run(root,kind):
    source=root/f'{kind}_20260908_v1';out=root/f'{kind}_regression_addendum_v1';out.mkdir(exist_ok=True)
    cfg=load(source/'protocol.json')['config'];klass=CoverageDesign if kind=='coverage' else CascadeDesign
    torch.set_num_threads(1)
    training.model=lambda c:RepositoryGraphSurrogate(node_dim=klass.node_dim,context_dim=6,hidden=c['hidden'],layers=c['layers'])
    save(out/'PROTOCOL.json',dict(posthoc=True,reason='Check whether the ranking gain depends on per-task target standardization. Added after completed coverage outcomes were seen; no change to primary protocol.',
         primary_run=source.name,losses=['global_mse','global_log_mse'],config=cfg,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inference='Descriptive robustness checks, not additional confirmatory tests. Select among all four regressors by validation alone.'))
    allrows=[]
    for b in range(3):
        base=source/f'block_{b:02d}';folder=out/f'block_{b:02d}';folder.mkdir(exist_ok=True)
        while not (base/'selection.json').exists(): time.sleep(5)
        datasets={};tests=[]
        for specpath in sorted((base/'tasks').glob('*.json')):
            spec=load(specpath);problem=klass(spec['seed'],spec['index'],spec['split']);tid=specpath.stem
            if spec['split']=='test': tests.append((tid,problem));continue
            arr=np.load(base/'offline'/f'{tid}.npz')
            datasets[tid]=dict(features=features(problem,arr['designs']),values=arr['values'],split=spec['split'])
        assert len(datasets)==16 and len(tests)==12
        records=[];models={}
        for ms in cfg['model_seeds']:
            for loss in ['global_mse','global_log_mse']:
                save(out/'state.json',dict(phase='training',block=b,loss=loss,model_seed=ms))
                info=folder/f'{loss}_{ms}_info.json'
                if info.exists():
                    record=load(info);net=training.model(cfg);net.load_state_dict(torch.load(folder/f'{loss}_{ms}.pt',weights_only=True));net.eval()
                else: net,record=training.train(loss,ms,datasets,cfg,folder);save(info,record)
                original=[r for r in load(base/'selection.json')['training'] if r['loss']=='mse' and r['model_seed']==ms][0]
                for key in ['initial_weights_sha256','optimizer_steps','parameters']: assert original[key]==record[key]
                records.append(record);models[loss,ms]=net
        combined=load(base/'selection.json')['training']+records
        chosen=min(['mse','log_mse','global_mse','global_log_mse'],key=lambda l:np.mean([r['validation_log_regret'] for r in combined if r['loss']==l]))
        save(folder/'selection.json',dict(selected_regression=chosen,training=combined))
        rows=[]
        for tid,problem in tests:
            for ss in cfg['search_seeds']:
                for (method,ms),net in models.items():
                    dest=folder/'search'/f'{tid}_{method}_{ms}_{ss}.json'
                    if dest.exists(): row=load(dest)
                    else:
                        row=search(problem,method,net,ss,cfg);row.update(block=b,task_id=tid,method=method,model_seed=ms,search_seed=ss);save(dest,row)
                    rows.append(row);save(out/'state.json',dict(phase='search',block=b,completed_in_block=len(rows),expected_in_block=96))
            print(kind,'pooled regression',b,tid,len(rows),flush=True)
        save(folder/'rows.json',rows);save(folder/'COMPLETED.json',dict(searches=len(rows)));allrows+=rows
    save(out/'rows.json',allrows);save(out/'COMPLETED.json',dict(searches=len(allrows),online_evaluations=sum(r['evaluations'] for r in allrows),new_offline_evaluations=0));save(out/'state.json',dict(phase='completed'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--benchmark',choices=['coverage','cascade'],required=True);a=p.parse_args();run(a.root,a.benchmark)
