"""One resumable offline-data block, with validation-only surrogate selection."""
import argparse,hashlib,json,os,platform,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from .data import ROOT,load,read_data,run_search,original_training
from .models import descriptor,standardized_targets,validation,train_gnn,recipes,Forest,RFFGaussianProcess
METHODS=['listmle','huber','rf','rff_gp']
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def hashes(paths):return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def run(kind,block,out,smoke=False):
    torch.set_num_threads(1);threadpool_limits(1)
    out=Path(out)/kind/f'block_{block:02d}';out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__).parent/n for n in ['worker.py','models.py','data.py','PROTOCOL.md']]
    for name in ['gendesign_rank','network_robustness','ranking_stress']:
        files+=list((ROOT/'experiments'/name).glob('*.py'))
    files+=list((ROOT/'gamlet/surrogate/encoders').glob('*.py'))
    frozen=hashes(files);manifest=out/'source_hashes.json'
    if manifest.exists():assert load(manifest)==frozen,'Code changed; use a new output root'
    else:
        save(manifest,frozen)
        for p in files:
            target=out/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    if (out/'COMPLETED.json').exists():return
    save(out/'state.json',dict(phase='loading',updated=time.time()))
    data,tests,cfg,inputs=read_data(kind,block,smoke);save(out/'input_hashes.json',hashes(inputs));save(out/'config.json',cfg)
    progress=lambda obj:save(out/'state.json',dict(obj,updated=time.time()))
    selected={};models={};flat=None
    for method in METHODS:
        path=out/f'{method}_selection.json'
        if path.exists():
            selected[method]=load(path)
            for ms in cfg['model_seeds']:models[method,ms]=joblib.load(out/f'{method}_{ms}.joblib')
            continue
        all_records=[];best=float('inf');best_models=None;chosen=None
        candidates=recipes(method)
        if smoke:candidates=candidates[:1]
        if method in ['rf','rff_gp'] and flat is None:
            progress(dict(phase='descriptors'))
            flat={tid:descriptor(*d['features']) for tid,d in data.items()}
        for ri,recipe in enumerate(candidates):
            fitted={};records=[]
            for ms in cfg['model_seeds']:
                progress(dict(phase='training',method=method,recipe=ri,recipes=len(candidates),model_seed=ms))
                tick=time.time()
                if method in ['listmle','huber']:
                    net,record=train_gnn(recipe,ms,data,cfg,kind,lambda p:progress(dict(p,phase='training',method=method,recipe=ri,model_seed=ms)))
                    if not smoke:
                        original=original_training(kind,block,ms)
                        assert record['initial_weights_sha256']==original['initial_weights_sha256']
                        assert record['optimizer_steps']==int(original['optimizer_steps'])
                        assert record['parameters']==int(original.get('parameters',original.get('trainable_parameters')))
                else:
                    train=[d for d in data.values() if d['split']=='train']
                    x=np.concatenate([flat[tid] for tid,d in data.items() if d['split']=='train'])
                    y=np.concatenate(standardized_targets(train,kind,recipe['transform'],recipe['normalization']))
                    net=(Forest(ms,recipe['leaf']) if method=='rf' else RFFGaussianProcess(ms,recipe['length'],recipe['noise'])).fit(x,y)
                    vals=[]
                    for tid,d in data.items():
                        if d['split']!='validation':continue
                        j=int(np.argmin(net.predict(flat[tid])));v=d['values']
                        vals.append(np.log(v[j]/v.min()) if kind=='truss' else np.log(v.max()/v[j]))
                    record=dict(recipe=recipe,model_seed=ms,validation_log_regret=float(np.mean(vals)),seconds=time.time()-tick,descriptor_dimension=x.shape[1],training_designs=len(x))
                fitted[ms]=net;records.append(record);all_records.append(record)
                save(out/f'{method}_training_records.json',all_records)
            score=float(np.mean([r['validation_log_regret'] for r in records]))
            print(kind,block,method,'recipe',ri,'validation',score,flush=True)
            if score<best:best=score;chosen=recipe;best_models=fitted
        for ms,net in best_models.items():joblib.dump(net,out/f'{method}_{ms}.joblib',compress=3);models[method,ms]=net
        selected[method]=dict(method=method,selected_recipe=chosen,mean_validation_log_regret=best,records=all_records)
        save(path,selected[method])
    save(out/'selection.json',selected)
    # This order is fixed by a task-independent seed, not by observed method results.
    order=np.random.default_rng(260909+block).permutation(METHODS).tolist()
    rows=[];expected=len(tests)*len(cfg['model_seeds'])*len(cfg['search_seeds'])*len(METHODS)
    for task in tests:
        tid=task[0] if kind in ['coverage','cascade'] else task.task_id
        for ss in cfg['search_seeds']:
            for ms in cfg['model_seeds']:
                for method in order:
                    dest=out/'runs'/f'{tid}__{method}__{ms}__{ss}.json'
                    if dest.exists():obj=load(dest)
                    else:
                        obj=run_search(kind,task,method,models[method,ms],ms,ss,cfg);obj['row']['block']=block;save(dest,obj)
                    rows.append(obj['row']);progress(dict(phase='search',completed=len(rows),expected=expected,task=tid,method=method))
        print(kind,block,'searched',tid,len(rows),'/',expected,flush=True)
    assert len(rows)==expected
    save(out/'rows.json',rows);save(out/'COMPLETED.json',dict(searches=expected,online_evaluations=sum(int(r['evaluations']) for r in rows),new_offline_evaluations=0,smoke=smoke))
    progress(dict(phase='completed',completed=expected,expected=expected))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['truss','network','coverage','cascade']);p.add_argument('--block',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.kind,a.block,a.out,a.smoke)
    except Exception:
        save(Path(a.out)/a.kind/f'block_{a.block:02d}'/'LAST_ERROR.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise
