"""Bounded campaign scheduler. No hypothesis-dependent stopping or retries."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from .worker import save
from .data import load,ROOT
def run(out,workers):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    jobs=[(k,b) for b in range(5) for k in ['coverage','cascade','network','truss'] if b<(3 if k in ['coverage','cascade'] else 5)]
    save(out/'CAMPAIGN.json',dict(jobs=jobs,workers=workers,expected_searches=6272,expected_online_evaluations=1216512,new_offline_evaluations=0,created=time.time()))
    pending=[(k,b) for k,b in jobs if not (out/k/f'block_{b:02d}'/'COMPLETED.json').exists()];active=[];failed=[]
    while pending or active:
        while pending and len(active)<workers:
            kind,b=pending.pop(0);log=out/f'{kind}_{b:02d}.log';stream=log.open('ab')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,'-m','experiments.surrogate_comparison.worker','--kind',kind,'--block',str(b),'--out',str(out)]
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((kind,b,process,stream));print('START',kind,b,process.pid,flush=True)
        for kind,b,process,stream in active[:]:
            if process.poll() is not None:
                stream.close();active.remove((kind,b,process,stream));print('END',kind,b,process.returncode,flush=True)
                if process.returncode:failed.append(dict(kind=kind,block=b,exit_code=process.returncode))
        save(out/'state.json',dict(active=[dict(kind=k,block=b,pid=p.pid) for k,b,p,_ in active],pending=pending,failed=failed,updated=time.time()))
        if active:time.sleep(5)
    assert not failed,failed
    complete=[load(out/k/f'block_{b:02d}'/'COMPLETED.json') for k,b in jobs]
    counts=dict(searches=sum(c['searches'] for c in complete),online_evaluations=sum(c['online_evaluations'] for c in complete),new_offline_evaluations=0,blocks=len(jobs))
    assert counts['searches']==6272 and counts['online_evaluations']==1216512
    save(out/'COMPLETED.json',counts);print(counts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.out,a.workers)
