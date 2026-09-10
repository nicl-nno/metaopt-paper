"""Operational recovery only: retry transient file locks, retain frozen science.

This adapter replaces JSON persistence, not models, datasets, selection, RNG or
evaluation budgets. The original worker source hashes remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.05 * 2**min(attempt,5), 1.))


def run(role, out, kind=None, block=None, workers=4):
    from . import worker
    worker.save = save
    out = Path(out).resolve()
    if role == 'worker':
        worker.run(kind, block, out)
    elif role == 'schedule':
        from . import schedule
        original = subprocess.Popen
        def launch(command, *args, **kwargs):
            command = list(command)
            if command[1:3] == ['-m', 'experiments.surrogate_comparison.worker']:
                command = command[:2] + ['experiments.surrogate_comparison.resilient', 'worker'] + command[3:]
            return original(command, *args, **kwargs)
        schedule.subprocess.Popen = launch
        schedule.run(out, workers)
    elif role == 'audit':
        from . import audit
        audit.run(out)
    elif role == 'analyze':
        from . import analyze
        analyze.analyze(out)
    elif role == 'workflow':
        # The Windows task owns this parent independently of any chat/tool process.
        from .data import ROOT
        os.chdir(ROOT)
        env = os.environ.copy()
        for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            env[name] = '1'
        record = dict(adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      unchanged_scientific_worker=True, started=time.time(), pid=os.getpid())
        save(out/'operational_recovery.json', record)
        logdir = ROOT.parent/'surrogate_controller_20260909'
        logdir.mkdir(exist_ok=True)
        for stage in ['schedule', 'audit', 'analyze']:
            save(out/'controller_status.json', dict(record, phase=stage, updated=time.time()))
            with (logdir/(stage+'.log')).open('ab') as stream:
                command = [sys.executable, '-u', '-m', 'experiments.surrogate_comparison.resilient', stage,
                           '--out', str(out), '--workers', str(workers)]
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), check=True)
        save(out/'controller_status.json', dict(record, phase='publication', updated=time.time()))
        repository = ROOT.parent/'metaopt-paper'
        with (logdir/'publication.log').open('ab') as stream:
            subprocess.run([sys.executable, '-u', 'tools/finalize_surrogate_comparison.py', '--source', str(ROOT),
                            '--name', 'snapshot_20260908_surrogate_comparison_final', '--push'], cwd=repository,
                           stdout=stream, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), check=True)
        save(out/'controller_status.json', dict(record, phase='completed', updated=time.time()))
    else:
        raise ValueError(role)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('role', choices=['worker','schedule','audit','analyze','workflow'])
    p.add_argument('--out', required=True)
    p.add_argument('--kind')
    p.add_argument('--block', type=int)
    p.add_argument('--workers', type=int, default=4)
    a=p.parse_args()
    try:
        run(a.role,a.out,a.kind,a.block,a.workers)
    except Exception:
        dest=Path(a.out)
        if a.role=='worker':dest=dest/a.kind/f'block_{a.block:02d}'
        save(dest/('RESILIENT_'+a.role+'_ERROR.json'),dict(traceback=traceback.format_exc(),updated=time.time()))
        raise
