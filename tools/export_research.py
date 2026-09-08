"""Export local research code and a dated, lossless snapshot of experiment outputs.

No training process is stopped or modified. Result file inventory and progress
metadata are frozen before packing. Active campaigns remain explicitly partial.
Archives contain original paths relative to the GAMLET root; no pickle is loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

REPO=Path(__file__).resolve().parents[1]
REPORT_NAMES={'REPORT.md','REPORT_RU.md','INTERPRETATION_RU.md','SUMMARY.json','summary.json',
              'statistics.json','COMPLETED.json','verification.json','state.json','PROTOCOL.md'}
SKIP_PARTS={'__pycache__','.pytest_cache','.git','.ipynb_checkpoints'}
LEGACY_ASSETS={'data','experiment_logs','model_checkpoints'}

def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def stable_bytes(path):
    for _ in range(5):
        before=path.stat(); data=path.read_bytes(); after=path.stat()
        if before.st_size==after.st_size==len(data) and before.st_mtime_ns==after.st_mtime_ns:
            return data
        time.sleep(.05)
    raise RuntimeError('File changed repeatedly during snapshot: '+str(path))
def transient(path):
    return bool(set(path.parts)&SKIP_PARTS) or path.suffix in {'.pyc','.pyo','.tmp','.lock'}

def export(source, name):
    source=source.resolve()
    destination=REPO/'results'/name
    if destination.exists(): raise SystemExit('Use a new snapshot name; existing snapshots are immutable.')
    if not (source/'.git').exists(): raise SystemExit('Expected the original GAMLET checkout.')
    destination.mkdir(parents=True)
    started=stamp()
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    tracked=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
    code_records=[]; exclusions=[]
    for relative in filter(None,tracked):
        p=Path(relative)
        if (p.parts[0] in LEGACY_ASSETS and p.suffix not in {'.py','.ipynb'}) or (p.parts[0]=='notebooks' and p.suffix in {'.zip','.pickle','.pkl','.npz','.pt','.pth'}):
            exclusions.append({'path':relative,'reason':'Legacy upstream AutoML data/output, not used by the new optimization experiments'})
            continue
        src=source/p; data=stable_bytes(src); target=REPO/'code/GAMLET'/p
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        code_records.append({'path':target.relative_to(REPO).as_posix(),'sha256':sha(data),'bytes':len(data),'origin':'upstream '+commit})
    for src in sorted((source/'experiments').rglob('*')):
        if not src.is_file() or transient(src.relative_to(source)) or 'results' in src.relative_to(source).parts: continue
        p=src.relative_to(source); data=stable_bytes(src); target=REPO/'code/GAMLET'/p
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        code_records.append({'path':target.relative_to(REPO).as_posix(),'sha256':sha(data),'bytes':len(data),'origin':'local experiment additions'})
    for src in sorted(source.parent.glob('network_*.py')):
        data=stable_bytes(src); target=REPO/'code'/src.name
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        code_records.append({'path':target.relative_to(REPO).as_posix(),'sha256':sha(data),'bytes':len(data),'origin':'local diagnostic helper'})
    dump(REPO/'code/SOURCE_MANIFEST.json',{'captured_utc':started,'upstream_url':'https://github.com/ITMO-NSS-team/GAMLET',
         'upstream_commit':commit,'source_files':code_records,'excluded_legacy_assets':exclusions,
         'note':'Exact source bytes retained; no active worker files were edited.'})

    # Freeze the file list and progress metadata before reading large outputs.
    groups={}; records=[]; progress={}; metadata={}; omitted=[]
    for area in sorted((source/'experiments').iterdir()):
        result_root=area/'results'
        if not result_root.is_dir(): continue
        for run in sorted(result_root.iterdir()):
            if not run.is_dir(): continue
            files=sorted(p for p in run.rglob('*') if p.is_file() and not transient(p.relative_to(run)))
            runkey=f'{area.name}/{run.name}'
            progress[runkey]={'files_at_inventory':len(files),'root_completed_marker_at_inventory':(run/'COMPLETED.json').exists()}
            for p in files:
                rel=p.relative_to(run)
                if p.name in {'state.json','COMPLETED.json','FAILED.json','LAST_ERROR.json','SUMMARY.json'}:
                    metadata[p]=stable_bytes(p)
                if p.name=='state.json' and p.parent==run:
                    progress[runkey]['root_state_at_inventory']=json.loads(metadata[p].decode('utf-8-sig'))
                # Large campaigns are partitioned by their first-level block;
                # smaller files at a run root are stored in metadata.zip.
                unit=rel.parts[0] if len(rel.parts)>1 else 'metadata'
                groups.setdefault((area.name,run.name,unit),[]).append(p)
    archive_records=[]
    for (area,run,unit), files in groups.items():
        archive=destination/'archives'/area/run/(unit+'.zip')
        archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files:
                data=metadata[p] if p in metadata else stable_bytes(p)
                entry=p.relative_to(source).as_posix()
                z.writestr(entry,data)
                records.append({'archive':archive.relative_to(REPO).as_posix(),'member':entry,'sha256':sha(data),'bytes':len(data)})
                if p.name in REPORT_NAMES and 'source_snapshot' not in p.parts and 'code_snapshot' not in p.parts:
                    relative=p.relative_to(source/'experiments'/area/'results'/run)
                    out=destination/'reports'/area/run/relative
                    out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(data)
        size=archive.stat().st_size
        if size>=48*1024*1024:
            raise RuntimeError('Archive exceeds selected repository size bound; partition it further: '+str(archive))
        archive_records.append({'path':archive.relative_to(REPO).as_posix(),'sha256':sha(archive.read_bytes()),'bytes':size,'members':len(files)})
        print(f'Packed {area}/{run}/{unit}: {len(files)} files, {size/1024/1024:.1f} MiB',flush=True)
    # Timing/oracle diagnostics were produced next to the source checkout.
    extras=destination/'workspace_diagnostics'
    for p in source.parent.glob('network-*.json'):
        data=stable_bytes(p); extras.mkdir(exist_ok=True); (extras/p.name).write_bytes(data)
    manifest={'schema':1,'snapshot':name,'started_utc':started,'finished_utc':stamp(),
              'scope':'All saved outputs of the new truss/network/statistics experiments, including pilots and smoke tests.',
              'active_run_semantics':'Observational snapshot, not a globally atomic process checkpoint. Inventory and progress metadata fixed before packing; only committed files included. Never resume the original active process from this copy.',
              'excluded_transient_items':['__pycache__','*.pyc','*.tmp','*.lock','.pytest_cache'],
              'upstream_commit':commit,'runs':progress,'archives':archive_records,'files':records,
              'raw_bytes':sum(r['bytes'] for r in records),'archive_bytes':sum(r['bytes'] for r in archive_records)}
    dump(destination/'MANIFEST.json',manifest)
    lines=['# Snapshot '+name,'','Exported: '+manifest['finished_utc'],'',
           'All archives preserve the original paths relative to the GAMLET root. See MANIFEST.json for per-file SHA-256 and completion state.',
           '',f"{len(records):,} files; {manifest['raw_bytes']/1024/1024:.1f} MiB raw; {manifest['archive_bytes']/1024/1024:.1f} MiB compressed.",'',
           'Active runs are partial snapshots. A saved LAST_ERROR can describe a recovered historical failure; read completion markers and audits together.',
           '', '| Experiment / run | Saved files | Root completion marker |','|---|---:|---|']
    lines += [f"| {key} | {v['files_at_inventory']} | {v['root_completed_marker_at_inventory']} |" for key,v in progress.items()]
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    dump(REPO/'results/LATEST.json',{'snapshot':name,'manifest':(destination/'MANIFEST.json').relative_to(REPO).as_posix()})
    print(json.dumps({'snapshot':name,'files':len(records),'archives':len(archive_records),'raw_MiB':manifest['raw_bytes']/1024/1024,'archive_MiB':manifest['archive_bytes']/1024/1024}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in {'.','..'}: parser.error('name must be one path component')
    export(args.source,args.name)
