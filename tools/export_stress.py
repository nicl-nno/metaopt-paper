"""Add completed exploratory benchmarks without duplicating existing archives."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def export(source,name):
    destination=ROOT/'results'/name
    assert not destination.exists(),'Snapshots are immutable; choose a new name'
    area=source/'experiments/ranking_stress'
    for run in ['coverage_20260908_v1','cascade_20260908_v1','analysis_20260908_v1','coverage_regression_addendum_v1','cascade_regression_addendum_v1']:
        assert (area/'results'/run/'COMPLETED.json').exists(),run
    assert json.loads((area/'results/analysis_20260908_v1/verification.json').read_text())['passed']
    assert json.loads((area/'results/analysis_20260908_v1/regression_addendum.json').read_text())['passed']
    latest=json.loads((ROOT/'results/LATEST.json').read_text())
    previous=json.loads((ROOT/latest['manifest']).read_text())
    manifest={**previous,'snapshot':name,'parent_manifest':latest['manifest'],'started_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Completed truss/network/statistical outputs plus all preselected exploratory coverage/cascade outputs. Unchanged archives are referenced from the preceding snapshot.',
              'archives':list(previous['archives']),'files':list(previous['files']),'runs':dict(previous['runs'])}
    reports={'REPORT.md','statistics.json','verification.json','COMPLETED.json','protocol.json','PROTOCOL.json','state.json','selection.json','REGRESSION_ADDENDUM.md','regression_addendum.json','coverage_example.png','coverage_example.pdf','coverage_example.md'}
    for run in sorted((area/'results').iterdir()):
        if not run.is_dir(): continue
        groups={}
        for p in sorted(run.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.tmp']: continue
            rel=p.relative_to(run); unit=rel.parts[0] if len(rel.parts)>1 else 'metadata'
            groups.setdefault(unit,[]).append(p)
        manifest['runs']['ranking_stress/'+run.name]=dict(files_at_inventory=sum(map(len,groups.values())),root_completed_marker_at_inventory=(run/'COMPLETED.json').exists())
        for unit,files in groups.items():
            archive=destination/'archives/ranking_stress'/run.name/(unit+'.zip'); archive.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in files:
                    data=p.read_bytes(); member=p.relative_to(source).as_posix(); z.writestr(member,data)
                    manifest['files'].append(dict(archive=archive.relative_to(ROOT).as_posix(),member=member,sha256=sha(data),bytes=len(data)))
                    if p.name in reports and 'source_snapshot' not in p.parts:
                        out=destination/'reports/ranking_stress'/run.name/p.relative_to(run);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
            data=archive.read_bytes(); assert len(data)<48*1024*1024
            manifest['archives'].append(dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(data),bytes=len(data),members=len(files)))
    code=json.loads((ROOT/'code/SOURCE_MANIFEST.json').read_text())
    code_records={r['path']:r for r in code['source_files']}
    for folder in [area,source/'experiments/ranking_stress_analysis']:
        for p in folder.iterdir():
            if not p.is_file() or p.suffix not in ['.py','.md','.json','.txt']: continue
            dst=ROOT/'code/GAMLET'/p.relative_to(source); data=p.read_bytes();dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
            rel=dst.relative_to(ROOT).as_posix();code_records[rel]=dict(path=rel,sha256=sha(data),bytes=len(data),origin='exploratory stress benchmark additions')
    code['source_files']=list(code_records.values());code['captured_utc']=datetime.now(timezone.utc).isoformat();dump(ROOT/'code/SOURCE_MANIFEST.json',code)
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    manifest['raw_bytes']=sum(r['bytes'] for r in manifest['files']);manifest['archive_bytes']=sum(r['bytes'] for r in manifest['archives'])
    dump(destination/'MANIFEST.json',manifest)
    dump(ROOT/'results/LATEST.json',dict(snapshot=name,manifest=(destination/'MANIFEST.json').relative_to(ROOT).as_posix()))
    (destination/'README.md').write_text(f'# {name}\n\nCompleted exploratory supplement. MANIFEST.json references unchanged main-experiment archives in the parent snapshot and new archives here. Verification and extraction through tools/research_artifacts.py include both.\n\n{len(manifest["files"])} original files; {len(manifest["archives"])} archives in the combined snapshot. All positive, null and negative stress-test outcomes are retained.\n',encoding='utf-8')
    print(json.dumps(dict(files=len(manifest['files']),archives=len(manifest['archives']),source_files=len(code['source_files'])),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    assert Path(a.name).name==a.name and a.name not in ['.','..'];export(a.source.resolve(),a.name)
