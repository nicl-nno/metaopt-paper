"""Verify or extract every archived result using a snapshot manifest (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(data): return hashlib.sha256(data).hexdigest()

def run(manifest_path,destination,verify_members):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    expected={}
    for entry in manifest['files']: expected.setdefault(entry['archive'],{})[entry['member']]=entry
    count=0
    for arc in manifest['archives']:
        path=ROOT/arc['path']
        if sha(path.read_bytes())!=arc['sha256']: raise ValueError('Archive checksum mismatch: '+str(path))
        if destination or verify_members:
            with zipfile.ZipFile(path) as z:
                if set(z.namelist())!=set(expected[arc['path']]): raise ValueError('Archive member inventory mismatch')
                for member in z.namelist():
                    pure=PurePosixPath(member)
                    if pure.is_absolute() or '..' in pure.parts or ':' in member or '\\' in member: raise ValueError('Unsafe archive path')
                    data=z.read(member); spec=expected[arc['path']][member]
                    if len(data)!=spec['bytes'] or sha(data)!=spec['sha256']: raise ValueError('Result checksum mismatch: '+member)
                    if destination:
                        target=(destination/member).resolve()
                        if not target.is_relative_to(destination): raise ValueError('Path escapes output root')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists():
                            if sha(target.read_bytes())!=spec['sha256']: raise FileExistsError('Refusing to overwrite different data: '+str(target))
                        else: target.write_bytes(data)
                    count+=1
        print('Verified '+arc['path'],flush=True)
    print(f"PASS: {len(manifest['archives'])} archives; {count} per-file checks; snapshot {manifest['snapshot']}.")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--extract',type=Path,help='Restore paths beneath this GAMLET root; never overwrite differing files')
    p.add_argument('--verify-members',action='store_true')
    args=p.parse_args()
    manifest=args.manifest or ROOT/json.loads((ROOT/'results/LATEST.json').read_text())['manifest']
    run(manifest.resolve(),args.extract.resolve() if args.extract else None,args.verify_members)
