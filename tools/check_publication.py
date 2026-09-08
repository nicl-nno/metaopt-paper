"""Check exact source/archive bytes staged for Git publication, not just disk copies."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sources=json.loads((ROOT/'code/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
latest=json.loads((ROOT/'results/LATEST.json').read_text(encoding='utf-8'))
results=json.loads((ROOT/latest['manifest']).read_text(encoding='utf-8'))
expected=[(r['path'],r['sha256']) for r in sources['source_files']+results['archives']]
process=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
try:
    for path,digest in expected:
        process.stdin.write((':'+path+'\n').encode()); process.stdin.flush()
        header=process.stdout.readline().decode().strip().split()
        if len(header)!=3 or header[1]!='blob': raise RuntimeError('Missing staged blob: '+path)
        size=int(header[2]); data=process.stdout.read(size)
        if process.stdout.read(1)!=b'\n': raise RuntimeError('Invalid git batch framing')
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=digest:
            raise RuntimeError('Staged byte checksum mismatch: '+path)
    process.stdin.close()
    if process.wait()!=0: raise RuntimeError('git cat-file failed')
finally:
    if process.poll() is None: process.terminate()
print(f"PASS: staged bytes match all {len(sources['source_files'])} source files and {len(results['archives'])} result archives.")
