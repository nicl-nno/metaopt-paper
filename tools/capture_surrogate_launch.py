"""Capture frozen source and technical checks while the main campaign runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from export_surrogate_comparison import ROOT, read, dump


def capture(source):
    area = source / 'experiments/surrogate_comparison'
    code = read(ROOT / 'code/SOURCE_MANIFEST.json')
    records = {r['path']: r for r in code['source_files']}
    for path in area.iterdir():
        if not path.is_file() or path.suffix not in ['.py', '.md', '.json', '.txt']:
            continue
        target = ROOT / 'code/GAMLET' / path.relative_to(source)
        data = path.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rel = target.relative_to(ROOT).as_posix()
        records[rel] = dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), origin='four-family frozen surrogate comparison')
    code.update(source_files=list(records.values()), captured_utc=datetime.now(timezone.utc).isoformat())
    dump(ROOT / 'code/SOURCE_MANIFEST.json', code)
    smoke = read(area / 'results/smoke_v1/AUDIT.json')
    assert smoke['passed'] and smoke['searches'] == 16
    preflight = {kind: read(area / f'results/preflight_v1/replay_{kind}.json') for kind in ['truss', 'network', 'coverage', 'cascade']}
    assert all(r['passed'] for r in preflight.values())
    dump(ROOT / 'surrogate_comparison_launch/checks.json', dict(smoke=smoke, ranknet_replay=preflight,
         note='Technical checks only; no completed main-series scientific result is claimed.'))
    main = area / 'results/comparison_20260908_v1'
    dump(ROOT / 'surrogate_comparison_launch/campaign.json', read(main / 'CAMPAIGN.json'))
    dump(ROOT / 'surrogate_comparison_launch/state_at_capture.json', read(main / 'state.json'))
    print(json.dumps(dict(source_files=len(records), smoke_searches=16, exact_replays=4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    capture(parser.parse_args().source.resolve())
