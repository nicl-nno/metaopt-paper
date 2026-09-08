"""Export a completed, audited surrogate comparison without duplicating old data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def export(source, name):
    destination = ROOT / 'results' / name
    assert not destination.exists(), 'Snapshots are immutable; choose a new name'
    area = source / 'experiments/surrogate_comparison'
    main = area / 'results/comparison_20260908_v1'
    complete, audit = read(main / 'COMPLETED.json'), read(main / 'AUDIT.json')
    assert complete['searches'] == audit['searches'] == 6272
    assert complete['online_evaluations'] == 1216512 and audit['passed']
    assert read(main / 'ANALYSIS_COMPLETED.json')['primary_comparisons'] == 16
    latest = read(ROOT / 'results/LATEST.json')
    previous = read(ROOT / latest['manifest'])
    manifest = {**previous, 'snapshot': name, 'parent_manifest': latest['manifest'],
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Complete prior benchmarks plus the audited four-family surrogate comparison. All positive, null and negative outcomes retained.',
                'archives': list(previous['archives']), 'files': list(previous['files']), 'runs': dict(previous['runs'])}
    existing = {r['member'] for r in manifest['files']}
    reports = {'REPORT.md', 'statistics.json', 'COMPLETED.json', 'AUDIT.json', 'CAMPAIGN.json',
               'ANALYSIS_COMPLETED.json', 'state.json', 'selection.json', 'config.json',
               'input_hashes.json', 'source_hashes.json', 'analysis_input_hashes.json'}
    for run in sorted((area / 'results').iterdir()):
        if not run.is_dir():
            continue
        groups = {}
        for path in sorted(run.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ['.pyc', '.tmp']:
                continue
            rel = path.relative_to(run)
            unit = '_'.join(rel.parts[:2]) if len(rel.parts) > 2 else 'metadata'
            groups.setdefault(unit, []).append(path)
        manifest['runs']['surrogate_comparison/' + run.name] = dict(
            files_at_inventory=sum(map(len, groups.values())),
            root_completed_marker_at_inventory=(run / 'COMPLETED.json').exists())
        for unit, paths in groups.items():
            # Bound raw bytes per ZIP; individual selected models must also fit.
            batches, current, size = [], [], 0
            for path in paths:
                n = path.stat().st_size
                assert n < 40 * 1024 * 1024, f'Split oversized artifact explicitly: {path}'
                if current and size + n > 40 * 1024 * 1024:
                    batches.append(current)
                    current, size = [], 0
                current.append(path)
                size += n
            if current:
                batches.append(current)
            for index, batch in enumerate(batches):
                archive = destination / 'archives/surrogate_comparison' / run.name / f'{unit}_{index:02d}.zip'
                archive.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                    for path in batch:
                        data = path.read_bytes()
                        member = path.relative_to(source).as_posix()
                        assert member not in existing, f'Duplicate original path: {member}'
                        existing.add(member)
                        z.writestr(member, data)
                        manifest['files'].append(dict(archive=archive.relative_to(ROOT).as_posix(), member=member, sha256=sha(data), bytes=len(data)))
                        if (path.name in reports or path.name.startswith('replay_')) and 'source_snapshot' not in path.parts:
                            target = destination / 'reports/surrogate_comparison' / run.name / path.relative_to(run)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(data)
                data = archive.read_bytes()
                assert len(data) < 48 * 1024 * 1024
                manifest['archives'].append(dict(path=archive.relative_to(ROOT).as_posix(), sha256=sha(data), bytes=len(data), members=len(batch)))
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
        records[rel] = dict(path=rel, sha256=sha(data), bytes=len(data), origin='four-family frozen surrogate comparison')
    code['source_files'] = list(records.values())
    code['captured_utc'] = datetime.now(timezone.utc).isoformat()
    dump(ROOT / 'code/SOURCE_MANIFEST.json', code)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['raw_bytes'] = sum(r['bytes'] for r in manifest['files'])
    manifest['archive_bytes'] = sum(r['bytes'] for r in manifest['archives'])
    dump(destination / 'MANIFEST.json', manifest)
    dump(ROOT / 'results/LATEST.json', dict(snapshot=name, manifest=(destination / 'MANIFEST.json').relative_to(ROOT).as_posix()))
    (destination / 'README.md').write_text(
        f'# {name}\n\nCompleted and independently audited comparison: 6,272 new searches, '
        '1,216,512 online evaluations, no new offline labels. Four methods on all four original '
        'benchmarks, paired with the identical archived RankNet seed slice.\n\n'
        f'The combined manifest references {len(manifest["files"])} original files in '
        f'{len(manifest["archives"])} archives, including unchanged archives from preceding snapshots. '
        'Use tools/research_artifacts.py to verify or restore the complete dataset.\n', encoding='utf-8')
    print(json.dumps(dict(files=len(manifest['files']), archives=len(manifest['archives']), source_files=len(code['source_files'])), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert Path(args.name).name == args.name and args.name not in ['.', '..']
    export(args.source.resolve(), args.name)
