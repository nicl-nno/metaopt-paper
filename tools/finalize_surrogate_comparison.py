"""Publish the comparison only after all searches, audits and analyses finish.

Fails safely on a changed worktree, diverged remote, failed campaign, or
checksum mismatch. It never resets files, force-pushes, or edits the paper.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from export_surrogate_comparison import export, ROOT, read, dump


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args}:\n{result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def finalize(source, name, push):
    main = source / 'experiments/surrogate_comparison/results/comparison_20260908_v1'
    log = main / 'publication_status.json'
    while not (main / 'ANALYSIS_COMPLETED.json').exists():
        state = read(main / 'state.json')
        if state['failed']:
            raise RuntimeError('Campaign worker failure: ' + str(state['failed']))
        # No outcome-based stopping. This waits without consuming a CPU core.
        time.sleep(30)
    assert not command(['git', 'status', '--porcelain']), 'Publication worktree changed; review before exporting'
    assert command(['git', 'branch', '--show-current']) == 'main'
    command(['git', 'fetch', 'origin'])
    assert command(['git', 'rev-parse', 'HEAD']) == command(['git', 'rev-parse', 'origin/main']), 'Remote changed; review and integrate before publishing'
    export(source, name)
    manifest = read(ROOT / 'results' / name / 'MANIFEST.json')
    relative = f'results/{name}/reports/surrogate_comparison/comparison_20260908_v1'
    report = (main / 'REPORT.md').read_text(encoding='utf-8')
    (ROOT / 'SURROGATE_COMPARISON.md').write_text(
        '# Сравнение с другими суррогатами\n\n'
        '**Полная серия завершена и независимо проверена:** 6 272 новых поиска, '
        '1 216 512 онлайн-оценок, без новых offline-меток. Все четыре метода и четыре '
        'бенчмарка включены, включая отрицательные и незначимые результаты.\n\n'
        f'[Численные результаты]({relative}/statistics.json), '
        f'[аудит]({relative}/AUDIT.json), '
        f'[составной манифест](results/{name}/MANIFEST.json). '
        f'Текущий полный снимок: {len(manifest["files"]):,} исходных файлов в '
        f'{len(manifest["archives"])} архивах, включая архивы предыдущих серий.\n\n'
        '[Методы и первичные источники](code/GAMLET/experiments/surrogate_comparison/LITERATURE.md), '
        '[фиксированный протокол](code/GAMLET/experiments/surrogate_comparison/PROTOCOL.md), '
        '[воспроизведение](code/GAMLET/experiments/surrogate_comparison/REPRODUCIBILITY.md).\n\n'
        + report, encoding='utf-8')
    command([sys.executable, 'tools/research_artifacts.py', '--verify-members'])
    paths = ['SURROGATE_COMPARISON.md', 'code/SOURCE_MANIFEST.json',
             'code/GAMLET/experiments/surrogate_comparison', 'results/LATEST.json', f'results/{name}']
    command(['git', 'add', '-f', '--'] + paths)
    command([sys.executable, 'tools/check_publication.py'])
    staged = command(['git', 'diff', '--cached', '--name-only']).splitlines()
    assert all(any(p == base or p.startswith(base + '/') for base in paths) for p in staged), 'Unexpected staged files; review before committing'
    command(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'])
    command(['git', 'commit', '-m', 'Publish audited four-family surrogate comparison on all benchmarks'])
    commit = command(['git', 'rev-parse', 'HEAD'])
    if push:
        command(['git', 'push', 'origin', 'main'])
        assert command(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0] == commit
    dump(log, dict(phase='published' if push else 'committed', commit=commit, snapshot=name, updated=time.time()))
    print(json.dumps(read(log)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    assert Path(args.name).name == args.name and args.name not in ['.', '..']
    try:
        finalize(args.source.resolve(), args.name, args.push)
    except Exception as error:
        dump(args.source / 'experiments/surrogate_comparison/results/comparison_20260908_v1/publication_status.json',
             dict(phase='failed', error=str(error), updated=time.time()))
        raise
