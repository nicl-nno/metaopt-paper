"""Read campaign progress without deriving partial scientific conclusions."""
import argparse
import json
from pathlib import Path
from .data import load


def status(root):
    root = Path(root)
    state = load(root / 'state.json')
    rows = []
    count = 0
    for folder in sorted(root.glob('*/block_*')):
        if not folder.is_dir():
            continue
        s = load(folder / 'state.json')
        runs = len(list((folder / 'runs').glob('*.json')))
        count += runs
        rows.append(dict(benchmark=folder.parent.name, block=folder.name, searches=runs,
                         audited=(folder / 'AUDIT.json').exists(), **s))
    result = dict(searches_written=count, expected_searches=6272,
                  completed_blocks=sum(r['phase'] == 'completed' for r in rows),
                  audited_blocks=sum(r['audited'] for r in rows), expected_blocks=16,
                  active=state['active'], pending_blocks=len(state['pending']),
                  failed=state['failed'], updated=state['updated'],
                  analysis_complete=(root / 'ANALYSIS_COMPLETED.json').exists(), blocks=rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    status(parser.parse_args().root)
