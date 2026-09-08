"""Freeze the named completed evidence; never follow an active campaign automatically.

Run once with --capture from the paper checkout. Subsequent table regeneration
uses only the committed snapshots. Standard library only; no experiment reruns.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / 'GAMLET'
TRUSS = CODE / 'experiments/gendesign_rank/results/overnight_20260906_v2'
STATS = CODE / 'experiments/network_statistics/results/interim_20260908_v1'
NETWORK = CODE / 'experiments/network_robustness/results/replication_20260907_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def capture():
    manifest = ROOT / 'evidence/source_manifest.json'
    if manifest.exists():
        raise SystemExit('Snapshot already exists; refusing to overwrite its provenance.')
    assert read(TRUSS / 'SUMMARY.json')['state'] == 'completed'
    for i in range(5):
        assert (TRUSS / f'main_{i:02d}/COMPLETED.json').exists()
        assert read(TRUSS / f'main_{i:02d}/verification.json')['passed']
        assert (TRUSS / f'mechanism_{i:02d}/COMPLETED.json').exists()
    stat = read(STATS / 'statistics.json')
    assert stat['primary_comparison_audited']
    assert stat['comparisons'][stat['primary']]['block_count'] == 2
    items = []
    def cp(src, dest, eid):
        target = ROOT / 'evidence' / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        items.append(dict(id=eid, source_path=str(src.relative_to(CODE)).replace('\\','/'),
                          snapshot=str(target.relative_to(ROOT)).replace('\\','/'),
                          sha256=digest(target), human_verification='pending'))
    for name in ['SUMMARY.json','REPORT.md','NIGHT_PROTOCOL.md','requirements-repo-lock.txt','source_manifest.json']:
        cp(TRUSS/name, 'truss/'+name, 'E-TRUSS-'+name)
    for i in range(5):
        for name in ['config.json','baseline_selection.json','verification.json','training_summary.csv','COMPLETED.json']:
            cp(TRUSS/f'main_{i:02d}'/name, f'truss/main_{i:02d}/'+name, f'E-TRUSS-{i}-{name}')
        for name in ['mechanism_summary.json','protocol.json','verification.json']:
            cp(TRUSS/f'mechanism_{i:02d}'/name, f'truss/mechanism_{i:02d}/'+name, f'E-MECH-{i}-{name}')
    for name in ['statistics.json','endpoint_snapshot.csv','task_effects.csv','input_manifest.json',
                 'primary_comparison_audit.json','assumption_diagnostics.json','PLAN.md','COMPLETED.json']:
        cp(STATS/name, 'network_interim/'+name, 'E-INTERIM-'+name)
    for name in ['campaign.json','PROTOCOL.md','requirements-lock.txt','source_hashes.json','source_graph_audit.json']:
        cp(NETWORK/name, 'network_protocol/'+name, 'E-NETWORK-'+name)
    for relative in ['experiments/gendesign_rank/model.py','experiments/gendesign_rank/truss.py',
                     'experiments/gendesign_rank/experiment.py','experiments/gendesign_rank/mechanism.py',
                     'experiments/gendesign_rank/nightly.py','experiments/network_statistics/analyze.py',
                     'gamlet/surrogate/encoders/simple_graph_encoder.py','gamlet/surrogate/encoders/gnn_layers.py']:
        cp(CODE/relative, 'code/'+relative, 'E-CODE-'+Path(relative).name)
    # Frozen network source, rather than copying mutable files from the live worker.
    for src in sorted((NETWORK/'source_snapshot').rglob('*.py')):
        relative = src.relative_to(NETWORK/'source_snapshot')
        cp(src, 'code/network_frozen/'+relative.as_posix(), 'E-NETCODE-'+relative.as_posix())
    archive = ROOT/'Paper_in_journal_Machine_Learning_and_Knowledge_Extraction.zip'
    assert digest(archive) == 'f1cf63d507dd6a22698037b759b0ec9ca2da3658f1ff5a335fda0741d76ca03e'
    manifest.write_text(json.dumps(dict(snapshot_date='2026-09-08', status='working_draft',
        numerical_verification='machine checks against named local outputs', human_verification='pending',
        network_analysis='two-block interim snapshot only; final five-block results TODO',
        legacy_archive_sha256=digest(archive), sources=items), indent=2), encoding='utf-8')
    print(f'Captured {len(items)} evidence files; archive hash unchanged.')

def tables():
    target=ROOT/'tables'
    target.mkdir(exist_ok=True)
    truss=read(ROOT/'evidence/truss/SUMMARY.json')
    lines=[r'\begin{table}[tb]',r'\centering\small',
        r'\caption{Completed truss experiment at 256 true evaluations: five offline-data blocks and 120 held-out tasks. Negative compliance changes favor RankNet. Intervals are exploratory hierarchical bootstrap intervals, not task-conditional network intervals.}',
        r'\label{tab:truss}',r'\begin{tabular}{lrr}\toprule',
        r'Comparator & $\Delta_C$ (\%) & Exploratory 95\% interval (\%)\\\midrule']
    for key,label in [('primary','Validation-selected GNN regression'),('mse','GNN MSE'),('log_mse','GNN log-MSE'),('ea','EA'),('random','Random search')]:
        v=truss['effects'][key]; lo,hi=v['exploratory_hierarchical_95_percent']
        lines.append(f"{label} & ${v['delta_percent']:.3f}$ & $[{lo:.3f}, {hi:.3f}]$"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (target/'truss.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    stats=read(ROOT/'evidence/network_interim/statistics.json')
    lines=[r'\begin{table}[tb]',r'\centering\small',
        r'\caption{Exploratory two-block network snapshot: 64 in-distribution source graphs. RankNet versus validation-selected GNN regression. Positive robustness changes favor RankNet. Holm correction is across 36 task-conditional sign tests; the unadjusted offline-block sign-test $p$ is 0.5 in every row.}',
        r'\label{tab:networkinterim}',r'\begin{tabular}{rlrrr}\toprule',
        r'$B$ & Scenario bank & $\Delta_R$ (\%) & Task wins & Holm $p$\\\midrule']
    for b in (96,192):
        for metric,label in [('best_r','Four search scenarios'),('fresh_r','32 fresh scenarios')]:
            # Resolve the frozen output schema explicitly, without choosing a result by its value.
            key=f'{b}/{metric}/selected_regression'
            if key not in stats['comparisons']:
                raise KeyError(f'{key}; available keys: {list(stats["comparisons"])}')
            v=stats['comparisons'][key]; test=v['conditional_sign_test']
            mantissa,exponent=f"{test['p_holm_36']:.3e}".split('e')
            p=mantissa+r'\times10^{'+str(int(exponent))+'}'
            lines.append(f"{b} & {label} & $+{v['gain_percent']:.3f}$ & {test['wins']}/{v['task_count']} & ${p}$"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (target/'network_interim.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Generated two result tables from frozen JSON evidence.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',action='store_true')
    args=parser.parse_args()
    if args.capture: capture()
    tables()
