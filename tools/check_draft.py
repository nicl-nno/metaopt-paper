"""Offline integrity and numerical-consistency checks for this named draft.

This is not a test of the scientific hypothesis or a substitute for author review.
"""
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise AssertionError(message)

manifest=read(ROOT/'evidence/source_manifest.json')
for item in manifest['sources']:
    require(sha(ROOT/item['snapshot'])==item['sha256'], 'Changed evidence: '+item['snapshot'])
require(sha(ROOT/'Paper_in_journal_Machine_Learning_and_Knowledge_Extraction.zip')==manifest['legacy_archive_sha256'], 'Legacy archive changed')
require(manifest['human_verification']=='pending','Do not infer human approval from machine checks')
before={p:p.read_bytes() for p in (ROOT/'tables').glob('*.tex')}
subprocess.run([sys.executable,str(ROOT/'tools/prepare_evidence.py')],check=True)
require(all(p.read_bytes()==data for p,data in before.items()),'Generated tables were stale; inspect regenerated values')
text=(ROOT/'main.tex').read_text(encoding='utf-8')
refs=(ROOT/'references.bib').read_text(encoding='utf-8')
keys=re.findall(r'@\w+\{([^,]+),',refs)
require(len(keys)==len(set(keys)),'Duplicate bibliography key')
cited={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for k in group.split(',')}
require(cited<=set(keys),'Undefined citation keys: '+str(cited-set(keys)))
for path in re.findall(r'\\input\{([^}]+)\}',text): require((ROOT/path).exists(),'Missing input '+path)
for old in ['41 minutes','DirectRanker','SetRank','OpenML','FEDOT','classification accuracy']:
    require(old not in text,'Old empirical content remains: '+old)
for tag in ['N1','N2','N3','N4','A1','A2','A3','R1']:
    require('\\todo{'+tag+'}' in text,'Missing unfinished-work marker '+tag)

truss=read(ROOT/'evidence/truss/SUMMARY.json')
primary=truss['effects']['primary']
require(primary['held_out_tasks']==120 and primary['training_data_replicates']==5,'Wrong completed truss population')
require(f"{abs(primary['delta_percent']):.2f}"=='0.42','Truss abstract no longer matches')
require(f"{abs(truss['effects']['ea']['delta_percent']):.2f}"=='12.80','EA abstract no longer matches')
total_searches=0
for i in range(5):
    base=ROOT/f'evidence/truss/main_{i:02d}'
    audit=read(base/'verification.json'); config=read(base/'config.json')
    total_searches+=audit['searches_verified']
    require(audit['passed'] and audit['exact_evaluation_budgets'],'Truss audit not passed')
    require(config['budget']==256 and config['epochs']==80,'Protocol mismatch')
    rows=list(csv.DictReader((base/'training_summary.csv').open(encoding='utf-8-sig')))
    for seed in config['model_seeds']:
        matched=[r for r in rows if int(r['model_seed'])==seed]
        require(len(matched)==3,'Incomplete matched training set')
        for field in ['initial_weights_sha256','optimizer_steps','trainable_parameters']:
            require(len({r[field] for r in matched})==1,'Training not matched: '+field)
        require(int(matched[0]['trainable_parameters'])==8769,'Truss parameter count mismatch')
require(total_searches==20400 and total_searches*256==5222400,'Search count mismatch')

mechanisms=[read(ROOT/f'evidence/truss/mechanism_{i:02d}/mechanism_summary.json') for i in range(5)]
for comparator,expected in [('pool_random',-13.61),('untrained',-21.07),('shuffled',-15.32)]:
    key='in_distribution/pool48/ranknet_vs_'+comparator
    actual=100*math.expm1(mean(mean(m['effects'][key]['task_log_ratios'].values()) for m in mechanisms))
    require(round(actual,2)==expected,'Mechanism effect mismatch: '+comparator)
require(all(m['effects']['in_distribution/pool4/ranknet_vs_ea']['delta_percent']==0 for m in mechanisms),'Pool4 equivalence mismatch')
for stage,label in [('initial','Initial'),('near_ea_solution','Near EA solution')]:
    for loss,name in [('mse','MSE'),('log_mse','log-MSE'),('ranknet','RankNet')]:
        v={k:mean(m['common_pools'][stage+'/'+loss][k] for m in mechanisms) for k in
           ['pair_accuracy','precision_at_3','selected_mean_vs_random_percent']}
        row=f"{label} & {name} & {v['pair_accuracy']:.3f} & {v['precision_at_3']:.3f} & ${v['selected_mean_vs_random_percent']:.2f}$"
        require(row in text,'Common-pool table mismatch: '+stage+'/'+loss)

stat=read(ROOT/'evidence/network_interim/statistics.json')
v=stat['comparisons'][stat['primary']]
require(v['block_count']==2 and v['task_count']==64,'Do not silently replace interim scope')
require(v['conditional_sign_test']['wins']==55,'Network wins mismatch')
require(f"{v['gain_percent']:.3f}"=='1.167','Network text mismatch')
require(v['offline_block_tests']['sign_test']['p_two_sided']==0.5,'Block inference mismatch')
require(len(stat['comparisons'])==36,'Holm comparison family changed')
require(stat['primary_comparison_audited'],'Missing interim primary audit')
log=ROOT/'main.log'
if log.exists():
    logtext=log.read_text(encoding='utf-8',errors='replace')
    for bad in ['There were undefined references','There were undefined citations','LaTeX Error','Overfull \\hbox']:
        require(bad not in logtext,'Build issue: '+bad)
print(f'PASS: {len(manifest["sources"])} snapshot hashes, unchanged archive, {len(cited)} citations, matched training, numerical tables/counts and explicit TODO/interim scope.')
print('Human scientific verification and final author approval remain pending.')
