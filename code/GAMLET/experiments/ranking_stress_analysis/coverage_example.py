"""Fixed illustration: block0/test00, model17, search101; no best-case selection."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiments.ranking_stress.problems import CoverageDesign
ROOT=Path(__file__).resolve().parents[2]/'experiments/ranking_stress/results'
p=ROOT/'coverage_20260908_v1/block_00'
spec=json.loads((p/'tasks/test_00.json').read_text());problem=CoverageDesign(spec['seed'],spec['index'],spec['split'])
selected=json.loads((p/'selection.json').read_text())['selected_regression']
fig,axes=plt.subplots(1,3,figsize=(9,3.1),layout='constrained')
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
for ax,method,label,ms in zip(axes,['ea',selected,'ranknet'],['EA','Selected GNN regression','GNN RankNet'],[None,17,17]):
    row=json.loads((p/f'search/test_00_{method}_{ms}_101.json').read_text());x=np.array(row['design']);x[x==10.]=np.nan
    on=np.flatnonzero(np.isfinite(x));diff=np.arctan2(np.sin(problem.bearing[on]-x[on,None]),np.cos(problem.bearing[on]-x[on,None]))
    prob=.95*np.exp(-problem.distance[on]/.8)*problem.los[on]*(problem.distance[on]<=.55)*(abs(diff)<=math.pi/4)
    covered=1-np.prod(1-prob,axis=0)
    assert abs(problem.weights@covered-row['best'])<1e-12
    for r in problem.rects:ax.add_patch(Rectangle(r[:2],r[2]-r[0],r[3]-r[1],facecolor='.65'))
    ax.scatter(*problem.xy[:24].T,marker='+',color='.5',s=15)
    im=ax.scatter(*problem.xy[24:].T,s=12+650*problem.weights,c=covered,vmin=0,vmax=1,cmap='viridis',edgecolors='none')
    ax.scatter(*problem.xy[on].T,marker='s',s=20,color='orangered')
    ax.quiver(*problem.xy[on].T,np.cos(x[on])*.12,np.sin(x[on])*.12,angles='xy',scale_units='xy',scale=1,color='orangered',width=.006)
    ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',title=f'{label}\nWeighted coverage={row["best"]:.3f}');ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axes,shrink=.7,label='Target detection probability')
out=ROOT/'analysis_20260908_v1';out.mkdir(exist_ok=True)
fig.savefig(out/'coverage_example.pdf');fig.savefig(out/'coverage_example.png',dpi=180)
(out/'coverage_example.md').write_text('Fixed illustrative task: block0/test00, model17 and search101, selected before comparing visual outcomes. Each method used96 true evaluations. Orange squares/arrows: six sensors and orientations; gray crosses: unused sites; rectangles: opaque obstacles; demand marker areas reflect demand weights. Color shows combined detection probability, including occlusion. This illustration is not a separate replicate or a best-case selection.\n',encoding='utf-8')
