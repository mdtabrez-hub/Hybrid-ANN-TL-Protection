"""Derived quantities quoted in the manuscript (from reimplementation results and dataset facts)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json
P=pd.read_csv('results/paired_comparisons.csv'); S=pd.read_csv('results/summary_by_condition.csv'); F=json.load(open('results/data_facts.json'))
EX=pd.read_csv('results/excluded_rows_predictions.csv.gz',dtype=str)
for c in ['label','pred']: EX[c]=EX[c].str.zfill(4)
EX['phys']=EX.phys.where(~EX.phys.str.fullmatch(r'\d+'),EX.phys.str.zfill(4))
VALID={'0000','1001','0110','1011','0111','1111'}
e=EX[(EX.data=='E1')&(EX.part=='tail')&(EX.regime=='partial_interruption')]; ok=e.phys.isin(VALID)
X={}
X['excl_acc_phys']=float((e[ok].pred==e[ok].phys).mean()); X['excl_acc_label']=float((e[ok].pred==e[ok].label).mean()); X['excl_rows_per_seed']=float(ok.sum()/5)
lll=e[e.label=='0111']; X['excl_LLL_to_LLG']=float((lll.pred=='1011').mean())
def seeds(name): return [float(x) for x in P[P.comparison==name].per_seed.iloc[0].split(';')]
r=sorted(seeds('random: robust - minmax')); big=[v for v in r if v>0.05]; small=[v for v in r if v<=0.05]
X['robust_gain_big_min']=min(big); X['robust_gain_big_max']=max(big); X['robust_gain_n_big']=len(big); X['robust_gain_small_maxabs']=max(abs(v) for v in small)
cl=P[P.comparison.str.contains('global clip - train clip')]; X['clip_effect_maxabs']=float(cl.mean_diff.abs().max())
g=S[(S.task=='class')&(S.data=='E0')&(S['clip']=='train')&(S.scaler=='minmax')&(S.part=='random')&(S.gap==0)&(S.shuffle==1)&(S.drop_v0==0)].iloc[0]
cm=np.array(json.loads(g.cm_sum)); X['llg_to_lg_random']=float(cm[3,1]/cm[3].sum())
X['band_fraction_minmax']=F['LLL_band_width']/F['I0_span']
gb=P[P.comparison.str.contains('guard100')]; X['guard_maxabs']=float(gb.mean_diff.abs().max())
json.dump(X,open('results/derived.json','w'),indent=1); print(json.dumps(X,indent=1))
