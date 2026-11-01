import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json
S=pd.read_csv('results/summary_by_condition.csv'); D=pd.read_csv('results/all_runs.csv'); P=pd.read_csv('results/paired_comparisons.csv')
N=['NoFault','LG','LL','LLG','LLL','LLLG']
def lab(r):
    s=f"{r['data']} | {r['part']} | clip={r['clip']} | {r['scaler']}"
    if r['gap']: s+=f" | guard={int(r['gap'])}"
    if not r['shuffle']: s+=' | unshuffled-val'
    if r['drop_v0']: s+=' | 23 feats'
    return s
out=[]
cl=S[S.task=='class'].copy(); cl['pair_n']=cl['sup_LLL']+cl['sup_LLLG']; cl['pair_acc']=1-cl['pair_err_mean']/cl['pair_n']
out.append('## Table R1 — Six-class accuracy by condition (independent reimplementation; 5 seeds)\n')
out.append('| Condition | n_test | Accuracy mean ± s.d. (%) | 95 % t-CI (%) | Per-seed (%) | Merged LLL+LLLG (%) | LLL/LLLG pair-only acc. (%) | Non-pair errors (mean) |\n|---|---|---|---|---|---|---|---|')
for _,r in cl.iterrows():
    ps=', '.join(f'{100*float(x):.2f}' for x in str(r.seeds).split(';'))
    out.append(f"| {lab(r)} | {int(r.n_test)} | {100*r.acc_mean:.2f} ± {100*r.acc_sd:.2f} | {100*r.acc_ci_lo:.2f}–{100*r.acc_ci_hi:.2f} | {ps} | {100*r.merged_mean:.2f} | {100*r.pair_acc:.1f} | {r.nonpair_err_mean:.1f} |")
dt=S[S.task=='detect']
out.append('\n## Table R2 — Detection (detection file; 5 seeds)\n\n| Partition | n_test | Accuracy mean ± s.d. (%) | 95 % t-CI | Fault recall | Fault precision |\n|---|---|---|---|---|---|')
for _,r in dt.iterrows(): out.append(f"| {r.part} | {int(r.n_test)} | {100*r.acc_mean:.2f} ± {100*r.acc_sd:.2f} | {100*r.acc_ci_lo:.2f}–{100*r.acc_ci_hi:.2f} | {100*r.rec_fault_mean:.2f} | {100*r.prec_fault_mean:.2f} |")
out.append('\n## Table R3 — Class-wise precision / recall / F1 (mean over 5 seeds; recall range in brackets) and test support\n')
for cond in [('E0','random'),('E0','tail'),('E0','mid'),('E1','tail')]:
    r=cl[(cl.data==cond[0])&(cl.part==cond[1])&(cl['clip']=='train')&(cl.scaler=='minmax')&(cl.gap==0)&(cl.shuffle==1)&(cl.drop_v0==0)].iloc[0]
    out.append(f"\n**{cond[0]}, {cond[1]}** (train-fitted clip, min–max)\n\n| Class | Support | Precision | Recall [min–max] | F1 |\n|---|---|---|---|---|")
    for n in N: out.append(f"| {n} | {int(r[f'sup_{n}'])} | {100*r[f'prec_{n}_mean']:.1f} | {100*r[f'rec_{n}_mean']:.1f} [{100*r[f'rec_{n}_min']:.1f}–{100*r[f'rec_{n}_max']:.1f}] | {100*r[f'f1_{n}_mean']:.1f} |")
    cm=np.array(json.loads(r.cm_sum)); out.append('\nSummed confusion matrix (rows true, cols predicted; order '+', '.join(N)+'):\n\n```\n'+'\n'.join(' '.join(f'{v:5d}' for v in row) for row in cm)+'\n```')
out.append('\n## Table R4 — Paired comparisons (same seed; difference in accuracy, percentage points)\n\n| Comparison | Mean diff (pp) | s.d. | 95 % t-CI (pp) | Seeds negative / 5 | Per-seed (pp) |\n|---|---|---|---|---|---|')
for _,r in P.iterrows(): out.append(f"| {r.comparison} ({r.metric}) | {100*r.mean_diff:+.2f} | {100*r.sd:.2f} | {100*r.ci_lo:+.2f} to {100*r.ci_hi:+.2f} | {int(r.n_negative)} | {', '.join(f'{100*float(x):+.2f}' for x in r.per_seed.split(';'))} |")
open('results/RESULTS_TABLES.md','w').write('\n'.join(out)); print('\n'.join(out[:5]))
