"""Addendum D1: composition of tail-hold-out errors by regime for every model family. Writes results/phase3d_D1.json and figures/fig_errcomp.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, numpy as np, pandas as pd
import run_phase3b as P3B
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3})
c, y = P3B.c, P3B.y; REG = ['sustained', 'clearing_sametype', 'partial_interruption', 'post_interruption', 'healthy']
out = {}
for m in ['LR', 'kNN', 'DT', 'RF', 'HGB']:
    cnt = {r: [] for r in REG}
    for seed in range(5):
        tr, te = P3B.split_idx('tail', seed, 'E0'); mk, scale = P3B.models(seed)[m]
        Xtr, Xte = P3B.X_raw[tr], P3B.X_raw[te]
        if scale: s = P3B.StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        p = mk().fit(Xtr, y[tr]).predict(Xte); wrong = p != y[te]; rg = c.regime.values[te]
        for r in REG: cnt[r].append(int((wrong & (rg == r)).sum()))
    out[m] = {r: float(np.mean(v)) for r, v in cnt.items()}
S = pd.read_csv(ROOT+'/results/summary_by_condition.csv', dtype={'clip': str})
g = S[(S.task == 'class') & (S.data == 'E0') & (S['clip'] == 'train') & (S.scaler == 'minmax') & (S.part == 'tail') & (S.gap == 0) & (S['shuffle'] == 1) & (S.drop_v0 == 0)].iloc[0]
out['Network'] = {r: float(g[f'nreg_{r}'] * (1 - g[f'accreg_{r}_mean'])) for r in REG}
for m, v in out.items():
    tot = sum(v.values()); v['total'] = tot; v['frac_inconsistent'] = (v['partial_interruption'] + v['post_interruption']) / tot
    v['n_inconsistent_err'] = v['partial_interruption'] + v['post_interruption']
fr = [v['frac_inconsistent'] for k, v in out.items()]
out['_derived'] = dict(frac_inconsistent_min=min(fr), frac_inconsistent_max=max(fr),
                       frac_trees_min=min(out[m]['frac_inconsistent'] for m in ['DT', 'RF', 'HGB']), frac_trees_max=max(out[m]['frac_inconsistent'] for m in ['DT', 'RF', 'HGB']),
                       incons_err_min=min(v['n_inconsistent_err'] for k, v in out.items()), incons_err_max=max(v['n_inconsistent_err'] for k, v in out.items()),
                       rows_inconsistent_in_tail=int(c.regime.isin(['partial_interruption', 'post_interruption']).sum()))
json.dump(out, open(ROOT+'/results/phase3d_D1.json', 'w'), indent=1)
lab = {'sustained': 'sustained', 'clearing_sametype': 'clearing, same type', 'partial_interruption': 'partial interruption', 'post_interruption': 'post-interruption', 'healthy': 'no fault'}
col = {'sustained': '#7f7f7f', 'clearing_sametype': '#bdd7ee', 'partial_interruption': '#c00000', 'post_interruption': '#f4b183', 'healthy': '#548235'}
names = ['LR', 'kNN', 'DT', 'RF', 'HGB', 'Network']; fig, ax = plt.subplots(figsize=(3.45, 1.9)); left = np.zeros(len(names))
for r in REG:
    v = np.array([out[n][r] for n in names]); ax.barh(np.arange(len(names)), v, left=left, color=col[r], label=lab[r], height=.6); left += v
ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(['Logistic regr.', 'k-NN', 'Decision tree', 'Random forest', 'Grad. boosting', 'Network']); ax.invert_yaxis()
ax.set_xlabel('Tail hold-out errors per run (mean over five seeds)'); ax.grid(axis='x', alpha=.25)
h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=5.4, frameon=False)
plt.tight_layout(rect=(0, 0.16, 1, 1)); plt.savefig(ROOT+'/figures/fig9_error_composition.png', dpi=300); plt.close()
print(json.dumps(out, indent=0))
