"""Aggregate Phase 3c -> results/phase3c_summary.json and figures/fig7_importance.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3})
R = ROOT+'/results'; OUT = {}
def ms(x):
    x = np.asarray(x, float); n = len(x); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(.975, n - 1) * s / np.sqrt(n) if n > 1 else 0.0
    return dict(mean=float(x.mean()), sd=float(s), lo=float(x.mean() - h), hi=float(x.mean() + h), min=float(x.min()), max=float(x.max()), n=n)
# C1
C1 = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/phase3c/C1_*.json'))]
F = C1[0]['features']; A = np.array([r['imp_all'] for r in C1]); Bp = np.array([r['imp_pair'] for r in C1])
OUT['C1'] = {f: dict(pair=ms(Bp[:, i]), all=ms(A[:, i])) for i, f in enumerate(F)}
others = [f for f in F if f != 'I0']
OUT['C1_derived'] = dict(I0_pair=float(Bp[:, F.index('I0')].mean()), I0_all=float(A[:, F.index('I0')].mean()),
                         max_other_pair=float(max(Bp[:, F.index(f)].mean() for f in others)),
                         max_other_all=float(max(A[:, F.index(f)].mean() for f in others)),
                         base_pair=ms([r['base_pair'] for r in C1]), base_all=ms([r['base_all'] for r in C1]))
o = np.argsort(-(Bp.mean(0) + A.mean(0)))[:8][::-1]
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
for a_, M, t in [(ax[0], Bp, '(a) LLL/LLLG rows'), (ax[1], A, '(b) all test rows')]:
    a_.barh(np.arange(len(o)), 100 * M.mean(0)[o], xerr=100 * M.std(0, ddof=1)[o], color=['#c00000' if F[i] == 'I0' else '#1f4e79' for i in o], height=.6, error_kw=dict(lw=.6, capsize=1.2))
    a_.set_title(t, fontsize=6.8); a_.set_xlabel('Accuracy drop (points)', fontsize=6.3); a_.grid(axis='x', alpha=.25)
ax[0].set_yticks(np.arange(len(o))); ax[0].set_yticklabels([F[i].replace('I0', 'I₀ (residual)').replace('I2', 'I₂').replace('V2', 'V₂').replace('_rms', ' rms').replace('_peak', ' peak') for i in o])
plt.tight_layout(w_pad=.4); plt.savefig(ROOT+'/figures/fig7_importance.png', dpi=300); plt.close()
# C2 / C3
rows = []
for f in glob.glob(f'{R}/phase3c/C[23]_*.json'):
    r = json.load(open(f))
    for m, v in r['res'].items(): rows.append(dict(tag=r['tag'], thr=r['thr'], data=r['data'], part=r['part'], seed=r['seed'], model=m, n_removed=r['n_removed'], **v))
C = pd.DataFrame(rows)
if len(C):
    C.to_csv(f'{R}/phase3c_C23_runs.csv', index=False)
    OUT['C23'] = {f'{m}|{d}|{t}|{p}': dict(acc=ms(g.acc), pair_acc=ms(g.pair_acc), n_removed=int(g.n_removed.iloc[0]), n_seeds=int(len(g)))
                  for (m, d, t, p), g in C.groupby(['model', 'data', 'thr', 'part'])}
if len(C):
    c23 = OUT['C23']; B = json.load(open(f'{R}/phase3b_summary.json')); Sm = pd.read_csv(f'{R}/summary_by_condition.csv', dtype={'clip': str})
    base = Sm[(Sm.task == 'class') & (Sm['clip'] == 'train') & (Sm.scaler == 'minmax') & (Sm.gap == 0) & (Sm['shuffle'] == 1) & (Sm.drop_v0 == 0)].set_index(['data', 'part'])
    e1_200 = {'RF': B['B1']['RF|E1|tail']['acc']['mean'], 'MLP': float(base.loc[('E1', 'tail'), 'acc_mean'])}
    e0 = {m: c23[f'{m}|E0|150|tail']['acc']['mean'] for m in ('RF', 'MLP')}
    G = {}
    for m in ('RF', 'MLP'):
        g = {150: c23[f'{m}|E1|150|tail']['acc']['mean'] - e0[m], 200: e1_200[m] - e0[m], 300: c23[f'{m}|E1|300|tail']['acc']['mean'] - e0[m]}
        G[m] = dict(gain_150=g[150], gain_200=g[200], gain_300=g[300], gain_spread=max(g.values()) - min(g.values()), E0_tail=e0[m], E1_tail_200=e1_200[m],
                    E3_tail=c23[f'{m}|E3|200|tail']['acc']['mean'], E3_random=c23[f'{m}|E3|200|random']['acc']['mean'],
                    E3_recovered_frac=(c23[f'{m}|E3|200|tail']['acc']['mean'] - e0[m]) / (e1_200[m] - e0[m]))
    G['max_gain_spread'] = max(G['RF']['gain_spread'], G['MLP']['gain_spread'])
    OUT['C23_derived'] = G
import os
if os.path.exists(f'{R}/phase3c_E3_diagnostic.json'): OUT['E3_diag'] = json.load(open(f'{R}/phase3c_E3_diagnostic.json'))
json.dump(OUT, open(f'{R}/phase3c_summary.json', 'w'), indent=1)
print(json.dumps(OUT.get('C23_derived'), indent=0))
print(json.dumps(OUT['C1_derived'], indent=0)); print('C23 keys', sorted(OUT.get('C23', {})))
