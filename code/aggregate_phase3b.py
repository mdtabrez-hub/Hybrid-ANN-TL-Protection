"""Aggregate Phase 3b results -> results/phase3b_summary.json, phase3b tables (CSV) and figures 5-6."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 5.8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
R = ROOT+'/results/phase3b'; OUT = {}
def ms(x):
    x = np.asarray(x, float); n = len(x); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(.975, n - 1) * s / np.sqrt(n) if n > 1 else 0.0
    return dict(mean=float(x.mean()), sd=float(s), lo=float(x.mean() - h), hi=float(x.mean() + h), min=float(x.min()), max=float(x.max()), n=n)
# ---- B1
rows = []
for f in glob.glob(f'{R}/B1_*.json'):
    r = json.load(open(f))
    for m, v in r['res'].items(): rows.append(dict(model=m, data=r['data'], part=r['part'], seed=r['seed'], **v))
B1 = pd.DataFrame(rows); B1.to_csv(ROOT+'/results/phase3b_B1_runs.csv', index=False)
b1 = {}
for (m, d, p), g in B1.groupby(['model', 'data', 'part']):
    b1[f'{m}|{d}|{p}'] = {k: ms(g[k]) for k in ['acc', 'merged', 'pair_acc', 'nonpair_acc', 'nonpair_err']}
OUT['B1'] = b1
# paired tail-random and E1-E0 per model
pb = {}
for m in B1.model.unique():
    a = B1[(B1.model == m)].set_index(['data', 'part', 'seed']).acc
    pb[f'{m}|E0 tail-random'] = ms([a[('E0', 'tail', s)] - a[('E0', 'random', s)] for s in range(5)])
    pb[f'{m}|E0 mid-random'] = ms([a[('E0', 'mid', s)] - a[('E0', 'random', s)] for s in range(5)])
    pb[f'{m}|tail E1-E0'] = ms([a[('E1', 'tail', s)] - a[('E0', 'tail', s)] for s in range(5)])
OUT['B1_paired'] = pb
# ---- B2
rows = []
for f in glob.glob(f'{R}/B2_*.json'):
    r = json.load(open(f))
    for k, v in r['res'].items():
        m, kind, val = k.split('|'); rows.append(dict(model=m, kind=kind, val=float(val), seed=r['seed'], **v))
B2 = pd.DataFrame(rows)
if len(B2):
    B2.to_csv(ROOT+'/results/phase3b_B2_runs.csv', index=False)
    b2 = {}
    for (m, kind, val), g in B2.groupby(['model', 'kind', 'val']):
        b2[f'{m}|{kind}|{val:g}'] = {k: ms(g[k]) for k in ['acc', 'pair_acc', 'nonpair_acc', 'merged']}
    OUT['B2'] = b2
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True)
    sty = {'RF': ('#1f4e79', '-'), 'HGB': ('#548235', '-'), 'DT': ('#7f7f7f', '-'), 'kNN': ('#c55a11', '--'), 'LR': ('#bf9000', ':'), 'MLP-robust': ('#c00000', '-')}
    for kind, A, xl in [('delta', ax[0], 'Common-mode offset δ (A)'), ('sigma', ax[1], 'Noise σ per phase (A)')]:
        for m, (col, ls) in sty.items():
            g = B2[(B2.model == m) & (B2.kind == kind)].groupby('val')
            x = np.array(sorted(g.groups)); pa = g.pair_acc.mean().loc[x].values; na = g.nonpair_acc.mean().loc[x].values
            xx = x if kind == 'delta' else np.where(x == 0, 0.003, x)
            A.plot(xx, 100 * pa, color=col, ls=ls, lw=1, marker='o', ms=2, label=m)
            if m == 'RF': A.plot(xx, 100 * na, color='k', ls='-', lw=0.6, alpha=.5, label='other classes (RF)')
        if kind == 'sigma': A.set_xscale('log')
        A.axhline(50, color='0.6', lw=.5, ls=':'); A.set_xlabel(xl, fontsize=6.2); A.grid(alpha=.25)
    ax[0].set_ylabel('Accuracy (%)', fontsize=6.5); ax[0].set_title('(a) offset', fontsize=6.8); ax[1].set_title('(b) noise', fontsize=6.8)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=5.4, frameon=False, columnspacing=0.8, handlelength=1.8)
    plt.tight_layout(w_pad=0.3, rect=(0, 0.11, 1, 1)); plt.savefig(ROOT+'/figures/fig5_shortcut.png', dpi=300); plt.close()
# ---- B3
rows = []
for f in glob.glob(f'{R}/B3_*.json'):
    r = json.load(open(f))
    for m in ['RF', 'MLP']: rows.append(dict(model=m, frac=r['frac'], n_train=r['n_train'], seed=r['seed'], **r[m]))
B3 = pd.DataFrame(rows)
if len(B3):
    B3.to_csv(ROOT+'/results/phase3b_B3_runs.csv', index=False)
    OUT['B3'] = {f'{m}|{fr:g}': dict(n_train=int(g.n_train.mean()), **{k: ms(g[k]) for k in ['acc', 'pair_acc', 'nonpair_acc', 'merged']}) for (m, fr), g in B3.groupby(['model', 'frac'])}
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    for m, col in [('RF', '#1f4e79'), ('MLP', '#c00000')]:
        g = B3[B3.model == m].groupby('n_train')
        x = np.array(sorted(g.groups))
        for k, ls, lab in [('acc', '-', 'six-class'), ('pair_acc', ':', 'LLL vs LLLG only')]:
            mu = g[k].mean().loc[x].values; sd = g[k].std(ddof=1).loc[x].values
            ax.errorbar(x, 100 * mu, yerr=100 * sd, color=col, ls=ls, lw=.9, marker='o', ms=2, capsize=1.5, label=f'{m}, {lab}')
    ax.set_xscale('log'); ax.set_xlabel('Training rows (log scale)'); ax.set_ylabel('Accuracy (%)'); ax.grid(alpha=.25)
    ax.axhline(50, color='0.6', lw=.5, ls=':')
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=5.4, frameon=False)
    plt.tight_layout(rect=(0, 0.14, 1, 1)); plt.savefig(ROOT+'/figures/fig6_learning_curve.png', dpi=300); plt.close()
# ---- B4
rows = []
for f in glob.glob(f'{R}/B4_*.json'):
    r = json.load(open(f)); rows.append(r)
if rows:
    allerr = sum(r['n_err'] for r in rows); dist = np.concatenate([r['err_dist_ms'] for r in rows]) if allerr else np.array([])
    regs = pd.Series(sum([r['err_regime'] for r in rows], [])).value_counts().to_dict()
    OUT['B4'] = dict(acc=ms([r['acc'] for r in rows]), err_per_seed=ms([r['n_err'] for r in rows]), n_test=rows[0]['n_test'],
                     frac_within_1ms=float((dist <= 1.0).mean()) if allerr else None, frac_within_2ms=float((dist <= 2.0).mean()) if allerr else None,
                     median_dist_ms=float(np.median(dist)) if allerr else None, max_dist_ms=float(dist.max()) if allerr else None, regimes=regs, total_err=int(allerr))
# ---- derived quantities quoted in the manuscript (all computed here; none typed by hand)
P = pd.read_csv(ROOT+'/results/paired_comparisons.csv').set_index('comparison')
F = json.load(open(ROOT+'/results/data_facts.json'))
D = {}
tails = {m: pb[f'{m}|E0 tail-random']['mean'] for m in B1.model.unique()}; tails['MLP'] = float(P.loc['E0: tail - random', 'mean_diff'])
mids = {m: pb[f'{m}|E0 mid-random']['mean'] for m in B1.model.unique()}; mids['MLP'] = float(P.loc['E0: mid - random', 'mean_diff'])
gains = {m: pb[f'{m}|tail E1-E0']['mean'] for m in B1.model.unique()}; gains['MLP'] = float(P.loc['tail: E1 - E0', 'mean_diff'])
D['e1_gain_min'] = min(gains.values()); D['e1_gain_max'] = max(gains.values())
D['tail_drop_min'] = -max(tails.values()); D['tail_drop_max'] = -min(tails.values())   # magnitudes of the loss
D['mid_change_min'] = min(mids.values()); D['mid_change_max'] = max(mids.values())
D['mid_worst_loss'] = -min(mids.values())
a = B1.set_index(['model', 'data', 'part', 'seed']).acc
D['E1tail_minus_E0random_min'] = float(min(np.mean([a[(m, 'E1', 'tail', s)] - a[(m, 'E0', 'random', s)] for s in range(5)]) for m in B1.model.unique()))
D['tree_random_min'] = min(b1[f'{m}|E0|random']['acc']['mean'] for m in ['RF', 'HGB', 'DT'])
D['tree_random_max'] = max(b1[f'{m}|E0|random']['acc']['mean'] for m in ['RF', 'HGB', 'DT'])
D['tree_pair_min'] = min(b1[f'{m}|E0|random']['pair_acc']['mean'] for m in ['RF', 'HGB', 'DT'])
D['scaled_pair_max'] = max(b1[f'{m}|E0|random']['pair_acc']['mean'] for m in ['LR', 'kNN'])
if 'B2' in OUT:
    b2 = OUT['B2']
    D['tree_pair_at_pm07_max'] = max(b2[f'{m}|delta|{d:g}']['pair_acc']['mean'] for m in ['RF', 'HGB', 'DT'] for d in [-0.7, 0.7])
    D['tree_pair_at_pm07_min'] = min(b2[f'{m}|delta|{d:g}']['pair_acc']['mean'] for m in ['RF', 'HGB', 'DT'] for d in [-0.7, 0.7])
    D['tree_nonpair_at_pm07_min'] = min(b2[f'{m}|delta|{d:g}']['nonpair_acc']['mean'] for m in ['RF', 'HGB'] for d in [-0.7, 0.7])
    D['rf_nonpair_at_pm07_min'] = min(b2[f'RF|delta|{d:g}']['nonpair_acc']['mean'] for d in [-0.7, 0.7])
    D['rf_pair_at_pm07_max'] = max(b2[f'RF|delta|{d:g}']['pair_acc']['mean'] for d in [-0.7, 0.7])
    D['rf_pair_at_pm07_min'] = min(b2[f'RF|delta|{d:g}']['pair_acc']['mean'] for d in [-0.7, 0.7])
    D['scaled_pair_change_max'] = max(abs(v['pair_acc']['mean'] - b2[f'{k.split("|")[0]}|delta|0']['pair_acc']['mean']) for k, v in b2.items() if k.split('|')[0] in ('LR', 'kNN'))
    D['offset_rel_fault_peak'] = 0.7 / F['fault_peak_max_A']
    D['noise1_rel_fault_peak'] = 1.0 / F['fault_peak_max_A']
if 'B3' in OUT:
    b3 = OUT['B3']; mp = [v['pair_acc']['mean'] for k, v in b3.items() if k.startswith('MLP|')]
    D['mlp_pair_lc_min'] = min(mp); D['mlp_pair_lc_max'] = max(mp)
OUT['derived'] = D
import os
if os.path.exists(ROOT+'/results/phase3b_summary.json'):
    old = json.load(open(ROOT+'/results/phase3b_summary.json'))
    if 'B4_cat' in old: OUT['B4_cat'] = old['B4_cat']   # written by b4_localise.py (run after this script)
json.dump(OUT, open(ROOT+'/results/phase3b_summary.json', 'w'), indent=1)
print(json.dumps(D, indent=0))
print('B1 keys', len(OUT.get('B1', {})), '| B2', len(OUT.get('B2', {})), '| B3', len(OUT.get('B3', {})), '| B4', 'B4' in OUT)
