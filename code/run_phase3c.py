"""Phase 3c analyses (protocol Addendum C). Independent implementation; not the author's pipeline.
Writes results/phase3c/*.json (one file per unit; resumable)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import run_experiments as RE
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
OUT = ROOT+'/results/phase3c'; os.makedirs(OUT, exist_ok=True)
c, y, pos, RAW, d = RE.c, RE.y, RE.pos, RE.RAW, RE.d
FEAT = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', '|Ia|', '|Ib|', '|Ic|', '|Va|', '|Vb|', '|Vc|', 'I_rms', 'V_rms', 'I_peak', 'V_peak', 'I0', 'V0',
        'Re I2', 'Im I2', 'Re V2', 'Im V2', 'std I', 'sum VI']
X_raw = RE.feats(c[RAW])
def done(n): return os.path.exists(f'{OUT}/{n}.json')
def save(n, o): json.dump(o, open(f'{OUT}/{n}.json', 'w'), default=float)
def metrics(p, yt):
    pair = np.isin(yt, [4, 5])
    return dict(acc=float((p == yt).mean()), pair_acc=float((p[pair] == yt[pair]).mean()), n_test=int(len(yt)))
# ---- regimes at other thresholds (same rule as regimes.py)
I = d[['Ia', 'Ib', 'Ic']].values; S = d['Output (S)'].values; Ts = 50e-6
EV = {'1001': (0.025, 0.075, 'A'), '1011': (0.125, 0.175, 'AB'), '0110': (0.225, 0.275, 'BC'), '0111': (0.325, 0.375, 'ABC'), '1111': (0.425, 0.475, 'ABC')}
runs = np.r_[0, np.flatnonzero(np.diff(S)) + 1, len(S)]
def regimes(thr):
    reg = np.array(['healthy'] * len(S), dtype=object); intB = {}
    for v, (ton, toff, ph) in EV.items():
        i_off = int(round(toff / Ts))
        st = [r for r in runs if r <= i_off and S[r] == 1 and (r == 0 or S[r - 1] == 0)][-1]; en = [r for r in runs if r > st][0] - 1
        def it(j):
            a = np.abs(I[:, j])
            for k in range(i_off, i_off + 600):
                if a[k:k + 200].max() < thr: return k
        ints = {p: it('ABC'.index(p)) for p in ph}; first, last = min(ints.values()), max(ints.values())
        reg[st:i_off] = 'sustained'; reg[i_off:first] = 'clearing_sametype'; reg[first:last] = 'partial_interruption'; reg[last:en + 1] = 'post_interruption'
        if v == '1111': intB = ints['B']
    r = reg[c.t_idx.values]; r[c.vec.values == '0000'] = 'healthy'
    return r, intB
def variant(name, thr=200):
    """returns (keep mask, labels)"""
    r, intB = regimes(thr); yy = y.copy()
    if name == 'E0': return np.ones(len(c), bool), yy
    if name == 'E1': return ~np.isin(r, ['partial_interruption', 'post_interruption']), yy
    if name == 'E3':
        keep = np.ones(len(c), bool); v = c.vec.values; part = r == 'partial_interruption'
        yy[part & (v == '1011')] = 1                                   # LLG -> A-G
        yy[part & (v == '1111') & (c.t_idx.values < intB)] = 3          # LLLG -> AB-G
        yy[part & (v == '1111') & (c.t_idx.values >= intB)] = 1         # then A-G
        keep[part & (v == '0111')] = False                             # LLL -> A-B: no class
        yy[r == 'post_interruption'] = 0                               # no fault current
        return keep, yy
def fit_eval(model, tr, te, yy, seed):
    if model == 'RF':
        clf = RandomForestClassifier(200, random_state=seed, n_jobs=2).fit(X_raw[tr], yy[tr]); return metrics(clf.predict(X_raw[te]), yy[te])
    raw = c[RAW]; lo, hi = raw.iloc[tr].quantile(.01), raw.iloc[tr].quantile(.99); X = RE.feats(raw.clip(lo, hi, axis=1))
    s = MinMaxScaler().fit(X[tr]); trs = tr.copy(); np.random.default_rng(seed).shuffle(trs)
    net = RE.build(seed, 24, 6)
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)]
    net.fit(s.transform(X[trs]), yy[trs], batch_size=32, epochs=100, validation_split=0.15, callbacks=cb, verbose=0)
    return metrics(net.predict(s.transform(X[te]), verbose=0).argmax(1), yy[te])
def C1():
    for seed in range(5):
        n = f'C1_s{seed}'
        if done(n): continue
        tr, te, _ = RE.split('random', seed, len(c), y, pos)
        rf = RandomForestClassifier(200, random_state=seed, n_jobs=2).fit(X_raw[tr], y[tr])
        a = permutation_importance(rf, X_raw[te], y[te], n_repeats=10, random_state=seed, n_jobs=1)
        pr = te[np.isin(y[te], [4, 5])]
        b = permutation_importance(rf, X_raw[pr], y[pr], n_repeats=10, random_state=seed, n_jobs=1)
        save(n, dict(seed=seed, features=FEAT, imp_all=a.importances_mean.tolist(), imp_pair=b.importances_mean.tolist(),
                     base_all=float(rf.score(X_raw[te], y[te])), base_pair=float(rf.score(X_raw[pr], y[pr])))); print(n, flush=True)
def C23():
    jobs = [('C2', thr, v, part) for thr in (150, 300) for v in ('E0', 'E1') for part in ('tail',)] + [('C3', 200, 'E3', part) for part in ('random', 'tail')]
    for seed in range(5):
        for tag, thr, v, part in jobs:
            if tag == 'C2' and v == 'E0' and thr == 300: continue   # E0 does not depend on the threshold
            n = f'{tag}_{v}_thr{thr}_{part}_s{seed}'
            if done(n): continue
            keep, yy = variant(v, thr)
            tr, te, _ = RE.split(part, seed, len(c), y, pos)
            tr, te = tr[keep[tr]], te[keep[te]]
            res = {m: fit_eval(m, tr, te, yy, seed) for m in ('RF', 'MLP')}
            save(n, dict(seed=seed, tag=tag, thr=thr, data=v, part=part, n_removed=int((~keep).sum()), res=res)); print(n, {m: round(r['acc'], 4) for m, r in res.items()}, flush=True)
if __name__ == '__main__':
    for w in (sys.argv[1:] or ['C1', 'C23']): globals()[w]()
    print('ALL DONE', flush=True)
