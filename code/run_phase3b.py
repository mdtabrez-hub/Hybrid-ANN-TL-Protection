"""Phase 3b extension analyses (protocol Addendum B). Independent implementation; not the author's pipeline.
Writes results/phase3b/*.json (one file per unit of work; resumable)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import run_experiments as RE          # data, regimes, features, partitions, Keras builder (module-level code loads data only)
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
OUT = ROOT+'/results/phase3b'; os.makedirs(OUT, exist_ok=True)
c, y, pos, RAW = RE.c, RE.y, RE.pos, RE.RAW
PAIR = [4, 5]
def metrics(p, yt):
    pair = np.isin(yt, PAIR); non = ~pair
    mp = np.where(p == 5, 4, p); mt = np.where(yt == 5, 4, yt)
    return dict(acc=float((p == yt).mean()), merged=float((mp == mt).mean()), pair_acc=float((p[pair] == yt[pair]).mean()),
                nonpair_acc=float((p[non] == yt[non]).mean()), nonpair_err=int(((p != yt) & ~(np.isin(yt, PAIR) & np.isin(p, PAIR))).sum()),
                n_test=int(len(yt)))
def models(seed):
    return {'LR': (lambda: LogisticRegression(max_iter=2000), True), 'kNN': (lambda: KNeighborsClassifier(5), True),
            'DT': (lambda: DecisionTreeClassifier(random_state=seed), False), 'RF': (lambda: RandomForestClassifier(200, random_state=seed, n_jobs=2), False),
            'HGB': (lambda: HistGradientBoostingClassifier(random_state=seed), False)}
def split_idx(part, seed, data):
    keep = np.ones(len(c), bool)
    if data == 'E1': keep = ~c.regime.isin(['partial_interruption', 'post_interruption']).values
    tr, te, _ = RE.split(part, seed, len(c), y, pos)
    return tr[keep[tr]], te[keep[te]]
X_raw = RE.feats(c[RAW])
def done(name): return os.path.exists(f'{OUT}/{name}.json')
def save(name, obj): json.dump(obj, open(f'{OUT}/{name}.json', 'w'))
# ---------------- B1 ----------------
def B1():
    for seed in range(5):
        for data in ['E0', 'E1']:
            for part in ['random', 'mid', 'tail']:
                name = f'B1_{data}_{part}_s{seed}'
                if done(name): continue
                tr, te = split_idx(part, seed, data); res = {}
                for m, (mk, scale) in models(seed).items():
                    Xtr, Xte = X_raw[tr], X_raw[te]
                    if scale: s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
                    clf = mk().fit(Xtr, y[tr]); res[m] = metrics(clf.predict(Xte), y[te])
                save(name, dict(seed=seed, data=data, part=part, res=res)); print(name, {k: round(v['acc'], 4) for k, v in res.items()}, flush=True)
# ---------------- B2 ----------------
DELTAS = [-1.0, -0.7, -0.35, 0.0, 0.35, 0.7, 1.0]; SIGMAS = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
def perturbed(te, kind, val, seed):
    R = c[RAW].iloc[te].copy()
    if kind == 'delta':
        for ph in ['Ia', 'Ib', 'Ic']: R[ph] = R[ph] + val
    else:
        rng = np.random.default_rng(1000 + seed)
        for ph in ['Ia', 'Ib', 'Ic']: R[ph] = R[ph] + rng.normal(0, val, len(R)) if val > 0 else R[ph]
    return RE.feats(R)
def B2():
    for seed in range(5):
        name = f'B2_s{seed}'
        if done(name): continue
        tr, te = split_idx('random', seed, 'E0'); out = {}
        fitted = {}
        for m, (mk, scale) in models(seed).items():
            s = StandardScaler().fit(X_raw[tr]) if scale else None
            clf = mk().fit(s.transform(X_raw[tr]) if scale else X_raw[tr], y[tr]); fitted[m] = (clf, s)
        # Keras MLP, robust scaling, no clipping (published architecture)
        sc = RobustScaler().fit(X_raw[tr]); trs = tr.copy(); np.random.default_rng(seed).shuffle(trs)
        net = RE.build(seed, 24, 6)
        cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)]
        net.fit(sc.transform(X_raw[trs]), y[trs], batch_size=32, epochs=100, validation_split=0.15, callbacks=cb, verbose=0)
        fitted['MLP-robust'] = ('keras', sc)
        for kind, vals in [('delta', DELTAS), ('sigma', SIGMAS)]:
            for v in vals:
                Xp = perturbed(te, kind, v, seed)
                for m, (clf, s) in fitted.items():
                    if clf == 'keras': p = net.predict(s.transform(Xp), verbose=0).argmax(1)
                    else: p = clf.predict(s.transform(Xp) if s is not None else Xp)
                    out[f'{m}|{kind}|{v}'] = metrics(p, y[te])
        save(name, dict(seed=seed, res=out)); print(name, 'done', flush=True)
# ---------------- B3 ----------------
FRACS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
def B3():
    for seed in range(5):
        tr0, te = split_idx('random', seed, 'E0')
        for f in FRACS:
            name = f'B3_f{f}_s{seed}'
            if done(name): continue
            if f < 1.0: tr, _ = train_test_split(tr0, train_size=f, stratify=y[tr0], random_state=seed)
            else: tr = tr0
            rf = RandomForestClassifier(200, random_state=seed, n_jobs=2).fit(X_raw[tr], y[tr]); r_rf = metrics(rf.predict(X_raw[te]), y[te])
            raw = c[RAW]; lo, hi = raw.iloc[tr].quantile(.01), raw.iloc[tr].quantile(.99); X = RE.feats(raw.clip(lo, hi, axis=1))
            s = MinMaxScaler().fit(X[tr]); trs = tr.copy(); np.random.default_rng(seed).shuffle(trs)
            net = RE.build(seed, 24, 6)
            cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)]
            vs = 0.15 if len(trs) >= 40 else 0.0
            net.fit(s.transform(X[trs]), y[trs], batch_size=32, epochs=100, validation_split=vs, callbacks=cb if vs else [], verbose=0)
            r_mlp = metrics(net.predict(s.transform(X[te]), verbose=0).argmax(1), y[te])
            save(name, dict(seed=seed, frac=f, n_train=int(len(tr)), RF=r_rf, MLP=r_mlp)); print(name, len(tr), round(r_rf['acc'], 4), round(r_mlp['acc'], 4), flush=True)
# ---------------- B4 ----------------
def B4():
    D = RE.d; yy = D['Output (S)'].values; reg = pd.read_csv(ROOT+'/results/detect_row_regimes.csv')['regime'].values
    trans = np.flatnonzero(np.diff(yy)) + 0.5
    for seed in range(5):
        name = f'B4_s{seed}'
        if done(name): continue
        tr, te = train_test_split(np.arange(len(D)), test_size=0.2, stratify=yy, random_state=seed)
        raw = D[RAW]; lo, hi = raw.iloc[tr].quantile(.01), raw.iloc[tr].quantile(.99); X = RE.feats(raw.clip(lo, hi, axis=1))
        s = MinMaxScaler().fit(X[tr]); trs = tr.copy(); np.random.default_rng(seed).shuffle(trs)
        net = RE.build(seed, 24, 1)
        cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)]
        net.fit(s.transform(X[trs]), yy[trs], batch_size=32, epochs=100, validation_split=0.15, callbacks=cb, verbose=0)
        p = (net.predict(s.transform(X[te]), verbose=0)[:, 0] > .5).astype(int); wrong = te[p != yy[te]]
        dist_ms = [float(np.min(np.abs(trans - i)) * 0.05) for i in wrong]
        save(name, dict(seed=seed, n_test=int(len(te)), n_err=int(len(wrong)), acc=float((p == yy[te]).mean()),
                         err_rows=[int(i) for i in wrong], err_dist_ms=dist_ms, err_regime=[str(reg[i]) for i in wrong], err_true=[int(yy[i]) for i in wrong]))
        print(name, len(wrong), 'errors', flush=True)
if __name__ == '__main__':
    which = sys.argv[1:] or ['B1', 'B2', 'B3', 'B4']
    for w in which: globals()[w]()
    print('ALL DONE', flush=True)
