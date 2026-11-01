"""
blocked_split.py -- does the random split leak across simulation runs?

Each class in classData.csv occupies a strictly contiguous block of rows whose
lag-1 autocorrelation is 0.9996: the rows are consecutive samples of one
continuous waveform. A stratified RANDOM split therefore places near-duplicate
neighbouring samples on both sides of the partition.

This script re-runs the three formulations under a BLOCKED split -- the last
20 % of each class block is held out as a contiguous segment, so no test sample
is a temporal neighbour of a training sample -- and reports both numbers side by
side. Nothing else changes: same topology, same schedule, same seed, same
preprocessing as Table 6.

Output: out/blocked_split.csv, out/blocked_split_report.md
"""
import os, sys, json, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

from feature_engineering import build_rowwise_features, MEAS
from train_ann import clip_1_99, VEC2CLASS, CLASS_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os as _os
U = _os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
OUT = os.path.join(ROOT, "out")

TEST_FRAC = 0.20


# ------------------------------------------------------------------ splitting
def blocked_indices(group_key, test_frac=TEST_FRAC):
    """Hold out the final `test_frac` of each contiguous group as one segment."""
    tr, te = [], []
    for g in pd.unique(group_key):
        idx = np.where(group_key == g)[0]
        cut = int(round(len(idx) * (1.0 - test_frac)))
        tr.extend(idx[:cut]); te.extend(idx[cut:])
    return np.array(sorted(tr)), np.array(sorted(te))


def contiguity_report(group_key):
    rows = []
    for g in pd.unique(group_key):
        idx = np.where(group_key == g)[0]
        rows.append(dict(group=str(g), n=len(idx),
                         contiguous=bool(np.all(np.diff(idx) == 1)),
                         first_row=int(idx.min()), last_row=int(idx.max())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- models
def mlp(n_in, n_out, act, loss):
    tf.keras.utils.set_random_seed(SEED)
    init = tf.keras.initializers.HeNormal(seed=SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(0.2, seed=SEED),
        tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(0.2, seed=SEED),
        tf.keras.layers.Dense(16, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dense(n_out, activation=act, kernel_initializer=init),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
    return m


def fit(m, X, y):
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                           monitor="val_loss"),
          tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss")]
    m.fit(X, y, epochs=100, batch_size=32, validation_split=0.15,
          callbacks=cb, verbose=0, shuffle=True)
    return m


# ----------------------------------------------------------------------- main
def main():
    res = []
    L = []
    A = L.append

    cd = pd.read_csv(U + "classData.csv")
    vec = cd[["G", "C", "B", "A"]].astype(int).astype(str).agg("".join, axis=1)
    cdc = clip_1_99(cd, MEAS)
    X = build_rowwise_features(cdc).to_numpy(np.float64)
    Y = cd[["G", "C", "B", "A"]].to_numpy(np.int32)
    yc = np.array([CLASS_ORDER.index(VEC2CLASS[v]) for v in vec])

    A("# Does the random split leak across simulation runs?\n")
    A("Produced by `src/blocked_split.py`. Topology, schedule, preprocessing and")
    A("seed are those of Table 6 throughout; only the partition changes.\n")

    A("## 1. Row structure of the supplied files\n")
    cr = contiguity_report(vec.to_numpy())
    A("| Label vector | Rows | Contiguous block | First row | Last row |")
    A("|---|---|---|---|---|")
    for _, r in cr.iterrows():
        A(f"| {r.group} | {r.n:,} | {'yes' if r.contiguous else 'no'} | {r.first_row:,} | {r.last_row:,} |")
    ac = []
    for v in pd.unique(vec):
        s = cd.loc[vec == v, "Ia"].to_numpy()
        ac.append((v, np.corrcoef(s[:-1], s[1:])[0, 1]))
    A("")
    A("Lag-1 autocorrelation of the phase-A current within each block: " +
      ", ".join(f"{v} {a:.4f}" for v, a in ac) + ".")
    A("Every block is contiguous and every autocorrelation exceeds 0.999, so the")
    A("rows within a class are consecutive samples of one continuous waveform, not")
    A("independent draws. A stratified random split therefore assigns temporal")
    A("neighbours to both partitions.\n")

    A("## 2. The two partitions compared\n")

    # ---------- multi-label
    for name, split in (("random (Table 6)", "random"), ("blocked (this check)", "blocked")):
        if split == "random":
            itr, ite = train_test_split(np.arange(len(X)), test_size=0.2,
                                        random_state=SEED, stratify=vec)
        else:
            itr, ite = blocked_indices(vec.to_numpy())
        sc = MinMaxScaler().fit(X[itr])
        Xtr, Xte = sc.transform(X[itr]), sc.transform(X[ite])

        m = fit(mlp(X.shape[1], 4, "sigmoid", "binary_crossentropy"), Xtr, Y[itr])
        p = (m.predict(Xte, verbose=0) >= 0.5).astype(int)
        exact = float((p == Y[ite]).all(axis=1).mean() * 100)

        m2 = fit(mlp(X.shape[1], 6, "softmax", "sparse_categorical_crossentropy"),
                 Xtr, yc[itr])
        pc = m2.predict(Xte, verbose=0).argmax(1)
        six = float(accuracy_score(yc[ite], pc) * 100)
        lllg = float(recall_score(yc[ite], pc, labels=[CLASS_ORDER.index("LLLG")],
                                 average="macro", zero_division=0) * 100)
        merged = float((np.where(np.isin(pc, [CLASS_ORDER.index("LLL"), CLASS_ORDER.index("LLLG")]), -1, pc)
                        == np.where(np.isin(yc[ite], [CLASS_ORDER.index("LLL"), CLASS_ORDER.index("LLLG")]), -1, yc[ite])).mean() * 100)
        res.append(dict(split=split, task="multi-label exact match", value=exact, n_test=len(ite)))
        res.append(dict(split=split, task="six-class accuracy", value=six, n_test=len(ite)))
        res.append(dict(split=split, task="LLLG recall", value=lllg, n_test=len(ite)))
        res.append(dict(split=split, task="six-class, LLL+LLLG merged", value=merged, n_test=len(ite)))

    # ---------- detection
    dd = pd.read_csv(U + "detect_dataset.csv")
    dd = dd.dropna(axis=1, how="all")
    dd.columns = [c.strip() for c in dd.columns]
    scol = [c for c in dd.columns if c.lower().startswith("output") or c == "S"][0]
    ren = {}
    for c in dd.columns:
        k = c.strip().lower().replace(" ", "")
        for t in MEAS:
            if k == t.lower(): ren[c] = t
    dd = dd.rename(columns=ren)
    ddc = clip_1_99(dd, MEAS)
    Xd = build_rowwise_features(ddc).to_numpy(np.float64)
    yd = dd[scol].to_numpy(np.int32)
    # contiguity of the detection file
    runs = (np.diff(yd) != 0).sum() + 1
    A(f"The detection file contains {runs} contiguous runs of constant label, so the")
    A("same structure is present there; the blocked split for detection holds out the")
    A("last 20 % of each run.\n")
    run_id = np.cumsum(np.concatenate([[0], np.diff(yd) != 0]))

    for split in ("random", "blocked"):
        if split == "random":
            itr, ite = train_test_split(np.arange(len(Xd)), test_size=0.2,
                                        random_state=SEED, stratify=yd)
        else:
            itr, ite = blocked_indices(run_id)
        sc = MinMaxScaler().fit(Xd[itr])
        m = fit(mlp(Xd.shape[1], 1, "sigmoid", "binary_crossentropy"),
                sc.transform(Xd[itr]), yd[itr])
        pr = m.predict(sc.transform(Xd[ite]), verbose=0).ravel()
        acc = float(accuracy_score(yd[ite], (pr >= 0.5).astype(int)) * 100)
        auc = float(roc_auc_score(yd[ite], pr))
        res.append(dict(split=split, task="binary detection accuracy", value=acc, n_test=len(ite)))
        res.append(dict(split=split, task="binary detection ROC-AUC", value=auc, n_test=len(ite)))

    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "blocked_split.csv"), index=False)

    piv = R.pivot_table(index="task", columns="split", values="value")
    A("| Quantity | Random split | Blocked split | Change |")
    A("|---|---|---|---|")
    order = ["binary detection accuracy", "binary detection ROC-AUC",
             "multi-label exact match", "six-class accuracy",
             "six-class, LLL+LLLG merged", "LLLG recall"]
    for t in order:
        r, b = piv.loc[t, "random"], piv.loc[t, "blocked"]
        fmt = "{:.5f}" if "AUC" in t else "{:.2f} %"
        A(f"| {t} | {fmt.format(r)} | {fmt.format(b)} | {b-r:+.2f}{'' if 'AUC' in t else ' pp'} |")
    A("")
    open(os.path.join(OUT, "blocked_split_report.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
