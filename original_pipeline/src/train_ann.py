"""
train_ann.py -- Phase 1 of the Hybrid ANN-Proteus framework (Blueprint v1.0).

Executes, on the supplied scalar-row datasets:

  A. Multi-label fault classification  -- EXACT Keras model from the Blueprint
                                          (24 -> 64 -> 32 -> 16 -> 4 sigmoid)
  B. Multi-label scikit-learn MLP      -- independent cross-check of A
  C. Binary fault detection            -- EXACT Keras topology, 1 sigmoid
  D. Six-class fault typing            -- EXACT Keras topology, 6 softmax
  E. SMOTE six-class diagnostic        -- balancing sensitivity study

Reproducibility: all randomness seeded with SEED = 42; splits are stratified on
the composite label vector; MinMaxScaler is fitted on the training partition
only.  Artifacts required by the MATLAB / Proteus stages are exported to
`out/`.

Honesty contract
----------------
* Nothing here computes FFT, THD or DWT: the supplied CSVs hold one scalar
  sample per channel per row, so those window operations are not defined on
  them.  The functions exist in feature_engineering.py for the waveform run.
* Every reported number in out/metrics.csv is produced by this script on the
  supplied data.  No target, expectation or literature value is written there.

Usage:  python3 src/train_ann.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
random.seed(SEED)
np.random.seed(SEED)

import joblib
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, hamming_loss,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from feature_engineering import build_rowwise_features, FEATURE_NAMES, MEAS  # noqa: E402

import os as _os
DATA = Path(_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")))
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Label vectors are read in the order (G, C, B, A) as supplied.
# NOTE: the LL vector observed in the data is 0110 (phases B and C, no ground).
# Blueprint v1.0 Sec. 0.1 states 0011 (phases A and B); that vector does not
# occur in classData.csv.  See out/label_audit.json for the physical evidence.
VEC2CLASS = {
    "0000": "No fault",
    "1001": "LG (A-G)",
    "0110": "LL (B-C)",
    "1011": "LLG (A-B-G)",
    "0111": "LLL",
    "1111": "LLLG",
}
CLASS_ORDER = ["No fault", "LG (A-G)", "LL (B-C)", "LLG (A-B-G)", "LLL", "LLLG"]

_metrics: list[dict] = []


def rec(experiment, model, task, metric, value, unit="", note=""):
    _metrics.append(dict(experiment=experiment, model=model, task=task,
                         metric=metric, value=value, unit=unit, note=note))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- load/audit
def load_data():
    cd = pd.read_csv(DATA / "classData.csv")
    dd = pd.read_csv(DATA / "detect_dataset.csv")
    empty = [c for c in dd.columns if dd[c].isna().all()]
    dd = dd.drop(columns=empty)
    dd = dd.rename(columns={"Output (S)": "S"})
    log(f"classData.csv      {cd.shape[0]} rows x {cd.shape[1]} cols")
    log(f"detect_dataset.csv {dd.shape[0]} rows x {dd.shape[1]} cols "
        f"(dropped all-empty: {empty})")

    rec("audit", "-", "-", "classData_rows", cd.shape[0], "rows")
    rec("audit", "-", "-", "classData_cols", cd.shape[1], "cols")
    rec("audit", "-", "-", "detect_rows", dd.shape[0], "rows")
    rec("audit", "-", "-", "detect_cols_after_drop", dd.shape[1], "cols")
    rec("audit", "-", "-", "classData_missing_required", int(cd.isna().sum().sum()))
    rec("audit", "-", "-", "detect_missing_required", int(dd.isna().sum().sum()))
    rec("audit", "-", "-", "classData_duplicate_rows", int(cd.duplicated().sum()))
    rec("audit", "-", "-", "detect_duplicate_rows", int(dd.duplicated().sum()))
    rec("audit", "-", "-", "detect_empty_cols_dropped", len(empty), "cols",
        ";".join(empty))
    return cd, dd


def label_audit(cd):
    """Verify label semantics against measured currents; save the evidence."""
    vec = cd[["G", "C", "B", "A"]].astype(int).astype(str).agg("".join, axis=1)
    counts = vec.value_counts().to_dict()
    ev = {}
    for v, g in cd.assign(vec=vec).groupby("vec"):
        I = {p: float(g[f"I{p.lower()}"].abs().mean()) for p in "ABC"}
        V = {p: float(g[f"V{p.lower()}"].abs().mean()) for p in "ABC"}
        I0 = float((g[["Ia", "Ib", "Ic"]].sum(axis=1) / 3).abs().mean())
        mx = max(I.values())
        ev[v] = dict(n=int(len(g)), mean_abs_I=I, mean_abs_V=V,
                     mean_abs_I0=I0,
                     elevated_current_phases=[p for p, x in I.items() if x > 0.5 * mx],
                     ground_involved=bool(I0 > 10.0))
    for v, n in counts.items():
        rec("audit", "-", "label", f"count_{v}", int(n), "samples",
            VEC2CLASS.get(v, "UNMAPPED"))
    blueprint_ll = "0011"
    audit = dict(
        counts=counts, evidence=ev,
        blueprint_LL_vector=blueprint_ll,
        blueprint_LL_present_in_data=blueprint_ll in counts,
        observed_LL_vector="0110",
        finding=("Blueprint v1.0 Sec. 0.1 lists 0011 (LL, phases A-B). That "
                 "vector does not occur in classData.csv. The data contains "
                 "0110 (G=0, C=1, B=1, A=0), whose measured currents are "
                 "elevated in phases B and C with near-zero zero-sequence "
                 "current, i.e. an ungrounded line-to-line fault between B "
                 "and C. The bit-to-phase mapping is self-consistent across "
                 "all six vectors; only the Blueprint's LL entry is in error. "
                 "The Blueprint's further statement that every faulted case "
                 "involves phase A is therefore also contradicted by the LL "
                 "class."))
    (OUT / "label_audit.json").write_text(json.dumps(audit, indent=2))
    log("label audit -> out/label_audit.json")
    return vec


def iqr_screen(cd):
    rows = []
    for c in MEAS:
        q1, q3 = cd[c].quantile([.25, .75])
        iqr = q3 - q1
        n = int(((cd[c] < q1 - 1.5 * iqr) | (cd[c] > q3 + 1.5 * iqr)).sum())
        p1, p99 = cd[c].quantile([.01, .99])
        rows.append(dict(channel=c, iqr_flags=n, p1=float(p1), p99=float(p99)))
        rec("preprocess", "-", "-", f"iqr_flags_{c}", n, "samples")
        rec("preprocess", "-", "-", f"p01_{c}", float(p1))
        rec("preprocess", "-", "-", f"p99_{c}", float(p99))
    pd.DataFrame(rows).to_csv(OUT / "iqr_screening.csv", index=False)
    return pd.DataFrame(rows)


def clip_1_99(df, cols):
    """Clip to the 1st/99th percentile. Rows are never deleted."""
    out = df.copy()
    for c in cols:
        lo, hi = df[c].quantile([.01, .99])
        out[c] = df[c].clip(lo, hi)
    return out


# ------------------------------------------------------------------ Keras
def keras_mlp(n_in, n_out, out_act, loss, lr=1e-3):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    init = tf.keras.initializers.HeNormal(seed=SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(0.2, seed=SEED),
        tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(0.2, seed=SEED),
        tf.keras.layers.Dense(16, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dense(n_out, activation=out_act,
                              kernel_initializer=init),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=loss, metrics=["accuracy"])
    return m


def keras_fit(model, Xtr, ytr, tag):
    import tensorflow as tf
    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                         monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5,
                                             monitor="val_loss", min_lr=1e-6),
    ]
    t0 = time.time()
    h = model.fit(Xtr, ytr, epochs=100, batch_size=32, validation_split=0.15,
                  callbacks=cbs, verbose=0, shuffle=True)
    dt = time.time() - t0
    ep = len(h.history["loss"])
    log(f"  [{tag}] trained {ep} epochs in {dt:.1f}s "
        f"(val_loss {min(h.history['val_loss']):.5f})")
    pd.DataFrame(h.history).to_csv(OUT / f"history_{tag}.csv", index=False)
    rec(tag, "Keras", "-", "epochs_run", ep, "epochs", "early stopping")
    rec(tag, "Keras", "-", "train_wall_time", round(dt, 2), "s")
    rec(tag, "Keras", "-", "best_val_loss", round(float(min(h.history["val_loss"])), 6))
    return h


def multilabel_metrics(exp, model_name, y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    exact = accuracy_score(y_true, y_pred)
    mi = precision_recall_fscore_support(y_true, y_pred, average="micro",
                                         zero_division=0)
    ma = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                         zero_division=0)
    hl = hamming_loss(y_true, y_pred)
    rec(exp, model_name, "multi-label", "exact_match_accuracy",
        round(exact * 100, 4), "%")
    rec(exp, model_name, "multi-label", "micro_precision", round(mi[0] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "micro_recall", round(mi[1] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "micro_f1", round(mi[2] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "macro_precision", round(ma[0] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "macro_recall", round(ma[1] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "macro_f1", round(ma[2] * 100, 4), "%")
    rec(exp, model_name, "multi-label", "hamming_loss", round(hl, 6))
    per = []
    for i, lab in enumerate(["G", "C", "B", "A"]):
        p, r, f, s = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        per.append(dict(label=lab, precision=p * 100, recall=r * 100,
                        f1=f * 100, roc_auc=auc,
                        support=int(y_true[:, i].sum())))
        rec(exp, model_name, f"label_{lab}", "precision", round(p * 100, 4), "%")
        rec(exp, model_name, f"label_{lab}", "recall", round(r * 100, 4), "%")
        rec(exp, model_name, f"label_{lab}", "f1", round(f * 100, 4), "%")
        rec(exp, model_name, f"label_{lab}", "roc_auc", round(float(auc), 6))
    pd.DataFrame(per).to_csv(OUT / f"per_label_{exp}_{model_name}.csv", index=False)
    log(f"  [{exp}/{model_name}] exact={exact*100:.2f}%  microF1={mi[2]*100:.2f}%  "
        f"hamming={hl:.4f}")
    return dict(exact=exact, micro=mi, macro=ma, hamming=hl,
                per_label=pd.DataFrame(per), y_pred=y_pred)


# ------------------------------------------------------------------- main
def main():
    log("=" * 70)
    log("PHASE 1 -- Python feature engineering and ANN training")
    log("=" * 70)

    cd, dd = load_data()
    vec = label_audit(cd)
    iqr_screen(cd)

    # ---------------- Experiment A/B: multi-label classification ----------
    log("\n[A] Multi-label classification (exact Keras)")
    cdc = clip_1_99(cd, MEAS)
    X = build_rowwise_features(cdc).to_numpy(dtype=np.float64)
    Y = cd[["G", "C", "B", "A"]].to_numpy(dtype=np.int32)

    Xtr, Xte, ytr, yte, vtr, vte = train_test_split(
        X, Y, vec.to_numpy(), test_size=0.2, random_state=SEED, stratify=vec)
    scaler = MinMaxScaler().fit(Xtr)          # fitted on TRAIN ONLY
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    joblib.dump(scaler, OUT / "scaler.pkl")
    log(f"  split: train={len(Xtr)}  test={len(Xte)}  features={X.shape[1]}")
    rec("A", "-", "-", "n_train", len(Xtr), "samples")
    rec("A", "-", "-", "n_test", len(Xte), "samples")
    rec("A", "-", "-", "n_features", X.shape[1], "features")

    mA = keras_mlp(X.shape[1], 4, "sigmoid", "binary_crossentropy")
    n_par = int(mA.count_params())
    rec("A", "Keras", "-", "trainable_parameters", n_par, "weights")
    rec("A", "Keras", "-", "int8_weight_memory", round(n_par / 1024, 3), "KiB",
        "1 byte/weight after 8-bit quantisation")
    log(f"  parameters: {n_par}  ({n_par/1024:.2f} KiB as int8)")
    keras_fit(mA, Xtr_s, ytr, "A")
    probA = mA.predict(Xte_s, verbose=0)
    resA = multilabel_metrics("A", "Keras", yte, probA)

    log("\n[B] Multi-label scikit-learn MLP cross-check")
    mB = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu",
                       solver="adam", learning_rate_init=1e-3, batch_size=32,
                       max_iter=300, early_stopping=True, n_iter_no_change=10,
                       validation_fraction=0.15, random_state=SEED)
    mB.fit(Xtr_s, ytr)
    probB = np.column_stack([p[:, 1] for p in mB.predict_proba(Xte_s)]) \
        if isinstance(mB.predict_proba(Xte_s), list) else mB.predict_proba(Xte_s)
    resB = multilabel_metrics("B", "sklearn-MLP", yte, np.asarray(probB))

    # ---------------- Experiment C: binary detection ----------------------
    log("\n[C] Binary fault detection (exact Keras topology, 1 sigmoid)")
    ddc = clip_1_99(dd, MEAS)
    Xd = build_rowwise_features(ddc).to_numpy(dtype=np.float64)
    yd = dd["S"].to_numpy(dtype=np.int32)
    Xdtr, Xdte, ydtr, ydte = train_test_split(
        Xd, yd, test_size=0.2, random_state=SEED, stratify=yd)
    sc_d = MinMaxScaler().fit(Xdtr)
    Xdtr_s, Xdte_s = sc_d.transform(Xdtr), sc_d.transform(Xdte)
    joblib.dump(sc_d, OUT / "scaler_detect.pkl")
    rec("C", "-", "-", "n_train", len(Xdtr), "samples")
    rec("C", "-", "-", "n_test", len(Xdte), "samples")

    mC = keras_mlp(Xd.shape[1], 1, "sigmoid", "binary_crossentropy")
    keras_fit(mC, Xdtr_s, ydtr, "C")
    probC = mC.predict(Xdte_s, verbose=0).ravel()
    predC = (probC >= 0.5).astype(int)
    accC = accuracy_score(ydte, predC)
    pC, rC, fC, _ = precision_recall_fscore_support(ydte, predC, average="binary",
                                                    zero_division=0)
    aucC = roc_auc_score(ydte, probC)
    tn, fp, fn, tp = confusion_matrix(ydte, predC).ravel()
    for k, v, u in [("accuracy", accC * 100, "%"), ("precision", pC * 100, "%"),
                    ("recall", rC * 100, "%"), ("f1", fC * 100, "%"),
                    ("roc_auc", aucC, ""), ("TN", tn, "samples"),
                    ("FP", fp, "samples"), ("FN", fn, "samples"),
                    ("TP", tp, "samples")]:
        rec("C", "Keras", "binary-detection", k,
            round(float(v), 6) if u != "samples" else int(v), u)
    log(f"  acc={accC*100:.3f}%  AUC={aucC:.6f}  TN/FP/FN/TP={tn}/{fp}/{fn}/{tp}")
    np.save(OUT / "roc_C.npy", np.array(roc_curve(ydte, probC)[:2], dtype=object),
            allow_pickle=True)
    pd.DataFrame(dict(y_true=ydte, y_prob=probC)).to_csv(
        OUT / "detect_test_scores.csv", index=False)

    # ---------------- Experiment D: six-class softmax ---------------------
    log("\n[D] Six-class fault typing (exact Keras topology, 6 softmax)")
    cls = pd.Series(vec).map(VEC2CLASS)
    assert cls.notna().all(), "unmapped label vector present"
    yc = pd.Categorical(cls, categories=CLASS_ORDER).codes
    Xctr, Xcte, yctr, ycte = train_test_split(
        X, yc, test_size=0.2, random_state=SEED, stratify=yc)
    sc_c = MinMaxScaler().fit(Xctr)
    Xctr_s, Xcte_s = sc_c.transform(Xctr), sc_c.transform(Xcte)
    mD = keras_mlp(X.shape[1], 6, "softmax", "sparse_categorical_crossentropy")
    keras_fit(mD, Xctr_s, yctr, "D")
    probD = mD.predict(Xcte_s, verbose=0)
    predD = probD.argmax(1)
    accD = accuracy_score(ycte, predD)
    rec("D", "Keras", "six-class", "accuracy", round(accD * 100, 4), "%")
    pr, rr, ff, ss = precision_recall_fscore_support(ycte, predD, zero_division=0)
    rowsD = []
    for i, name in enumerate(CLASS_ORDER):
        rowsD.append(dict(cls=name, support=int(ss[i]), precision=pr[i] * 100,
                          recall=rr[i] * 100, f1=ff[i] * 100))
        for k, v in [("precision", pr[i]), ("recall", rr[i]), ("f1", ff[i])]:
            rec("D", "Keras", f"class_{name}", k, round(v * 100, 4), "%")
        rec("D", "Keras", f"class_{name}", "support", int(ss[i]), "samples")
    pd.DataFrame(rowsD).to_csv(OUT / "per_class_D.csv", index=False)
    np.savetxt(OUT / "cm_D.csv", confusion_matrix(ycte, predD), fmt="%d",
               delimiter=",")
    log(f"  six-class accuracy = {accD*100:.2f}%")
    log("\n" + classification_report(ycte, predD, target_names=CLASS_ORDER,
                                     zero_division=0, digits=4))

    # ---------------- Experiment E: SMOTE diagnostic ----------------------
    log("[E] SMOTE six-class diagnostic")
    from imblearn.over_sampling import SMOTE
    Xsm, ysm = SMOTE(random_state=SEED).fit_resample(Xctr_s, yctr)
    rec("E", "SMOTE", "six-class", "train_size_after_smote", int(len(Xsm)), "samples")
    mE = keras_mlp(X.shape[1], 6, "softmax", "sparse_categorical_crossentropy")
    keras_fit(mE, Xsm, ysm, "E")
    predE = mE.predict(Xcte_s, verbose=0).argmax(1)
    accE = accuracy_score(ycte, predE)
    rec("E", "Keras+SMOTE", "six-class", "accuracy", round(accE * 100, 4), "%")
    prE, rrE, ffE, ssE = precision_recall_fscore_support(ycte, predE, zero_division=0)
    rowsE = []
    for i, name in enumerate(CLASS_ORDER):
        rowsE.append(dict(cls=name, support=int(ssE[i]), precision=prE[i] * 100,
                          recall=rrE[i] * 100, f1=ffE[i] * 100))
        for k, v in [("precision", prE[i]), ("recall", rrE[i]), ("f1", ffE[i])]:
            rec("E", "Keras+SMOTE", f"class_{name}", k, round(v * 100, 4), "%")
    pd.DataFrame(rowsE).to_csv(OUT / "per_class_E.csv", index=False)
    np.savetxt(OUT / "cm_E.csv", confusion_matrix(ycte, predE), fmt="%d",
               delimiter=",")
    log(f"  SMOTE six-class accuracy = {accE*100:.2f}%  "
        f"(no-SMOTE baseline {accD*100:.2f}%)")

    # ---------------- exports for MATLAB / Proteus ------------------------
    log("\n[export] artifacts for the MATLAB and Proteus stages")
    W = {}
    dense = [l for l in mA.layers if l.__class__.__name__ == "Dense"]
    for i, l in enumerate(dense, start=1):
        w, b = l.get_weights()
        W[f"W{i}"], W[f"b{i}"] = w.astype(np.float64), b.astype(np.float64)
    W["feature_names"] = np.array(FEATURE_NAMES, dtype=object)
    W["scaler_min"] = scaler.data_min_.astype(np.float64)
    W["scaler_max"] = scaler.data_max_.astype(np.float64)
    W["label_order"] = np.array(["G", "C", "B", "A"], dtype=object)
    sio.savemat(OUT / "ann_weights.mat", W)
    mA.save(OUT / "ann_model.keras")
    try:
        mA.save(OUT / "ann_model.h5")
    except Exception as e:                       # legacy format may be absent
        log(f"  (.h5 export skipped: {e})")

    pd.DataFrame(Xte_s, columns=FEATURE_NAMES).to_csv(OUT / "X_test.csv", index=False)
    pd.DataFrame(yte, columns=["G", "C", "B", "A"]).to_csv(OUT / "y_test.csv", index=False)
    pd.DataFrame(resA["y_pred"], columns=["G", "C", "B", "A"]).to_csv(
        OUT / "y_pred_python.csv", index=False)
    pd.DataFrame(probA, columns=["pG", "pC", "pB", "pA"]).to_csv(
        OUT / "y_prob_python.csv", index=False)

    hdr = ["// scaler_params.h -- generated by train_ann.py; do not edit",
           "#ifndef SCALER_PARAMS_H", "#define SCALER_PARAMS_H",
           f"#define N_FEATURES {len(FEATURE_NAMES)}",
           "const float scaler_min[N_FEATURES] = {" +
           ", ".join(f"{v:.8g}f" for v in scaler.data_min_) + "};",
           "const float scaler_max[N_FEATURES] = {" +
           ", ".join(f"{v:.8g}f" for v in scaler.data_max_) + "};",
           "#endif"]
    (OUT / "scaler_params.h").write_text("\n".join(hdr) + "\n")

    pd.DataFrame(_metrics).to_csv(OUT / "metrics.csv", index=False)
    log(f"\nWrote {len(_metrics)} metric rows -> out/metrics.csv")
    log("Artifacts: scaler.pkl, ann_weights.mat, ann_model.keras, X_test.csv, "
        "y_test.csv, y_pred_python.csv, scaler_params.h")
    log("=" * 70)


if __name__ == "__main__":
    main()
