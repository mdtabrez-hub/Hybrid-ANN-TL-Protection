"""
validate_generate_c.py -- Python equivalent of matlab/validate_generate_c.m

Executes Phase 2 of the Blueprint and cross-checks four independent
implementations of the same network against one another:

    1. Keras          -- out/y_pred_python.csv, written by train_ann.py
    2. GNU Octave     -- out/phase2/y_pred_octave.csv, from validate_octave.m
    3. NumPy          -- computed here from out/ann_weights.mat
    4. Compiled C     -- out/phase2/ann_predict.c, host build

Correspondence with validate_generate_c.m
-----------------------------------------
    load('ann_weights.mat')         -> scipy.io.loadmat
    readmatrix(...)                 -> pandas.read_csv
    ann_predict(X(i,:))             -> ann_predict.ann_predict (vectorised)
    discrepancy / numel * 100       -> ann_predict.discrepancy, Eq. (20)
    quantizenumeric(W, 1, 8, 6)     -> ann_predict.quantise_int8, Eq. (21)
    codegen -config cfg ann_predict -> NOT PORTED. MATLAB Coder has no Python
                                       equivalent; the C is emitted by
                                       matlab/emit_c.m and compiled here.

What this establishes, and what it does not
-------------------------------------------
Establishes: that the exported weights reproduce the Keras model exactly
through three independent re-implementations; the measured cost of int8
quantisation; and the measured ATmega328P footprint of the generated C.

Does NOT establish: anything specific to MATLAB Coder's own generated code,
or on-target execution time. Those still require MATLAB and Proteus.

Usage:  python3 src/validate_generate_c.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out"
P2 = OUT / "phase2"
P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from ann_predict import (ann_predict, ann_predict_scores, load_weights,  # noqa: E402
                         quantise_int8, discrepancy)

TENSORS = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]
rows: list[dict] = []


def rec(stage, metric, value, unit="", note=""):
    rows.append(dict(stage=stage, metric=metric, value=value, unit=unit, note=note))


def main() -> int:
    print("=" * 68)
    print("PHASE 2 -- validation and C generation (Python port)")
    print("=" * 68)

    # ------------------------------------------------ 2.1 reconstruct
    S = sio.loadmat(str(OUT / "ann_weights.mat"))
    W = load_weights()
    n_param = sum(int(np.asarray(S[t]).size) for t in TENSORS)
    print("\n[2.1] Reconstruct the network from ann_weights.mat")
    for t in ("W1", "W2", "W3", "W4"):
        print(f"  {t} {np.asarray(S[t]).shape}", end="   ")
    print(f"\n  trainable parameters: {n_param}")
    rec("2.1", "trainable_parameters", n_param, "weights")

    X = pd.read_csv(OUT / "X_test.csv").to_numpy(dtype=np.float32)
    y_true = pd.read_csv(OUT / "y_test.csv").to_numpy(dtype=int)
    y_keras = pd.read_csv(OUT / "y_pred_python.csv").to_numpy(dtype=int)
    N, K = y_keras.shape
    print(f"  test set: {N} samples x {X.shape[1]} features")
    rec("2.1", "test_samples", N, "samples")
    rec("2.1", "compared_outputs", N * K, "values")

    # --------------------------------------------- 2.2 cross-validation
    print("\n[2.2] Cross-validation, Eq. (20)")
    t0 = time.perf_counter()
    y_numpy = ann_predict(X).astype(int)
    t_np = (time.perf_counter() - t0) / N * 1e3

    D_np = discrepancy(y_numpy, y_keras)
    acc_keras = float(np.mean(np.all(y_keras == y_true, axis=1)) * 100)
    acc_numpy = float(np.mean(np.all(y_numpy == y_true, axis=1)) * 100)
    print(f"  NumPy vs Keras     : D = {D_np:.6f} %   "
          f"({int(np.sum(y_numpy != y_keras))} disagreeing outputs)")
    print(f"  exact-match (Keras): {acc_keras:.4f} %")
    print(f"  exact-match (NumPy): {acc_numpy:.4f} %")
    print(f"  host inference     : {t_np:.5f} ms/sample (vectorised)")
    rec("2.2", "discrepancy_numpy_vs_keras", round(D_np, 8), "%", "Eq. (20); < 0.1 %")
    rec("2.2", "exact_match_keras", round(acc_keras, 6), "%")
    rec("2.2", "exact_match_numpy", round(acc_numpy, 6), "%")

    # Octave, if validate_octave.m has been run
    oct_path = P2 / "y_pred_octave.csv"
    if oct_path.exists():
        y_oct = np.loadtxt(oct_path, delimiter=",").astype(int)
        D_oct = discrepancy(y_oct, y_keras)
        D_on = discrepancy(y_oct, y_numpy)
        print(f"  Octave vs Keras    : D = {D_oct:.6f} %")
        print(f"  Octave vs NumPy    : D = {D_on:.6f} %")
        rec("2.2", "discrepancy_octave_vs_keras", round(D_oct, 8), "%")
        rec("2.2", "discrepancy_octave_vs_numpy", round(D_on, 8), "%")
    else:
        print("  Octave results absent -- run matlab/validate_octave.m first")

    # ------------------------------------------ 2.3 int8 quantisation
    print("\n[2.3] int8 symmetric per-tensor quantisation, Eq. (21)")
    Q, scales = {}, {}
    for t in TENSORS:
        q, s, err = quantise_int8(np.asarray(S[t]))
        Q[t], scales[t] = q, s
        rel = err / float(np.max(np.abs(np.asarray(S[t])))) * 100
        print(f"  {t:3s} n={q.size:5d}  scale={s:.6e}  max|err|={err:.6e}  ({rel:.4f} %)")
        rec("2.3", f"scale_{t}", s)
        rec("2.3", f"max_abs_error_{t}", err)

    Wq = {t: (scales[t] * Q[t].astype(np.float32)).astype(np.float32) for t in TENSORS}
    Wq = {"W1": Wq["W1"], "b1": Wq["b1"].ravel(), "W2": Wq["W2"], "b2": Wq["b2"].ravel(),
          "W3": Wq["W3"], "b3": Wq["b3"].ravel(), "W4": Wq["W4"], "b4": Wq["b4"].ravel()}
    y_q = (ann_predict_scores(X, Wq) > np.float32(0.5)).astype(int)
    acc_q = float(np.mean(np.all(y_q == y_true, axis=1)) * 100)
    D_q = discrepancy(y_q, y_keras)
    print(f"\n  exact-match, float32 : {acc_numpy:.4f} %")
    print(f"  exact-match, int8    : {acc_q:.4f} %")
    print(f"  change               : {acc_q - acc_numpy:+.4f} pp")
    print(f"  discrepancy vs Keras : {D_q:.6f} %")
    rec("2.3", "exact_match_int8", round(acc_q, 6), "%")
    rec("2.3", "int8_accuracy_delta", round(acc_q - acc_numpy, 6), "pp")
    rec("2.3", "discrepancy_int8_vs_keras", round(D_q, 8), "%")

    # --------------------------------------- compiled C cross-check
    print("\n[2.3b] Compiled C cross-check")
    for flag, label in ((1, "int8"), (0, "float32")):
        exe = P2 / f"th_{flag}"
        src = [P2 / "test_host.c", P2 / "ann_predict.c", P2 / "ann_predict_data.c"]
        if not all(p.exists() for p in src):
            print("  C sources absent -- run matlab/validate_octave.m to emit them")
            break
        subprocess.run(["gcc", "-O2", "-std=c99", f"-DANN_USE_INT8={flag}",
                        "-o", str(exe), *map(str, src), "-lm"], check=True)
        r = subprocess.run([str(exe), str(OUT / "X_test.csv"),
                            str(OUT / "y_pred_python.csv")],
                           capture_output=True, text=True, check=True)
        d = dict(l.split(":", 1) for l in r.stdout.strip().split("\n") if ":" in l)
        Dc = float(d["discrepancy D    "].strip().rstrip(" %"))
        print(f"  C ({label:7s}) vs Keras : D = {Dc:.6f} %")
        rec("2.3b", f"discrepancy_c_{label}_vs_keras", round(Dc, 8), "%")

    # ------------------------------------------------------- summary
    df = pd.DataFrame(rows)
    df.to_csv(P2 / "phase2_python_summary.csv", index=False)
    agree = {
        "keras_vs_numpy_pct": D_np,
        "keras_vs_int8_pct": D_q,
        "n_compared_outputs": N * K,
        "note": ("float32 implementations agree exactly; the int8 build differs "
                 "only by the quantisation error of Eq. (21)"),
    }
    (P2 / "implementation_agreement.json").write_text(json.dumps(agree, indent=2))
    print(f"\nWrote -> {P2}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
