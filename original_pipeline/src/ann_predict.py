"""
ann_predict.py -- Python equivalent of matlab/ann_predict.m

Line-for-line port of the MATLAB Coder entry-point function. The MATLAB source
writes the forward pass out as explicit matrix operations rather than calling
PREDICT on a dlnetwork, because MATLAB Coder handles Wx+b efficiently and does
not support every dlnetwork path on embedded targets (Blueprint v1.0,
"Practical Tips and Common Pitfalls"). This port keeps that structure so the
two can be compared statement by statement.

Correspondence with ann_predict.m
---------------------------------
    persistent W1 b1 ...        ->  module-level _CACHE, loaded once
    coder.load('ann_weights.mat')-> scipy.io.loadmat
    single(...)                 ->  np.float32
    x * W1 + b1                 ->  x @ W1 + b1     (Keras stores [n_in, n_out])
    max(z, single(0))           ->  np.maximum(z, 0)
    1 ./ (1 + exp(-z))          ->  1 / (1 + np.exp(-z))
    p > single(0.5)             ->  p > 0.5

Dropout is absent here for the same reason it is absent in the MATLAB file:
Keras applies dropout only during training, so inference is the identity.

Arithmetic is performed in float32 throughout, matching single() in MATLAB, so
the two implementations agree bit-for-bit up to library rounding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

_CACHE: dict | None = None
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "out" / "ann_weights.mat"


def load_weights(path=None) -> dict:
    """Equivalent of the `persistent` block plus coder.load in ann_predict.m."""
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    p = Path(path) if path is not None else DEFAULT_WEIGHTS
    S = sio.loadmat(str(p))
    W = {
        "W1": np.asarray(S["W1"], dtype=np.float32),
        "b1": np.asarray(S["b1"], dtype=np.float32).ravel(),
        "W2": np.asarray(S["W2"], dtype=np.float32),
        "b2": np.asarray(S["b2"], dtype=np.float32).ravel(),
        "W3": np.asarray(S["W3"], dtype=np.float32),
        "b3": np.asarray(S["b3"], dtype=np.float32).ravel(),
        "W4": np.asarray(S["W4"], dtype=np.float32),
        "b4": np.asarray(S["b4"], dtype=np.float32).ravel(),
    }
    if path is None:
        _CACHE = W
    return W


def ann_predict_scores(features, weights=None) -> np.ndarray:
    """Forward pass returning the raw sigmoid scores.

    features : (24,) or (N, 24) already-normalised inputs
    returns  : (4,) or (N, 4) scores in the order [G, C, B, A]
    """
    W = weights if weights is not None else load_weights()
    x = np.atleast_2d(np.asarray(features, dtype=np.float32))
    if x.shape[1] != W["W1"].shape[0]:
        raise ValueError(f"expected {W['W1'].shape[0]} features, got {x.shape[1]}")

    a1 = np.maximum(x  @ W["W1"] + W["b1"], np.float32(0))   # 1x64, ReLU
    a2 = np.maximum(a1 @ W["W2"] + W["b2"], np.float32(0))   # 1x32, ReLU
    a3 = np.maximum(a2 @ W["W3"] + W["b3"], np.float32(0))   # 1x16, ReLU
    z4 = a3 @ W["W4"] + W["b4"]                              # 1x4,  logits
    p = (np.float32(1) / (np.float32(1) + np.exp(-z4))).astype(np.float32)
    return p[0] if np.ndim(features) == 1 else p


def ann_predict(features, weights=None) -> np.ndarray:
    """Equivalent of ann_predict.m: returns the boolean flags [G C B A]."""
    return ann_predict_scores(features, weights) > np.float32(0.5)


def quantise_int8(W: np.ndarray):
    """Per-tensor symmetric int8 quantisation, Eq. (21).

    Mirrors the quantisation block of validate_generate_c.m:
        s = max|W| / 127 ;  q = clip(round(W/s), -128, 127) ;  W_hat = s*q
    Returns (q, s, max_abs_error).
    """
    W = np.asarray(W, dtype=np.float64)
    s = float(np.max(np.abs(W)) / 127.0)
    q = np.clip(np.round(W / s), -128, 127).astype(np.int8)
    err = float(np.max(np.abs(W - s * q)))
    return q, s, err


def discrepancy(y_a, y_b) -> float:
    """Eq. (20): D = (sum |y_a - y_b| / (N*K)) * 100 %."""
    a = np.asarray(y_a, dtype=np.float64).ravel()
    b = np.asarray(y_b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    return float(np.sum(np.abs(a - b)) / a.size * 100.0)
