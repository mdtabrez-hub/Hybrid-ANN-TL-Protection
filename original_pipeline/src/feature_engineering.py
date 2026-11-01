"""
feature_engineering.py
======================
Feature extraction for the Hybrid ANN-Proteus transmission-line protection
framework (Blueprint v1.0, Phase 1.1).

Two distinct feature regimes are implemented and kept strictly separate:

A. ROW-WISE SCALAR FEATURES  -- build_rowwise_features()
   Executable on the supplied classData.csv / detect_dataset.csv, which contain
   ONE scalar sample per channel per row.  Produces the 24-feature vector
   specified in the manuscript.  Quantities named *_rms / *_peak here are
   ACROSS-PHASE aggregates of a single time sample, NOT time-window RMS or peak.
   They are labelled explicitly as proxies.

B. WAVEFORM-WINDOW FEATURES  -- window_rms(), window_peak(), fft_features(),
   thd(), dwt_energy()
   These are the physically correct definitions required by the Blueprint.
   They REQUIRE a waveform window x[0..N-1] and a sampling frequency fs.
   They are provided for the future waveform run and are NOT invoked on the
   scalar CSVs.  Calling them with a scalar raises ValueError by design.

Author:  Md Tabrez, Dept. of EEE, Motihari College of Engineering, Bihar, India
"""

import numpy as np
import pandas as pd

MEAS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

FEATURE_NAMES = [
    "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
    "abs_Ia", "abs_Ib", "abs_Ic", "abs_Va", "abs_Vb", "abs_Vc",
    "I_rms", "V_rms", "I_peak", "V_peak",
    "I_zero_seq", "V_zero_seq",
    "I_neg_seq_real", "I_neg_seq_imag",
    "V_neg_seq_real", "V_neg_seq_imag",
    "I_phase_std", "P_3ph_proxy",
]

# Fortescue operator a = exp(j2pi/3)
A_OP = np.exp(2j * np.pi / 3)


# --------------------------------------------------------------------------
# A. ROW-WISE SCALAR FEATURES (executable on the supplied data)
# --------------------------------------------------------------------------
def build_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 24-feature row-wise representation from scalar three-phase
    samples.

    Parameters
    ----------
    df : DataFrame containing columns Ia, Ib, Ic, Va, Vb, Vc.

    Returns
    -------
    DataFrame with columns FEATURE_NAMES, same row order as `df`.

    Notes
    -----
    * I_rms / V_rms are across-phase quadratic means of one time sample:
          I_rms = sqrt((Ia^2 + Ib^2 + Ic^2)/3)
      For a balanced sinusoidal set sampled simultaneously this equals the
      true RMS; for a faulted (unbalanced) set it does not.  It is therefore
      an approximation and is reported as such.
    * I_neg_seq_* is the Fortescue negative-sequence combination evaluated on
      real instantaneous samples rather than phasors.  It is an algebraic
      proxy, valid as a discriminative feature but NOT a measured phasor
      sequence component.
    """
    missing = [c for c in MEAS if c not in df.columns]
    if missing:
        raise KeyError(f"missing measurement columns: {missing}")

    Ia, Ib, Ic = df["Ia"].to_numpy(), df["Ib"].to_numpy(), df["Ic"].to_numpy()
    Va, Vb, Vc = df["Va"].to_numpy(), df["Vb"].to_numpy(), df["Vc"].to_numpy()

    I2 = (Ia + (A_OP ** 2) * Ib + A_OP * Ic) / 3.0
    V2 = (Va + (A_OP ** 2) * Vb + A_OP * Vc) / 3.0

    out = {
        "Ia": Ia, "Ib": Ib, "Ic": Ic, "Va": Va, "Vb": Vb, "Vc": Vc,
        "abs_Ia": np.abs(Ia), "abs_Ib": np.abs(Ib), "abs_Ic": np.abs(Ic),
        "abs_Va": np.abs(Va), "abs_Vb": np.abs(Vb), "abs_Vc": np.abs(Vc),
        "I_rms": np.sqrt((Ia ** 2 + Ib ** 2 + Ic ** 2) / 3.0),
        "V_rms": np.sqrt((Va ** 2 + Vb ** 2 + Vc ** 2) / 3.0),
        "I_peak": np.maximum.reduce([np.abs(Ia), np.abs(Ib), np.abs(Ic)]),
        "V_peak": np.maximum.reduce([np.abs(Va), np.abs(Vb), np.abs(Vc)]),
        "I_zero_seq": (Ia + Ib + Ic) / 3.0,
        "V_zero_seq": (Va + Vb + Vc) / 3.0,
        "I_neg_seq_real": I2.real, "I_neg_seq_imag": I2.imag,
        "V_neg_seq_real": V2.real, "V_neg_seq_imag": V2.imag,
        "I_phase_std": np.std(np.vstack([Ia, Ib, Ic]), axis=0, ddof=0),
        "P_3ph_proxy": Va * Ia + Vb * Ib + Vc * Ic,
    }
    return pd.DataFrame(out, columns=FEATURE_NAMES, index=df.index)


# --------------------------------------------------------------------------
# B. WAVEFORM-WINDOW FEATURES (require waveform data; NOT run on scalar CSVs)
# --------------------------------------------------------------------------
def _require_window(x, min_len=8):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < min_len:
        raise ValueError(
            f"waveform window of at least {min_len} samples required; got "
            f"{x.size}. The supplied CSVs hold one scalar per channel per row, "
            "so this function cannot be applied to them. Supply waveform "
            "windows with a documented sampling frequency."
        )
    return x


def window_rms(x):
    """True time-window RMS:  sqrt( (1/N) * sum_{n=0}^{N-1} x[n]^2 )."""
    x = _require_window(x)
    return float(np.sqrt(np.mean(x ** 2)))


def window_peak(x):
    """True time-window peak:  max_n |x[n]|."""
    x = _require_window(x)
    return float(np.max(np.abs(x)))


def fft_features(x, fs, f0=50.0, harmonics=(1, 2, 3, 5, 7)):
    """
    Single-sided amplitude spectrum sampled at the requested harmonic orders.

    Parameters
    ----------
    x  : waveform window
    fs : sampling frequency [Hz]  (REQUIRED -- no default is assumed)
    f0 : fundamental frequency [Hz]
    harmonics : harmonic orders to return

    Returns
    -------
    dict {f"h{k}": amplitude}  with amplitudes scaled to physical units
    (2/N * |X[k]|, DC and Nyquist unscaled).
    """
    x = _require_window(x)
    N = x.size
    X = np.fft.rfft(x)
    amp = (2.0 / N) * np.abs(X)
    if N % 2 == 0:
        amp[-1] = np.abs(X[-1]) / N
    amp[0] = np.abs(X[0]) / N
    df = fs / N
    out = {}
    for k in harmonics:
        idx = int(round(k * f0 / df))
        out[f"h{k}"] = float(amp[idx]) if 0 <= idx < amp.size else float("nan")
    return out


def thd(x, fs, f0=50.0, H=40):
    """
    Total harmonic distortion referred to the fundamental:

        THD = sqrt( sum_{h=2}^{H} X_h^2 ) / X_1

    Returned as a ratio; multiply by 100 for per cent.
    """
    x = _require_window(x)
    spec = fft_features(x, fs, f0, harmonics=tuple(range(1, H + 1)))
    x1 = spec["h1"]
    if not np.isfinite(x1) or x1 == 0:
        return float("nan")
    harm_sq = sum(spec[f"h{h}"] ** 2 for h in range(2, H + 1)
                  if np.isfinite(spec[f"h{h}"]))
    return float(np.sqrt(harm_sq) / x1)


def dwt_energy(x, wavelet="db4", level=4, normalise=True):
    """
    Energy of each DWT sub-band:  E_j = sum_n |c_j[n]|^2

    Returns [E_A_level, E_D_level, ..., E_D_1].  If `normalise`, each energy is
    divided by the total so the vector is scale-invariant.
    Requires PyWavelets.
    """
    x = _require_window(x, min_len=2 ** level)
    try:
        import pywt
    except ImportError as e:
        raise ImportError("PyWavelets is required for dwt_energy()") from e
    coeffs = pywt.wavedec(x, wavelet, level=level)
    E = np.array([float(np.sum(np.asarray(c, dtype=float) ** 2)) for c in coeffs])
    if normalise:
        tot = E.sum()
        if tot > 0:
            E = E / tot
    return E.tolist()


def sequence_components(pa, pb, pc):
    """
    Fortescue symmetrical components from COMPLEX phasors.

        [V0; V1; V2] = (1/3) * [[1,1,1],[1,a,a^2],[1,a^2,a]] * [Va; Vb; Vc]

    Inputs must be complex phasors (e.g. from the fundamental DFT bin of a
    waveform window).  Passing real instantaneous samples yields the algebraic
    proxy used in the row-wise feature set, not true sequence components.
    """
    a = A_OP
    v0 = (pa + pb + pc) / 3.0
    v1 = (pa + a * pb + (a ** 2) * pc) / 3.0
    v2 = (pa + (a ** 2) * pb + a * pc) / 3.0
    return v0, v1, v2
