"""
make_figures.py -- all manuscript figures, built only from executed artifacts
in out/ or from the supplied CSVs.  Figures for unexecuted external stages are
rendered as explicitly stamped placeholders and carry no invented data.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import apply, CAT, SEQ, DIV, INK, INK_MUTED, GRID, pending_banner
from feature_engineering import build_rowwise_features, MEAS
from train_ann import CLASS_ORDER, VEC2CLASS, clip_1_99

apply()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "out"), os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
import os as _os
U = _os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd = pd.read_csv(U + "classData.csv")
vec = cd[["G", "C", "B", "A"]].astype(int).astype(str).agg("".join, axis=1)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=400)
    plt.close(fig)
    print("  wrote", name)


# ---- Fig 1: verification chain -------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(10.2, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10.2); ax.set_ylim(0, 2.5)
    stages = [("MATLAB/\nSimulink", "fault data\n(CSV)", CAT[5], "supplied"),
              ("Python", "feature eng.\n+ ANN training", CAT[2], "executed"),
              ("Octave / C\n(not Coder)", "validation\n+ C code", CAT[2], "executed"),
              ("Proteus", "HIL emulation\n+ trip latency", "#8A8F96", "outofscope"),
              ("pandapower", "protection\ncoordination", CAT[2], "calibrated")]
    w, h, gap = 1.72, 1.05, 0.30
    for i, (t, s, c, st) in enumerate(stages):
        x = 0.2 + i * (w + gap)
        solid = st in ("executed", "calibrated")
        LBL = {"supplied": "NOT EXECUTED\ndata from [13]", "executed": "EXECUTED",
               "outofscope": "OUT OF SCOPE\n(this paper)",
               "calibrated": "EXECUTED\non a calibrated network"}
        if t.startswith("Octave"):
            LBL["executed"] = "EXECUTED\nD = 0.000000 %"
        ax.add_patch(FancyBboxPatch((x, 0.95), w, h, boxstyle="round,pad=0.05",
                                    facecolor=c if solid else "#FFFFFF",
                                    edgecolor=c, linewidth=1.8,
                                    linestyle="-" if solid else (0, (4, 2)),
                                    alpha=1.0 if solid else 0.95))
        ax.text(x + w / 2, 1.72, t, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white" if solid else c)
        ax.text(x + w / 2, 1.25, s, ha="center", va="center", fontsize=7.6,
                color="white" if solid else INK_MUTED)
        ax.text(x + w / 2, 0.63, LBL[st], ha="center", va="center", fontsize=6.6,
                fontweight="bold", linespacing=1.35,
                color=CAT[2] if solid else ("#6B7076" if st == "outofscope" else "#B25400"))
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 1.47),
                                         (x + w + gap - 0.03, 1.47),
                                         arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.6, color=INK_MUTED))
    ax.text(5.1, 0.10, "solid = executed in this study   ·   dashed = not executed here.   "
            "The dataset came from [13] and was audited, not regenerated; stage 2 ran in "
            "Octave, NumPy and C, not MATLAB Coder;\nstage 3 is out of scope for this paper "
            "(Section 8); stage 4 ran on a network calibrated from the measurements, not on "
            "the model that produced them, and not in ETAP.",
            ha="center", fontsize=7.2, color=INK_MUTED, style="italic",
            linespacing=1.5)
    save(fig, "fig01_verification_chain.png")


# ---- Fig 2: ANN architecture ---------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.axis("off"); ax.set_xlim(0, 8.4); ax.set_ylim(0, 3.5)
    layers = [("Input", "24", CAT[5], ""), ("Dense", "64", CAT[0], "ReLU"),
              ("Dropout", "0.2", GRID, ""), ("Dense", "32", CAT[0], "ReLU"),
              ("Dropout", "0.2", GRID, ""), ("Dense", "16", CAT[0], "ReLU"),
              ("Output", "4", CAT[1], "sigmoid")]
    w, gap = 0.92, 0.26
    for i, (t, n, c, a) in enumerate(layers):
        x = 0.25 + i * (w + gap)
        hh = 0.55 + 1.55 * (int(n) / 64) ** 0.42 if n not in ("0.2",) else 0.5
        y = 1.95 - hh / 2
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.04",
                                    facecolor=c, edgecolor=c, linewidth=1.4,
                                    alpha=0.35 if t == "Dropout" else 1.0))
        ax.text(x + w / 2, y + hh / 2, n, ha="center", va="center", fontsize=11,
                fontweight="bold", color=INK if t == "Dropout" else "white")
        ax.text(x + w / 2, 3.12, t, ha="center", fontsize=8, fontweight="bold",
                color=INK)
        if a:
            ax.text(x + w / 2, y - 0.2, a, ha="center", fontsize=7.4,
                    color=INK_MUTED, style="italic")
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.95),
                                         (x + w + gap - 0.02, 1.95),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.3, color=INK_MUTED))
    ax.text(4.2, 0.42, "He-normal initialisation · Adam (lr = 0.001) · binary "
            "cross-entropy · batch 32 · max 100 epochs\nearly stopping "
            "(patience 10, restore best) · ReduceLROnPlateau (factor 0.5, "
            "patience 5) · 4,276 trainable parameters",
            ha="center", fontsize=7.6, color=INK_MUTED)
    save(fig, "fig02_ann_architecture.png")


# ---- Fig 3: label-vector distribution ------------------------------------
def fig3():
    vc = vec.value_counts()
    order = ["0000", "1001", "0110", "1011", "0111", "1111"]
    vals = [int(vc[o]) for o in order]
    lbl = [f"{o}\n{VEC2CLASS[o]}" for o in order]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(range(6), vals, color=CAT[0], width=0.62, zorder=3)
    bars[0].set_color(CAT[2])
    for i, v in enumerate(vals):
        ax.text(i, v + 45, f"{v:,}", ha="center", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel("Samples"); ax.set_ylim(0, max(vals) * 1.16)
    ax.set_xlabel("Label vector [G C B A]  and resolved fault class")
    ax.set_title("Label-vector distribution, classData.csv (n = 7,861)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.text(0.99, 0.96, "green = healthy   ·   blue = faulted", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.6, color=INK_MUTED)
    save(fig, "fig03_label_distribution.png")


# ---- Fig 4: correlation matrix -------------------------------------------
def fig4():
    C = cd[MEAS].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(MEAS)
    ax.set_yticks(range(6)); ax.set_yticklabels(MEAS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C[i, j]) > 0.55 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=8); cb.outline.set_visible(False)
    ax.set_title("Correlation of measured channels", loc="left")
    save(fig, "fig04_correlation.png")


# ---- Fig 5: training / validation loss -----------------------------------
def fig5():
    h = pd.read_csv(os.path.join(OUT, "history_A.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(h.index + 1, h["loss"], color=CAT[0], label="Training loss")
    ax.plot(h.index + 1, h["val_loss"], color=CAT[1], label="Validation loss")
    k = int(h["val_loss"].idxmin())
    ax.scatter([k + 1], [h["val_loss"].iloc[k]], s=52, color=CAT[1], zorder=5,
               edgecolor="white", linewidth=1.6)
    ax.annotate(f"best val_loss = {h['val_loss'].iloc[k]:.4f}\n(epoch {k+1})",
                (k + 1, h["val_loss"].iloc[k]), textcoords="offset points",
                xytext=(14, 26), fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9))
    ax.text(len(h) * 0.97, h["loss"].iloc[-1], " Training", color=CAT[0],
            fontsize=8, va="center", fontweight="bold")
    ax.text(len(h) * 0.97, h["val_loss"].iloc[-1], " Validation", color=CAT[1],
            fontsize=8, va="center", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("Multi-label ANN training history (exact Keras model)", loc="left")
    ax.legend(loc="upper right")
    save(fig, "fig05_loss_curve.png")


# ---- Fig 6: ROC for binary detection -------------------------------------
def fig6():
    from sklearn.metrics import roc_curve, roc_auc_score
    d = pd.read_csv(os.path.join(OUT, "detect_test_scores.csv"))
    fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
    auc = roc_auc_score(d.y_true, d.y_prob)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, linestyle=(0, (4, 3)), linewidth=1.4,
            label="Chance")
    ax.plot(fpr, tpr, color=CAT[0], label=f"ANN detector (AUC = {auc:.6f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Binary fault detection ROC (exact Keras, n = 2,401 test)", loc="left")
    ax.legend(loc="lower right")
    axin = ax.inset_axes([0.42, 0.18, 0.5, 0.42])
    axin.plot(fpr, tpr, color=CAT[0]); axin.set_xlim(0, 0.03); axin.set_ylim(0.97, 1.001)
    axin.tick_params(labelsize=6.5); axin.grid(alpha=0.5)
    axin.set_title("detail", fontsize=7, color=INK_MUTED, fontweight="normal")
    save(fig, "fig06_roc_detection.png")


# ---- Fig 7/8: confusion matrices ----------------------------------------
def _cm(path, title, fname, note=None):
    M = np.loadtxt(os.path.join(OUT, path), delimiter=",", dtype=int)
    Mn = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_ORDER, rotation=32, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_ORDER)
    for i in range(6):
        for j in range(6):
            if M[i, j]:
                ax.text(j, i, f"{M[i,j]}\n{Mn[i,j]*100:.1f}%", ha="center",
                        va="center", fontsize=7.4,
                        color="white" if Mn[i, j] > 0.5 else INK)
    ax.grid(False)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised share", fontsize=8); cb.outline.set_visible(False)
    if note:
        ax.text(0, -0.30, note, transform=ax.transAxes, fontsize=7.4,
                color=INK_MUTED, va="top")
    save(fig, fname)


def fig7():
    _cm("cm_D.csv", "Six-class confusion matrix (exact Keras, MinMax as specified)",
        "fig07_cm_sixclass.png",
        "90.7% of all errors are LLL<->LLLG confusions; mechanism in Section 6.4.")


def fig8():
    _cm("cm_E.csv", "Six-class confusion matrix after SMOTE balancing",
        "fig09_cm_smote.png",
        "SMOTE does not recover the LLLG class: 85.25% vs 85.57% unbalanced.")


# ---- Fig 9: per-class metrics -------------------------------------------
def fig9():
    d = pd.read_csv(os.path.join(OUT, "per_class_D.csv"))
    x = np.arange(len(d)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for k, (col, c) in enumerate(zip(["precision", "recall", "f1"], CAT[:3])):
        ax.bar(x + (k - 1) * w, d[col], width=w - 0.03, color=c,
               label=col.capitalize(), zorder=3)
        for xi, v in zip(x + (k - 1) * w, d[col]):
            ax.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(d["cls"], fontsize=8)
    ax.set_ylabel("Per cent"); ax.set_ylim(0, 114)
    ax.set_title("Per-class performance, six-class model (exact Keras)",
                 loc="left", pad=26)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=7.8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig08_per_class.png")


# ---- Fig 10: zero-sequence separability (key diagnostic) -----------------
def fig10():
    I0 = (cd[["Ia", "Ib", "Ic"]].sum(axis=1) / 3).abs()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.2, 3.9))
    order = ["0000", "1001", "0110", "1011", "0111", "1111"]
    data = [I0[vec == o].to_numpy() for o in order]
    bp = a1.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    whis=(1, 99), medianprops=dict(color=INK, linewidth=1.4))
    for p, c in zip(bp["boxes"], CAT):
        p.set_facecolor(c); p.set_alpha(0.9); p.set_edgecolor(c)
    a1.set_yscale("symlog", linthresh=0.5)
    a1.set_xticklabels([VEC2CLASS[o] for o in order], rotation=32, ha="right",
                       fontsize=7.6)
    a1.set_ylabel(r"$|I_0|$  (A, symlog;  whiskers = 1st-99th pct)")
    a1.set_title("Zero-sequence current by class", loc="left")
    a1.grid(axis="x", visible=False)
    a1.axhspan(-0.5, 1.2, color=CAT[3], alpha=0.10, zorder=0)
    a1.text(0.98, 0.06, "shaded: ungrounded classes,\n"
            r"$|I_0|$ pinned near zero", transform=a1.transAxes,
            ha="right", fontsize=7, color=INK_MUTED, style="italic")

    # ECDF: both series fully visible, no occlusion.
    for v, c, lab in [("0111", CAT[0], "LLL (0111)"), ("1111", CAT[1], "LLLG (1111)")]:
        s = np.sort(I0[vec == v].to_numpy())
        a2.step(s, np.arange(1, s.size + 1) / s.size * 100, where="post",
                color=c, label=lab)
    a2.set_xscale("symlog", linthresh=1)
    a2.set_xlim(-0.05, 300); a2.set_ylim(0, 103)
    a2.set_xlabel(r"$|I_0|$  (A, symlog)")
    a2.set_ylabel("Cumulative share of samples (%)")
    a2.set_title("LLL vs LLLG: empirical CDF", loc="left")
    a2.legend(loc="lower right")
    lo, hi = I0[vec == "0111"].min(), I0[vec == "0111"].max()
    a2.axvspan(lo, hi, color=CAT[0], alpha=0.18, zorder=0)
    a2.text(0.04, 0.60, "LLL occupies a 0.067 A band\n"
            r"$|I_0|\in[0.663,\,0.730]$ A (shaded)."
            "\nNo LLLG sample falls inside it;\n88.8% fall below it.\n"
            "That band is 1.0e-4 of the\nMinMax-scaled feature axis.",
            transform=a2.transAxes, fontsize=7.0, color=INK_MUTED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF",
                      edgecolor=GRID))

    # Panel C: within-class dispersion -- evidence of fixed-parameter runs.
    cv = [float(I0[vec == o].std() / max(I0[vec == o].mean(), 1e-12)) for o in order]
    cols = [CAT[2] if c < 0.1 else CAT[0] for c in cv]
    bars = a3.bar(range(6), cv, color=cols, width=0.62, zorder=3)
    for i, v in enumerate(cv):
        a3.text(i, v + 0.07, f"{v:.3f}", ha="center", fontsize=7.4,
                fontweight="bold", color=INK)
    a3.set_xticks(range(6))
    a3.set_xticklabels([VEC2CLASS[o] for o in order], rotation=32, ha="right",
                       fontsize=7.6)
    a3.set_ylabel(r"Coefficient of variation of $|I_0|$")
    a3.set_ylim(0, max(cv) * 1.30)
    a3.set_title("Within-class dispersion", loc="left")
    a3.grid(axis="x", visible=False)
    a3.text(0.03, 0.97, "green = the two ungrounded FAULT\nclasses, whose residual current is\n"
            "pinned near-constant across the run",
            transform=a3.transAxes, ha="left", va="top", fontsize=7.0,
            color=INK_MUTED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF",
                      edgecolor=GRID))
    save(fig, "fig10_zero_sequence.png")


# ---- Fig 11: preprocessing ablations ------------------------------------
def fig11():
    d1 = pd.read_csv(os.path.join(OUT, "scaling_ablation.csv"))
    d2 = pd.read_csv(os.path.join(OUT, "clipping_ablation.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.0),
                                 gridspec_kw=dict(width_ratios=[1, 1.18]))

    def panel(ax, d, labels, title, base_idx):
        x = np.arange(len(d)); w = 0.38
        ax.bar(x - w/2, d["accuracy"], w - 0.03, color=CAT[0],
               label="Six-class accuracy", zorder=3)
        ax.bar(x + w/2, d["LLLG_recall"], w - 0.03, color=CAT[1],
               label="LLLG recall", zorder=3)
        for xi, v in zip(x - w/2, d["accuracy"]):
            ax.text(xi, v + 2.0, f"{v:.1f}", ha="center", fontsize=7.0,
                    fontweight="bold", color=INK)
        for xi, v in zip(x + w/2, d["LLLG_recall"]):
            ax.text(xi, v + 2.0, f"{v:.1f}", ha="center", fontsize=7.0, color=INK)
        ax.axvspan(base_idx - 0.5, base_idx + 0.5, color=CAT[3], alpha=0.12, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2)
        ax.set_ylabel("Per cent"); ax.set_ylim(0, 116)
        ax.set_title(title, loc="left", fontsize=9, pad=26)
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=7.6)
        ax.grid(axis="x", visible=False)

    panel(a1, d1, ["MinMax\n(as specified)", "Standard", "Robust",
                   "signed-log\n+ MinMax"],
          "(a) Feature scaler varied", 0)
    panel(a2, d2, ["A\nclip raw\n+MinMax", "B\nno clip\n+MinMax",
                   "C\nclip feats\n+MinMax", "D\nno clip\n+Robust",
                   "E\nclip feats\n+Robust"],
          "(b) Percentile clipping varied", 0)
    a1.text(0.02, 0.93, "shaded = the pipeline as specified", transform=a1.transAxes,
            fontsize=6.8, color=INK_MUTED, style="italic")
    save(fig, "fig11_preprocessing_ablation.png")



# ---- Proteus circuit schematic (specification, not a measurement) --------
def fig_proteus():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axis("off"); ax.set_xlim(0, 9.6); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, label, sub, c, fs=8.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    facecolor="#FFFFFF", edgecolor=c, linewidth=1.5))
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=c)
        if sub:
            ax.text(x + w/2, y + h*0.26, sub, ha="center", va="center",
                    fontsize=6.8, color=INK_MUTED)

    def arrow(x1, y1, x2, y2, c=INK_MUTED, lbl=None, dy=0.13):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.3, color=c))
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2 + dy, lbl, ha="center", fontsize=6.6,
                    color=INK_MUTED, fontweight="bold")

    # sensors
    box(0.15, 3.20, 1.75, 0.80, "ACS712 x3", "current Ia, Ib, Ic", CAT[0])
    box(0.15, 2.15, 1.75, 0.80, "ZMPT101B x3", "voltage Va, Vb, Vc", CAT[0])
    # MCU
    box(2.85, 1.85, 2.35, 2.15, "Arduino Uno", "ATmega328P\n7,350 B flash / 794 B SRAM\n(measured, Table 7)", CAT[2], fs=9.2)
    # outputs
    box(6.30, 3.20, 1.60, 0.80, "4 x LED", "G, C, B, A flags", CAT[3])
    box(6.30, 2.15, 1.60, 0.80, "ULN2003", "relay driver", CAT[3])
    box(8.25, 2.15, 1.20, 0.80, "5 V SPDT", "trip relay", CAT[1])
    # serial
    box(2.85, 0.55, 2.35, 0.80, "COMPIM (RS232)", "9600 8N1 -> VSPE COM1", CAT[4])

    arrow(1.90, 3.60, 2.85, 3.35, lbl="A0-A2")
    arrow(1.90, 2.55, 2.85, 2.60, lbl="A3-A5")
    arrow(5.20, 3.60, 6.30, 3.60, lbl="D8-D11")
    arrow(5.20, 2.55, 6.30, 2.55, lbl="D12")
    arrow(7.90, 2.55, 8.25, 2.55)
    arrow(4.02, 1.85, 4.02, 1.35)
    ax.text(4.15, 1.60, "D0/D1 (RX/TX)", fontsize=6.6, color=INK_MUTED,
            va="center", ha="left", fontweight="bold")

    ax.text(4.8, 0.18, "Circuit specified by the design specification [12]; component values and pin "
            "mapping as listed. Not an executed simulation.",
            ha="center", fontsize=7.2, color=INK_MUTED, style="italic")
    ax.add_patch(FancyBboxPatch((0.05, 1.70), 9.5, 2.55, boxstyle="round,pad=0.06",
                                facecolor="none", edgecolor=GRID, linewidth=1.0,
                                linestyle=(0, (5, 4))))
    ax.text(0.18, 4.32, "Sensor front end", fontsize=7.4, color=INK_MUTED, fontweight="bold")
    save(fig, "fig13_proteus_circuit.png")


# ---- MATLAB/host - Proteus co-simulation loop ---------------------------
def fig_cosim():
    fig, ax = plt.subplots(figsize=(9.6, 3.1))
    ax.axis("off"); ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.1)
    st = [("Host script", "writes 24 scaled\nfeatures to COM2", CAT[0]),
          ("VSPE", "virtual pair\nCOM2 <-> COM1", CAT[4]),
          ("COMPIM", "Proteus serial\nendpoint", CAT[3]),
          ("ATmega328P", "ann_predict()\n4,160 MACs", CAT[2]),
          ("Trip relay", "D12 asserts;\nflags on D8-D11", CAT[1])]
    w, gap = 1.62, 0.34
    for i, (t, s, c) in enumerate(st):
        x = 0.22 + i*(w+gap)
        ax.add_patch(FancyBboxPatch((x, 1.15), w, 1.15, boxstyle="round,pad=0.05",
                                    facecolor="#FFFFFF", edgecolor=c, linewidth=1.6))
        ax.text(x+w/2, 1.95, t, ha="center", fontsize=8.6, fontweight="bold", color=c)
        ax.text(x+w/2, 1.48, s, ha="center", fontsize=6.9, color=INK_MUTED, linespacing=1.4)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x+w+0.02, 1.72), (x+w+gap-0.02, 1.72),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.4, color=INK_MUTED))
    # timing annotation
    ax.annotate("", xy=(0.22, 0.80), xytext=(9.38, 0.80),
                arrowprops=dict(arrowstyle="<->", color=CAT[1], linewidth=1.3))
    ax.text(4.8, 0.55, "t_trip  =  t_relay,on  -  t_fault,inj  -  t_serial     (Eq. 22)",
            ha="center", fontsize=8.0, color=CAT[1], fontweight="bold")
    ax.text(4.8, 0.22, "The serial term must be measured separately and removed; the specification "
            "notes 1-5 ms for VSPE [12].",
            ha="center", fontsize=7.0, color=INK_MUTED, style="italic")
    ax.text(4.8, 2.62, "Co-simulation loop as specified. No latency has been measured.",
            ha="center", fontsize=7.4, color=INK_MUTED, style="italic")
    save(fig, "fig14_cosimulation.png")


# ---- ETAP one-line (proposed topology; parameters unconfirmed) ----------
def fig_etap():
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.axis("off"); ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.9)

    def gen(x, y, lbl):
        ax.add_patch(plt.Circle((x, y), 0.26, facecolor="#FFFFFF",
                                edgecolor=CAT[0], linewidth=1.5))
        ax.text(x, y, "G", ha="center", va="center", fontsize=9,
                fontweight="bold", color=CAT[0])
        ax.text(x, y-0.44, lbl, ha="center", fontsize=6.6, color=INK_MUTED)

    for i, x in enumerate([0.55, 1.35, 2.15, 2.95]):
        gen(x, 3.05, "11 kV")
        ax.plot([x, x], [2.79, 2.35], color=INK_MUTED, linewidth=1.2)
    ax.plot([0.55, 2.95], [2.35, 2.35], color=INK, linewidth=2.4)
    ax.text(1.75, 2.12, "11 kV bus", ha="center", fontsize=7, color=INK_MUTED)

    def xfmr(x, y, lbl):
        ax.add_patch(plt.Circle((x-0.13, y), 0.20, facecolor="none", edgecolor=CAT[3], linewidth=1.5))
        ax.add_patch(plt.Circle((x+0.13, y), 0.20, facecolor="none", edgecolor=CAT[3], linewidth=1.5))
        ax.text(x, y-0.42, lbl, ha="center", fontsize=6.6, color=INK_MUTED)

    ax.plot([1.75, 1.75], [2.35, 1.72], color=INK_MUTED, linewidth=1.4)
    xfmr(1.75, 1.50, "step-up")
    ax.plot([1.75, 3.10], [1.28, 1.28], color=INK_MUTED, linewidth=1.4)

    # line with relays
    ax.plot([3.10, 6.50], [1.28, 1.28], color=INK, linewidth=2.6)
    ax.text(4.80, 1.52, "transmission line", ha="center", fontsize=7.4,
            fontweight="bold", color=INK)
    ax.text(4.80, 1.02, "220 kV, 100 km  [parameters unconfirmed]", ha="center",
            fontsize=6.9, color="#B25400", style="italic")
    for x, side in [(3.35, "left"), (6.25, "right")]:
        ax.add_patch(FancyBboxPatch((x-0.30, 1.62), 0.60, 0.38,
                                    boxstyle="round,pad=0.03", facecolor="#FFFFFF",
                                    edgecolor=CAT[1], linewidth=1.4))
        ax.text(x, 1.81, "21", ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=CAT[1])
        ax.text(x, 2.12, "distance", ha="center", fontsize=6.4, color=INK_MUTED)
        ax.plot([x, x], [1.62, 1.28], color=CAT[1], linewidth=1.0, linestyle=(0,(2,2)))
    ax.add_patch(FancyBboxPatch((4.20, 0.42), 1.20, 0.40, boxstyle="round,pad=0.03",
                                facecolor="#FFFFFF", edgecolor=CAT[4], linewidth=1.4))
    ax.text(4.80, 0.62, "50/51 · 51G", ha="center", va="center", fontsize=7.4,
            fontweight="bold", color=CAT[4])
    ax.plot([4.80, 4.80], [0.82, 1.28], color=CAT[4], linewidth=1.0, linestyle=(0,(2,2)))

    ax.plot([6.50, 7.30], [1.28, 1.28], color=INK_MUTED, linewidth=1.4)
    xfmr(7.52, 1.28, "step-down")
    ax.plot([7.74, 8.60], [1.28, 1.28], color=INK_MUTED, linewidth=1.4)
    ax.plot([8.60, 8.60], [1.28, 0.95], color=INK_MUTED, linewidth=1.2)
    ax.add_patch(FancyBboxPatch((8.25, 0.55), 0.70, 0.40, boxstyle="round,pad=0.03",
                                facecolor="#FFFFFF", edgecolor=CAT[2], linewidth=1.4))
    ax.text(8.60, 0.75, "load", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=CAT[2])

    ax.text(7.55, 3.34, "PENDING EXTERNAL EXECUTION", ha="center", va="center",
            fontsize=9, color="#7A3000", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FDF3EC",
                      edgecolor="#D55E00", linewidth=1.4))
    ax.text(4.8, 0.10, "Topology proposed by the design specification [12]. Source impedance, line "
            "impedance, CT/VT ratios and relay settings are absent from the supplied data, so no "
            "coordination study can be run.",
            ha="center", fontsize=6.9, color=INK_MUTED, style="italic")
    save(fig, "fig13_oneline.png")


# ---- pending external stages --------------------------------
def pending(fname, title, need, axes_lbl):
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.set_xlabel(axes_lbl[0]); ax.set_ylabel(axes_lbl[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor="#D55E00", linewidth=1.6,
                           linestyle=(0, (6, 4))))
    pending_banner(ax, "PENDING EXTERNAL EXECUTION")
    ax.text(0.5, 0.25, need, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.6, color=INK_MUTED)
    ax.set_title(title, loc="left")
    save(fig, fname)


def fig12():
    """Phase 2 results -- measured, no longer a placeholder."""
    ag = pd.read_csv(os.path.join(OUT, "phase2", "agreement.csv"))
    fp = pd.read_csv(os.path.join(OUT, "phase2", "footprint_atmega328p.csv"))
    sf = np.load(os.path.join(OUT, "phase2", "scores_f32.npy"))
    sq = np.load(os.path.join(OUT, "phase2", "scores_int8.npy"))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.2, 3.9),
                                     gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

    # (a) agreement of every implementation with Keras
    y = np.arange(len(ag))[::-1]
    cols = [CAT[2] if v == 0 else CAT[3] for v in ag["D_pct"]]
    a1.barh(y, np.maximum(ag["D_pct"], 0.004), color=cols, height=0.62, zorder=3)
    for yi, v in zip(y, ag["D_pct"]):
        a1.text(max(v, 0.004) + 0.06, yi, f"{v:.6f} %" if v else "0.000000 %",
                va="center", fontsize=7.4, fontweight="bold", color=INK)
    a1.axvline(0.1, color=CAT[1], linestyle=(0, (4, 3)), linewidth=1.5, zorder=4)
    a1.text(0.16, 1.9, "acceptance criterion\nD < 0.1 %  (Eq. 20)", fontsize=7,
            color=CAT[1], va="center", ha="left")
    a1.set_yticks(y); a1.set_yticklabels(ag["implementation"], fontsize=7.6)
    a1.set_xlim(0, 2.6); a1.set_xlabel("Discrepancy D vs Keras (%), Eq. (20)")
    a1.set_title("(a) Implementation agreement", loc="left", fontsize=9)
    a1.grid(axis="y", visible=False)

    # (b) what int8 quantisation does to the sigmoid scores
    a2.plot([0, 1], [0, 1], color=GRID, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    a2.scatter(sf.ravel(), sq.ravel(), s=3, color=CAT[0], alpha=0.18,
               edgecolors="none", zorder=2)
    a2.axhline(0.5, color=INK_MUTED, linewidth=0.8, alpha=0.6)
    a2.axvline(0.5, color=INK_MUTED, linewidth=0.8, alpha=0.6)
    flip = int(np.sum((sf > 0.5) != (sq > 0.5)))
    a2.set_xlabel("float32 sigmoid score"); a2.set_ylabel("int8 sigmoid score")
    a2.set_xlim(-0.02, 1.02); a2.set_ylim(-0.02, 1.02)
    a2.set_title("(b) Effect of int8 quantisation", loc="left", fontsize=9)
    a2.text(0.04, 0.93, f"{flip} of {sf.size:,} outputs\ncross the 0.5 threshold\n"
            f"({flip/sf.size*100:.3f} %)\nexact-match loss 0.25 pp",
            transform=a2.transAxes, fontsize=7.0, color=INK_MUTED, va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor=GRID))

    # (c) measured ATmega328P footprint
    x = np.arange(2); w = 0.36
    a3.bar(x - w/2, fp["flash_pct"], w - 0.03, color=CAT[0], label="Flash (32 KiB)", zorder=3)
    a3.bar(x + w/2, fp["sram_pct"], w - 0.03, color=CAT[4], label="SRAM (2 KiB)", zorder=3)
    for xi, v, b in zip(x - w/2, fp["flash_pct"], fp["flash_B"]):
        a3.text(xi, v + 2.2, f"{b:,} B\n{v:.1f} %", ha="center", fontsize=6.8, color=INK)
    for xi, v, b in zip(x + w/2, fp["sram_pct"], fp["sram_B"]):
        a3.text(xi, v + 2.2, f"{b:,} B\n{v:.1f} %", ha="center", fontsize=6.8, color=INK)
    a3.axhline(100, color=CAT[1], linestyle=(0, (4, 3)), linewidth=1.4)
    a3.text(1.42, 101, "device limit", fontsize=6.8, color=CAT[1], ha="right")
    a3.set_xticks(x); a3.set_xticklabels(["int8 weights", "float32 weights"], fontsize=8)
    a3.set_ylabel("Share of device resource (%)"); a3.set_ylim(0, 118)
    a3.set_title("(c) Measured ATmega328P footprint", loc="left", fontsize=9, pad=26)
    a3.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=7.4)
    a3.grid(axis="x", visible=False)
    save(fig, "fig12_phase2_results.png")


def figs_pending():
    pending("fig14_tcc_coordination.png",
            "Time-current coordination (TCC) curves",
            "Requires ETAP 19 Star Protection & Coordination.\n"
            "Export TCC curves for relays 21, 50/51 and 51G and verify\n"
            "the coordination time margin of Eq. (23).",
            ("Current (A, log)", "Operating time (s, log)"))
    pending("fig15_operating_time.png",
            "Trip-time comparison: conventional vs ANN",
            "Requires measured ETAP relay operating times AND measured\n"
            "Proteus trip latency per Eq. (22). No comparative improvement\n"
            "is claimed until both are supplied.",
            ("Fault scenario", "Trip time (ms)"))
    pending("fig16_sensitivity.png",
            "Sensitivity to fault resistance and location",
            "Requires a regenerated dataset sweeping Rf = 0, 5, 20, 50 ohm\n"
            "and fault location 10-90% of line length, per Eq. (24).",
            ("Fault resistance / location", "Classification accuracy (%)"))


if __name__ == "__main__":
    print("building figures ->", FIG)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10,
              fig11, fig_proteus, fig_cosim, fig_etap, fig12, figs_pending):
        f()
    print("done")
