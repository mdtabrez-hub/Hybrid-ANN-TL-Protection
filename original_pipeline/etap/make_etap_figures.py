"""
make_etap_figures.py -- Figs 16-18 from the executed coordination study.

Replaces the three PENDING placeholders. Every value plotted comes from
etap/results/*.csv; nothing here is drawn from a specification.

Form choices, per the visualisation method:
  Fig 16a  identity (four relay elements)      -> categorical hues, fixed order
  Fig 16b  ordered magnitude (fault resistance)-> sequential, one hue light->dark
  Fig 17a  ordered magnitude (fault resistance)-> sequential
  Fig 17b  identity (which element cleared)    -> categorical, 2 pt surface gaps
  Fig 18a  magnitude (operating time)          -> sequential
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from plotstyle import apply, CAT, SEQ, INK, INK_MUTED, SURFACE, GRID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relays import iec_time, T_RELAY

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
os.makedirs(FIGS, exist_ok=True)

apply()
C = pd.read_csv(os.path.join(RES, "coordination.csv"))
S = pd.read_csv(os.path.join(RES, "sc_matrix.csv"))
SET = json.load(open(os.path.join(RES, "settings.json")))

RF = sorted(C.rf.unique())
RF_COL = {rf: SEQ(0.38 + 0.58 * i / (len(RF) - 1)) for i, rf in enumerate(RF)}
ELEM = ["21", "51", "51G"]
ELEM_COL = dict(zip(ELEM, CAT[:3]))
ELEM_LBL = {"21": "21 distance", "51": "51 overcurrent", "51G": "51G earth fault"}


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=400)
    plt.close(fig)
    print("wrote", name, os.path.getsize(p) // 1024, "kB")


# ============================================================== Fig 16
def fig16():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.1))

    # ---- (a) time-current curves
    series = [
        ("51 line", SET["oc_phase"]["pickup_A"], SET["oc_phase"]["tms_line"], CAT[0], "-"),
        ("51 transformer backup", SET["oc_phase"]["pickup_A"], SET["oc_phase"]["tms_transformer"], CAT[1], "-"),
        ("51G line", SET["oc_earth"]["pickup_A"], SET["oc_earth"]["tms_line"], CAT[2], "--"),
        ("51G transformer backup", SET["oc_earth"]["pickup_A"], SET["oc_earth"]["tms_transformer"], CAT[3], "--"),
    ]
    lo, hi = C.Ir_A.min(), C.Ir_A.max()
    ax.axvspan(lo, hi, color=CAT[0], alpha=0.09, lw=0, zorder=0)
    ax.text(np.sqrt(lo * hi), 17, "fault current\nseen by relay A",
            ha="center", va="center", fontsize=7.2, color=INK_MUTED)

    for nm, Is, tms, col, ls in series:
        Iq = np.logspace(np.log10(Is * 1.06), np.log10(3000), 400)
        t = np.array([iec_time(i, Is, tms) for i in Iq])
        ax.plot(Iq, t, color=col, ls=ls, lw=2.0, label=nm, zorder=3)

    inst = SET["oc_phase"]["instantaneous_A"]
    ax.plot([inst, inst, 3000], [30, T_RELAY, T_RELAY], color=CAT[4], lw=2.0,
            ls=":", label="50 instantaneous", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(110, 3400); ax.set_ylim(0.01, 30)
    ax.set_xlabel("Current in relay A (A)"); ax.set_ylabel("Operating time (s)")
    ax.set_title("(a) Time–current characteristics", loc="left")
    ax.legend(loc="lower left", fontsize=7.2, ncol=1)
    ax.set_xticks([200, 300, 500, 800, 1200, 2000, 3000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    # ---- (b) impedance plane
    z = SET["distance"]
    zang = np.deg2rad(z["angle_deg"])
    for i in (2, 1, 0):
        X, R = z["zone_X_ohm"][i], z["zone_R_ohm"][i]
        # quadrilateral: reactance line at X, resistive blinders at +/- R,
        # lower edge along the line angle
        pts = [(-R, -R / np.tan(zang)), (R, R / np.tan(zang)),
               (R, X), (-R, X)]
        bx.add_patch(Polygon(pts, closed=True, facecolor=CAT[0],
                             alpha=0.055 * (3 - i), edgecolor=CAT[0],
                             lw=1.1, ls=["-", "--", ":"][i], zorder=1))
        lbl = "inst." if z["zone_time_s"][i] == 0 else f"{z['zone_time_s'][i]:.2f} s"
        bx.text(-12.0, X - 1.8, f"Z{i+1} · {lbl}",
                fontsize=7.2, color=CAT[0], ha="left", va="top",
                fontweight="bold", zorder=6)

    zl = z["z_line_ohm"]
    bx.plot([0, zl * np.cos(zang)], [0, zl * np.sin(zang)], color=INK,
            lw=1.8, zorder=4)
    bx.annotate("protected line", (zl * np.cos(zang), zl * np.sin(zang)),
                xytext=(9, 5), textcoords="offset points", fontsize=7.2,
                color=INK, ha="left", va="bottom", fontweight="bold", zorder=6)

    for rf in RF:
        g = C[C.rf == rf]
        bx.scatter(g.Zapp_R, g.Zapp_X, s=27, color=RF_COL[rf],
                   edgecolor=SURFACE, linewidth=0.7, zorder=5,
                   label=f"{rf:.0f} Ω")
    bx.set_xlim(-13, 130); bx.set_ylim(-8, 92)
    bx.set_xlabel("Apparent resistance (Ω)"); bx.set_ylabel("Apparent reactance (Ω)")
    bx.set_title("(b) Apparent impedance against the zone reach", loc="left")
    bx.legend(title="$R_f$", loc="upper right", fontsize=7.2, title_fontsize=7.4)
    bx.annotate("high-resistance faults leave every zone",
                xy=(104, 20), xytext=(104, 62), fontsize=7.2, color=INK_MUTED,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.0))
    fig.tight_layout()
    save(fig, "fig14_tcc_coordination.png")


# ============================================================== Fig 17
def fig17():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.5),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))

    # ---- (a) operating time against fault position
    fr = sorted(C.frac.unique())
    med = {rf: C[C.rf == rf].groupby("frac").t_primary_s.median() for rf in RF}
    # R_f = 0 and 5 ohm give identical medians; draw the first as a wide halo so
    # both remain visible rather than one hiding the other.
    groups = []                                   # coincident curves share a label
    for rf in RF:
        for g in groups:
            if np.allclose(med[rf].values, med[g[0]].values):
                g.append(rf); break
        else:
            groups.append([rf])
    for grp in groups:
        for j, rf in enumerate(grp):
            g = med[rf]
            wide = len(grp) > 1 and j == 0
            ax.plot([f * 100 for f in g.index], g.values, marker="o",
                    color=RF_COL[rf], lw=5.0 if wide else 2.0,
                    markersize=9 if wide else 6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=f"$R_f$ = {rf:.0f} $\\Omega$", zorder=2 if wide else 3)
        lbl = " and ".join(f"{rf:.0f}" for rf in grp) + " Ω"
        ax.annotate(lbl, (fr[-1] * 100, med[grp[-1]].values[-1]),
                    xytext=(8, 0), textcoords="offset points", fontsize=7.4,
                    color=RF_COL[grp[-1]], va="center", fontweight="bold")
    ax.set_xlabel("Fault position (% of line length from relay A)")
    ax.set_ylabel("Operating time of the fastest element (s)")
    ax.set_xlim(2, 112); ax.set_ylim(0, 0.72)
    ax.set_title("(a) Median operating time across the study matrix", loc="left")
    ax.legend(loc="upper left", fontsize=7.4, ncol=2)

    # ---- (b) which element clears the fault
    bot = np.zeros(len(RF))
    x = np.arange(len(RF))
    for e in ELEM:
        share = [100 * ((C.rf == rf) & (C.primary == e)).sum() / (C.rf == rf).sum()
                 for rf in RF]
        bx.bar(x, share, bottom=bot, width=0.66, color=ELEM_COL[e],
               edgecolor=SURFACE, linewidth=1.4, label=ELEM_LBL[e], zorder=3)
        for xi, (s, b) in enumerate(zip(share, bot)):
            if s >= 8:
                bx.text(xi, b + s / 2, f"{s:.0f}%", ha="center", va="center",
                        fontsize=7.6, color=SURFACE, fontweight="bold")
        bot += np.array(share)
    bx.set_xticks(x); bx.set_xticklabels([f"{rf:.0f}" for rf in RF])
    bx.set_xlabel("Fault resistance $R_f$ (Ω)")
    bx.set_ylabel("Share of the 50 cases (%)")
    bx.set_ylim(0, 108)
    bx.set_title("(b) Element that clears the fault", loc="left")
    bx.legend(loc="upper center", fontsize=7.4, ncol=3,
              bbox_to_anchor=(0.5, 1.02))
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig15_operating_time.png")


# ============================================================== Fig 18
def fig18():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2),
                             gridspec_kw=dict(width_ratios=[1.12, 1, 1]))
    ax, bx, cx = axes

    # ---- (a) operating-time surface
    fr = sorted(C.frac.unique())
    M = np.array([[C[(C.rf == rf) & (C.frac == f)].t_primary_s.median()
                   for f in fr] for rf in RF])
    im = ax.imshow(M, cmap=SEQ, aspect="auto", origin="lower",
                   vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(fr))); ax.set_xticklabels([f"{f*100:.0f}" for f in fr])
    ax.set_yticks(range(len(RF))); ax.set_yticklabels([f"{r:.0f}" for r in RF])
    ax.set_xlabel("Fault position (% of line)"); ax.set_ylabel("$R_f$ (Ω)")
    ax.set_title("(a) Median operating time (s)", loc="left")
    for i in range(len(RF)):
        for j in range(len(fr)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color=SURFACE if v > 0.72 * np.nanmax(M) else INK,
                    fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)

    # ---- (b) zone-1 coverage against the resistive blinder
    cov = pd.read_csv(os.path.join(RES, "zone1_sweep.csv"))
    for rf in RF:
        g = cov[cov.rf == rf]
        bx.plot(g.r_reach, g.pct, marker="o", color=RF_COL[rf], lw=2.0,
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=f"{rf:.0f} Ω", zorder=3)
    bx.axvline(SET["distance"]["zone_R_ohm"][0], color=INK_MUTED, lw=1.0, ls="--",
               zorder=2)
    bx.text(SET["distance"]["zone_R_ohm"][0] + 1.5, 52, "setting used",
            fontsize=7, color=INK_MUTED, rotation=90, va="center")
    bx.set_xlabel("Zone-1 resistive reach (Ω)")
    bx.set_ylabel("Faults inside Zone 1 (%)")
    bx.set_ylim(-4, 108)
    bx.set_title("(b) Zone-1 coverage against the blinder", loc="left")
    bx.legend(title="$R_f$", fontsize=7.2, title_fontsize=7.4, loc="center right")

    # ---- (c) coordination margin
    for i, rf in enumerate(RF):
        g = C[C.rf == rf].ctm_s
        cx.scatter(np.full(len(g), i) + np.random.default_rng(0).normal(0, 0.07, len(g)),
                   g, s=16, color=RF_COL[rf], edgecolor=SURFACE, linewidth=0.5,
                   zorder=3, alpha=0.9)
        cx.plot([i - 0.28, i + 0.28], [g.median()] * 2, color=INK, lw=1.8, zorder=4)
    cx.axhline(0, color=CAT[1], lw=1.4, zorder=2)
    cx.text(-0.42, 0.028, "coordination limit", fontsize=7,
            color=CAT[1], ha="left", fontweight="bold")
    cx.set_xticks(range(len(RF))); cx.set_xticklabels([f"{r:.0f}" for r in RF])
    cx.set_xlabel("Fault resistance $R_f$ (Ω)")
    cx.set_ylabel("Coordination time margin (s)")
    cx.set_title("(c) Margin of Eq. (27), all 200 cases", loc="left")
    cx.grid(axis="x", visible=False)

    fig.tight_layout()
    save(fig, "fig16_sensitivity.png")


if __name__ == "__main__":
    fig16(); fig17(); fig18()
