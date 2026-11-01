"""
plotstyle.py -- shared figure style for the manuscript.

Palette: Okabe-Ito derived, reordered so every adjacent pair clears the
colour-vision-deficiency separation threshold (validated: CVD dE 9.6 worst
adjacent pair, normal-vision dE 20.0 worst adjacent pair).  Every figure also
carries a legend and/or direct labels, so identity is never colour-alone and
the figures remain readable in greyscale print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical order -- assigned in order, never cycled.
CAT = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]

SURFACE = "#FCFCFB"
INK = "#1A1A1A"
INK_MUTED = "#5A5A5A"
GRID = "#D8D8D4"

# Sequential: one hue, light -> dark (magnitude).
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#F4F8FB", "#C5DCEC", "#7FB4D6", "#3888BD", "#0072B2", "#004E7A"]
)
# Diverging: two hues + neutral grey midpoint (polarity).
DIV = LinearSegmentedColormap.from_list(
    "div_bo", ["#7A3000", "#D55E00", "#F0B487", "#E8E8E6",
               "#8FC0DD", "#0072B2", "#003F63"]
)

RC = {
    "figure.dpi": 150, "savefig.dpi": 400,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
}


def apply():
    plt.rcParams.update(RC)


def pending_banner(ax, text):
    """Stamp a figure that is a placeholder for an unexecuted external stage."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#7A3000", fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#FDF3EC",
                      edgecolor="#D55E00", linewidth=1.4))
