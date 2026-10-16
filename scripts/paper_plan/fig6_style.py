"""Shared style for the v6 manuscript figures (docs/paper_figures_v6.md @ d46b9a5): sans-serif 7 pt at final size, bold panel letters, one colour per
model (Okabe-Ito), per band (Tol light) and per class, PNG 300 dpi + PDF under paper/figures_v6/, aggregate source data under results/paper_final/figures_v6/.
Paths are relative to the repo root (two levels above this file), so the same scripts run locally and on the cluster checkout."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); FIG = ROOT + "/paper/figures_v6"; SRC = ROOT + "/results/paper_final/figures_v6"; RES = ROOT + "/results/paper_final"
os.makedirs(FIG, exist_ok=True); os.makedirs(SRC, exist_ok=True)
MM = 1 / 25.4   # inches per mm; Nature columns: 89 mm single, 183 mm double
MODEL = {"L-IMG": "#0072B2", "L-CNV": "#D55E00", "L-CNV-ps": "#E69F00", "L-CNV-past": "#CC79A7", "L-CNV-joint": "#7B3F00", "L-LATE": "#009E73", "P": "#56B4E9", "grade": "#444444"}
MODEL_NAME = {"L-IMG": "L-IMG (H&E model)", "L-CNV": "L-CNV, shipped matrix", "L-CNV-ps": "L-CNV, per-sample segmentation", "L-CNV-past": "L-CNV, past-only segmentation",
              "L-CNV-joint": "L-CNV, joint package", "L-LATE": "L-LATE (late fusion)", "P": "Published model (P)", "grade": "Pathologist grade"}
BAND = {"low": "#88CCEE", "mid": "#DDCC77", "high": "#CC6677"}; BAND_NAME = {"low": "low band: cleared", "mid": "middle band: sequenced", "high": "high band: flagged"}
CLASS = {"P": "#882255", "N": "#44AA99"}; CLASS_NAME = {"P": "progressor", "N": "non-progressor"}; GREY = "#777777"; LIGHT = "#BBBBBB"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,   # bundled with matplotlib on every machine (the cluster has no Arial/Helvetica); Type 42 text, swap at submission "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.0,
                     "axes.titlelocation": "left", "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.2, "mathtext.default": "regular"})
def letter(fig, ax, s, dx=-0.025, dy=0.008):
    """bold panel letter just outside the top-left corner of an axes (offsets in figure fraction)"""
    b = ax.get_position(); fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig); print("FIG6 SAVED", name, flush=True)
def dump(name, obj):
    json.dump(obj, open(f"{SRC}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))); print("FIG6 JSON", name, flush=True)
def J(rel): return json.load(open(f"{RES}/{rel}"))
def ci_point(ax, x, y, ci, color, horizontal=False, ms=4, marker="o", lw=0.9, zorder=3, mfc=None, label=None):
    """point estimate with its 95% CI as a line"""
    lo, hi = ci
    if horizontal: ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=zorder - 1, solid_capstyle="butt"); ax.plot([x], [y], marker=marker, color=color, ms=ms, lw=0, zorder=zorder, mfc=color if mfc is None else mfc, label=label)
    else: ax.plot([x, x], [lo, hi], color=color, lw=lw, zorder=zorder - 1, solid_capstyle="butt"); ax.plot([x], [y], marker=marker, color=color, ms=ms, lw=0, zorder=zorder, mfc=color if mfc is None else mfc, label=label)
def signed(v, d=3): return ("−" if v < 0 else "+") + f"{abs(v):.{d}f}"
def mi(v): return f"{v:.3f}".replace("-", "−")
def box(ax, x, y, w, h, text, fc, ec, fs=6.5, lw=0.8, weight="normal", color="black", pad=0.02, rounding=1.2):
    """rounded box with centred text, in data coordinates of an axes with axis off"""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={rounding}", fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color, linespacing=1.25)
def arrow(ax, x0, y0, x1, y1, color="#555555", lw=0.8):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7))
