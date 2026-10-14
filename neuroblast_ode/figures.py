"""Manuscript figures. Each make_* function builds, saves and closes its figure."""
import numpy as np
import matplotlib.pyplot as plt

from .ccd import ccd_M1, ccd_M2, ccd_M3, ccd_M4, ccd_M5
from .models import extract_counts_from_solution
from .params import GENOS, alpha_value, beta_value, t_eval
from .plotting import (
    TITLE_SIZE, AXISLABEL_SIZE,
    add_celltype_legend, add_row_label, canonical_box_px, draw_end_labels,
    plot_cell_counts, save_figure,
)
from .solvers import (
    solve_base_model, solve_model1, solve_model2, solve_model3, solve_model4, solve_model5,
)


def _fmt_param(val):
    try: iv = int(val); return str(iv) if iv == val else f"{float(val):.3g}"
    except Exception: return str(val)


def get_row_param_text(row_label):
    if row_label in ("model 3", "model 4"): return rf"$\alpha={_fmt_param(alpha_value)}$"
    if row_label == "model 5":             return rf"$\beta={_fmt_param(beta_value)}$"
    return ""


# (row label, runner taking a GENOS entry)
ROWS = [
    ("model 1", lambda g: solve_model1(sym_frac=g[1], g_nb_base_scale=g[2])),
    ("model 2", lambda g: solve_model2(sym_frac=g[1], g_nb_base_scale=g[2])),
    ("model 3", lambda g: solve_model3(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2])),
    ("model 4", lambda g: solve_model4(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2])),
    ("model 5", lambda g: solve_model5(p_div=beta_value, sym_frac=g[1])),
]


def _add_column_titles(axes_top):
    for j, (gname, _, _) in enumerate(GENOS):
        axes_top[j].set_title(gname, fontsize=TITLE_SIZE, pad=6, y=0.98)


# ======================= Figure 7: model output grid =======================

def make_models_grid(out_dir):
    nrows, ncols = len(ROWS), len(GENOS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 14.5),
                             constrained_layout=True, sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    _add_column_titles(axes[0])

    # draw once + canonical 2-digit box for the whole figure
    canon_px = canonical_box_px(fig)

    for row_idx, (_, runner) in enumerate(ROWS):
        for col_idx, geno in enumerate(GENOS):
            ax = axes[row_idx, col_idx]
            t, NB, GMC, Im, Mat = extract_counts_from_solution(runner(geno))
            series = plot_cell_counts(ax, t, NB, GMC, Im, Mat)
            # only the left column of models 1 and 2 uses the fixed bottom stack
            draw_end_labels(ax, t, series, canon_px, stacked=(row_idx in (0, 1) and col_idx == 0))

    for rr, (lbl, _) in enumerate(ROWS):
        param_txt = get_row_param_text(lbl)
        add_row_label(axes[rr, 0], lbl if param_txt == "" else f"{lbl}  {param_txt}")

    for rr in range(nrows):
        axes[rr, 0].set_ylabel("Cell count", fontsize=AXISLABEL_SIZE)
    for cc in range(ncols):
        axes[nrows - 1, cc].set_xlabel("Time (hours)", fontsize=AXISLABEL_SIZE)

    add_celltype_legend(fig, 1.02)
    save_figure(fig, out_dir, "models_grid")
    plt.close(fig)


# ======================= Base model supplemental figure =======================

def make_base_model(out_dir):
    # WT: 0% symmetric divisions; mutant: 15% symmetric divisions
    sol_WT = solve_base_model(0.0)
    sol_mut = solve_base_model(0.15)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                             constrained_layout=True, sharex=True, sharey=True)
    all_series = []
    for ax, (title, sol) in zip(axes, [("WT", sol_WT), ("mutant", sol_mut)]):
        all_series.append((sol.t, plot_cell_counts(ax, sol.t, *sol.y)))
        ax.set_title(title, fontsize=TITLE_SIZE, pad=6)

    axes[0].set_ylabel("Cell count", fontsize=AXISLABEL_SIZE)
    for ax in axes:
        ax.set_xlabel("Time (hours)", fontsize=AXISLABEL_SIZE)

    canon_px = canonical_box_px(fig)
    for ax, (t, series) in zip(axes, all_series):
        draw_end_labels(ax, t, series, canon_px, stacked=False)

    add_celltype_legend(fig, 1.08)
    save_figure(fig, out_dir, "base_model")
    plt.close(fig)


# ======================= Cell cycle duration (CCD) supplemental figure =======================

def compute_ccd():
    """Solve Models 1–5 for every genotype and reconstruct NB cell-cycle durations.

    Returns dict: name -> (sols, ccds) for "M1".."M5", plus "base" -> (ccd_WT, ccd_mutant).
    """
    sols = {
        "M1": [solve_model1(sym_frac=g[1], g_nb_base_scale=g[2]) for g in GENOS],
        "M2": [solve_model2(sym_frac=g[1], g_nb_base_scale=g[2]) for g in GENOS],
        "M3": [solve_model3(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2]) for g in GENOS],
        "M4": [solve_model4(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2]) for g in GENOS],
        "M5": [solve_model5(p_div=beta_value, sym_frac=g[1]) for g in GENOS],
    }
    ccd_fns = {"M1": ccd_M1, "M2": ccd_M2, "M3": ccd_M3, "M4": ccd_M4,
               "M5": lambda s: ccd_M5(s, beta_value)}
    out = {k: (sols[k], [ccd_fns[k](s) for s in sols[k]]) for k in sols}

    # base model (counts only): constant CCD = 1/k_NB
    k_NB_base = 1.0/1.5
    out["base"] = tuple(np.full_like(t_eval, 1.0/k_NB_base, dtype=float) for _ in range(2))
    return out


def draw_three(ax, sols, ys, title, *, loc='upper left', anchor=None):
    """Plot WT, mud, nano from (sols, ys) with a label and legend."""
    l_wt,   = ax.plot(sols[0].t, ys[0], color="black", linestyle="-",  linewidth=1.6, label="WT")
    l_mud,  = ax.plot(sols[1].t, ys[1], color="black", linestyle=":",  linewidth=2.4, label="mudmut")
    l_nano, = ax.plot(sols[2].t, ys[2], color="black", linestyle="--", linewidth=1.6, label="nanobody")
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if anchor is None:
        ax.legend(loc=loc, frameon=False, borderaxespad=0.0)
    else:
        ax.legend(loc=loc, bbox_to_anchor=anchor, frameon=False, borderaxespad=0.0)
    ax.grid(alpha=0.3)
    return l_wt, l_mud, l_nano


def _share_ylim(axs, extra_top=0.0):
    ymin = min(ax.get_ylim()[0] for ax in axs)
    ymax = max(ax.get_ylim()[1] for ax in axs) + extra_top
    for ax in axs:
        ax.set_ylim(ymin, ymax)


def make_cellcycle_duration(out_dir, ccd):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    (axB, ax1, ax2), (ax3, ax4, ax5) = axes

    # Top row: base / model 1 / model 2
    ccd_base_WT, ccd_base_MUD = ccd["base"]
    axB.plot(t_eval, ccd_base_WT,  color="black", linestyle="-",       linewidth=1.6, label="WT")
    axB.plot(t_eval, ccd_base_MUD, color="black", linestyle="dashdot", linewidth=2.4, label="mutant")
    axB.text(0.02, 0.98, "base model", transform=axB.transAxes, ha="left", va="top",
             fontsize=11, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axB.set_ylabel("Cell-cycle duration (h)")
    axB.legend(loc="lower left", frameon=False)
    axB.grid(alpha=0.3)
    draw_three(ax1, *ccd["M1"], "model 1", loc="lower left")
    draw_three(ax2, *ccd["M2"], "model 2", loc="lower left")

    # Bottom row: models 3 / 4 / 5
    draw_three(ax3, *ccd["M3"], f"model 3   α={alpha_value:g}", loc="upper left", anchor=(0.02, 0.90))
    draw_three(ax4, *ccd["M4"], f"model 4   α={alpha_value:g}", loc="upper left", anchor=(0.02, 0.90))
    draw_three(ax5, *ccd["M5"], f"model 5   β={beta_value:g}",  loc="upper left", anchor=(0.02, 0.90))

    # hide x tick labels on top row only; x labels on bottom row
    for ax in (axB, ax1, ax2):
        ax.tick_params(labelbottom=False)
    for ax in (ax3, ax4, ax5):
        ax.set_xlabel("Time (hours)")

    # separate y-scales per row
    _share_ylim((axB, ax1, ax2), extra_top=0.2)
    _share_ylim((ax3, ax4, ax5))

    # remove duplicate y tick labels on non-left columns
    for ax in (ax1, ax2, ax4, ax5):
        ax.set_yticklabels([])

    fig.suptitle("NB Cell Cycle Duration for All Models", fontsize=15, fontweight="bold", y=1.03)
    save_figure(fig, out_dir, "NB_cellcycle_duration")
    plt.close(fig)


# ======================= α / β sweep supplemental figures =======================

SWEEP_VALUES = [1, 2, 3, 4, 10]


def _make_sweep(out_dir, out_base, row_name, param_symbol, solve_for, *, stack_top_left):
    """One row per parameter value × genotype columns; each row shares a y-limit."""
    nrows, ncols = len(SWEEP_VALUES), len(GENOS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 14.5),
                             constrained_layout=True, sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    _add_column_titles(axes[0])

    canon_px = canonical_box_px(fig)

    for row_idx, val in enumerate(SWEEP_VALUES):
        row_cache = []           # (col_idx, ax, t, series) to label once y-limits are locked
        row_ymax  = 0.0

        # PASS 1: draw curves and track max for a common row ylim
        for col_idx, geno in enumerate(GENOS):
            ax = axes[row_idx, col_idx]
            t, NB, GMC, Im, Mat = extract_counts_from_solution(solve_for(val, geno))
            series = plot_cell_counts(ax, t, NB, GMC, Im, Mat)
            row_cache.append((col_idx, ax, t, series))
            row_ymax = max(row_ymax, NB.max(), GMC.max(), Im.max(), Mat.max())

        for _, ax, _, _ in row_cache:
            ax.set_ylim(0, row_ymax * 1.05)

        fig.canvas.draw()  # refresh transforms now that y-lims are locked

        # PASS 2: place end labels using the final transforms
        for col_idx, ax, t, series in row_cache:
            draw_end_labels(ax, t, series, canon_px,
                            stacked=(stack_top_left and row_idx == 0 and col_idx == 0))

    for rr, val in enumerate(SWEEP_VALUES):
        add_row_label(axes[rr, 0], f"{row_name}  {param_symbol}={val}")

    for rr in range(nrows):
        axes[rr, 0].set_ylabel("Cell count", fontsize=AXISLABEL_SIZE)
        for cc in range(ncols):
            axes[rr, cc].set_xlabel("Time (hours)", fontsize=AXISLABEL_SIZE)

    add_celltype_legend(fig, 1.02)
    save_figure(fig, out_dir, out_base)
    plt.close(fig)


def make_model3_alpha_sweep(out_dir):
    _make_sweep(out_dir, "Model3_alpha_sweep", "model 3", "α",
                lambda a, g: solve_model3(alpha_growth=a, sym_frac=g[1], g_nb_base_scale=g[2]),
                stack_top_left=True)


def make_model4_alpha_sweep(out_dir):
    _make_sweep(out_dir, "Model4_alpha_sweep", "model 4", "α",
                lambda a, g: solve_model4(alpha_growth=a, sym_frac=g[1], g_nb_base_scale=g[2]),
                stack_top_left=True)


def make_model5_beta_sweep(out_dir):
    _make_sweep(out_dir, "Model5_beta_sweep", "model 5", "β",
                lambda b, g: solve_model5(p_div=b, sym_frac=g[1]),
                stack_top_left=False)


def make_all(out_dir, groups=("manuscript", "supplemental")):
    if "manuscript" in groups:
        make_models_grid(out_dir)
    if "supplemental" in groups:
        make_base_model(out_dir)
        make_cellcycle_duration(out_dir, compute_ccd())
        make_model3_alpha_sweep(out_dir)
        make_model4_alpha_sweep(out_dir)
        make_model5_beta_sweep(out_dir)
