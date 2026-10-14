"""Shared figure styling, end-label helpers and vector export."""
import io
import os

import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .params import t_span

# ---- display tuning ----
TITLE_SIZE       = 14
ROWLABEL_SIZE    = 13
AXISLABEL_SIZE   = 13
TICKLABEL_SIZE   = 12
LEGEND_SIZE      = 12
DIGITS_FONTSIZE  = 14     # numbers inside the end boxes

CELL_COLORS = {"NB":"#1b9e77","GMC":"#d95f02","Im":"#7570b3","Mat":"#e7298a"}

# shared pixel paddings
RECT_PAD_X_PX, RECT_PAD_Y_PX = 6, 2
RIGHT_EDGE_PAD_PX = 2
DIGIT_FONT = dict(fontfamily="DejaVu Sans Mono")


def _txt_wh(text, *, fontsize, fig, renderer):
    tmp = fig.text(0, 0, text, fontsize=fontsize, **DIGIT_FONT)
    bb = tmp.get_window_extent(renderer=renderer); tmp.remove()
    return bb.width, bb.height


def _get_renderer(fig):
    ren = fig.canvas.get_renderer()
    if ren is None:
        fig.canvas.draw(); ren = fig.canvas.get_renderer()
    return ren


def canonical_box_px(fig):
    """Draw once and measure the canonical 2-digit end-label box (width, height) in pixels."""
    fig.canvas.draw()
    ren = _get_renderer(fig)
    w = _txt_wh("88", fontsize=DIGITS_FONTSIZE, fig=fig, renderer=ren)[0] + 2*RECT_PAD_X_PX
    h = _txt_wh("88", fontsize=DIGITS_FONTSIZE, fig=fig, renderer=ren)[1] + 2*RECT_PAD_Y_PX
    return w, h


def draw_fixed_bottom_stack_rects(ax, t, series_dict, colors, *,
                                  digits_fontsize, W_CANON_PX, H_CANON_PX,
                                  start_px=12, step_px=22):
    if len(t) == 0: return
    y_last = {k: float(v[-1]) for k, v in series_dict.items() if len(v)}
    if not y_last: return
    fig = ax.figure; ren = _get_renderer(fig)

    ax_bb = ax.get_window_extent(renderer=ren)
    ax_w_px, ax_h_px = ax_bb.width, ax_bb.height
    x_right_frac = 1.0 - RIGHT_EDGE_PAD_PX / ax_w_px

    start_frac = start_px / ax_h_px
    step_frac  = step_px  / ax_h_px
    order = sorted(y_last.keys(), key=lambda n: y_last[n])
    centers = {n: start_frac + i*step_frac for i, n in enumerate(order)}

    for n in order:
        s = f"{int(round(y_last[n]))}"
        if len(s) <= 2: w_px, h_px = W_CANON_PX, H_CANON_PX
        else:
            tw, th = _txt_wh(s, fontsize=digits_fontsize, fig=fig, renderer=ren)
            w_px, h_px = tw + 2*RECT_PAD_X_PX, th + 2*RECT_PAD_Y_PX
        w_frac, h_frac = w_px/ax_w_px, h_px/ax_h_px
        x_center_frac  = x_right_frac - 0.5*w_frac
        y_center_frac  = centers[n]
        ax.add_patch(Rectangle((x_center_frac - 0.5*w_frac, y_center_frac - 0.5*h_frac),
                               w_frac, h_frac, transform=ax.transAxes,
                               facecolor="white", edgecolor=colors[n], linewidth=1.1,
                               zorder=3.4, clip_on=False))
        ax.text(x_center_frac, y_center_frac, s, transform=ax.transAxes,
                ha="center", va="center", fontsize=digits_fontsize, **DIGIT_FONT, zorder=3.5)


def compute_curvy_y(ax, t, series_dict, fontsize=12, min_margin_px=6, pad_px=2):
    """
    Compact end-label layout:
      - place each label at the trajectory's final y (in axes fraction coords)
      - minimally separate overlapping labels using their *measured* text height
      - two-pass compaction (upward then downward) to avoid unnecessary spread
    Returns:
        y_centers : dict[name -> float]  (axes fraction)
        y_offsets : dict[name -> float]  (delta from target, axes fraction)
        labels    : dict[name -> str]    (rounded final counts)
        colors    : dict[name -> str]    (unused placeholder)
        ax_w_px, ax_h_px : float         (axes size in pixels)
    """
    if len(t) == 0:
        return {}, {}, {}, {}, None, None

    fig = ax.figure
    ren = _get_renderer(fig)
    ax_bb = ax.get_window_extent(renderer=ren)
    ax_w_px, ax_h_px = ax_bb.width, ax_bb.height
    margin_frac = min_margin_px / ax_h_px

    # final values + axis-fraction y targets
    y_last = {k: float(v[-1]) for k, v in series_dict.items() if len(v)}
    if not y_last:
        return {}, {}, {}, {}, ax_w_px, ax_h_px

    targets = {}
    for name, arr in series_dict.items():
        if len(arr) == 0:
            continue
        ydat = arr[-1]
        disp = ax.transData.transform((t[-1], ydat))
        inv  = ax.transAxes.inverted().transform(disp)
        targets[name] = float(inv[1])  # axis fraction

    # labels and measured heights (px) -> convert to axis frac
    labels = {n: f"{int(round(y_last[n]))}" for n in targets.keys()}
    heights_frac = {}
    for n, s in labels.items():
        tw, th = _txt_wh(s, fontsize=fontsize, fig=fig, renderer=ren)
        h_px = th + 2*RECT_PAD_Y_PX   # match the rect drawn later
        heights_frac[n] = h_px / ax_h_px

    # sort by target y (bottom -> top)
    order = sorted(targets.keys(), key=lambda n: targets[n])

    # forward pass: push up just enough to avoid overlap
    y_pos = {}
    for i, n in enumerate(order):
        h = heights_frac[n]
        tgt = np.clip(targets[n], margin_frac, 1 - margin_frac)
        if i == 0:
            y_pos[n] = max(tgt, margin_frac + 0.5*h)
        else:
            prev = order[i-1]
            min_sep = 0.5*heights_frac[prev] + 0.5*h + (pad_px/ax_h_px)
            y_pos[n] = max(tgt, y_pos[prev] + min_sep)

    # backward pass: pull down where there is extra slack (keeps things compact)
    for i in range(len(order)-2, -1, -1):
        n = order[i]
        h = heights_frac[n]
        nxt = order[i+1]
        min_sep = 0.5*h + 0.5*heights_frac[nxt] + (pad_px/ax_h_px)
        # max allowed so we still leave room for the next one
        max_allowed = y_pos[nxt] - min_sep
        y_pos[n] = min(y_pos[n], max_allowed)
        y_pos[n] = np.clip(y_pos[n], margin_frac + 0.5*h, 1 - margin_frac - 0.5*h)

    # clamp topmost too
    top = order[-1]
    y_pos[top] = np.clip(y_pos[top], margin_frac + 0.5*heights_frac[top], 1 - margin_frac - 0.5*heights_frac[top])

    # offsets relative to targets
    y_offsets = {n: y_pos[n] - targets[n] for n in order}
    return y_pos, y_offsets, labels, {}, ax_w_px, ax_h_px


def draw_rect_end_labels_using_curvy_y(ax, t, series_dict, colors, *,
                                       digits_fontsize, W_CANON_PX, H_CANON_PX, pad_px=4):
    if len(t) == 0: return
    y_centers, _, labels, _, ax_w_px, ax_h_px = compute_curvy_y(ax, t, series_dict, fontsize=digits_fontsize, pad_px=pad_px)
    if not y_centers: return
    fig = ax.figure; ren = _get_renderer(fig)
    x_right_frac = 1.0 - RIGHT_EDGE_PAD_PX / ax_w_px

    for n, s in labels.items():
        if len(s) <= 2: w_px, h_px = W_CANON_PX, H_CANON_PX
        else:
            tw, th = _txt_wh(s, fontsize=digits_fontsize, fig=fig, renderer=ren)
            w_px, h_px = tw + 2*RECT_PAD_X_PX, th + 2*RECT_PAD_Y_PX
        w_frac, h_frac = w_px/ax_w_px, h_px/ax_h_px
        x_center_frac  = x_right_frac - 0.5*w_frac
        y_center_frac  = y_centers[n]
        ax.add_patch(Rectangle((x_center_frac - 0.5*w_frac, y_center_frac - 0.5*h_frac),
                               w_frac, h_frac, transform=ax.transAxes,
                               facecolor="white", edgecolor=colors[n], linewidth=1.1,
                               zorder=3.4, clip_on=False))
        ax.text(x_center_frac, y_center_frac, s, transform=ax.transAxes,
                ha="center", va="center", fontsize=digits_fontsize, **DIGIT_FONT, zorder=3.5)


def draw_end_labels(ax, t, series, canon_px, *, stacked):
    """Final-count boxes at the right edge: a fixed bottom stack, or placed at each curve's end."""
    W_CANON_PX, H_CANON_PX = canon_px
    if stacked:
        draw_fixed_bottom_stack_rects(ax, t, series, CELL_COLORS,
                                      digits_fontsize=DIGITS_FONTSIZE,
                                      W_CANON_PX=W_CANON_PX, H_CANON_PX=H_CANON_PX,
                                      start_px=12, step_px=28)
    else:
        draw_rect_end_labels_using_curvy_y(ax, t, series, CELL_COLORS,
                                           digits_fontsize=DIGITS_FONTSIZE,
                                           W_CANON_PX=W_CANON_PX, H_CANON_PX=H_CANON_PX)


def plot_cell_counts(ax, t, NB, GMC, Im, Mat):
    """Plot the four cell-type trajectories with the shared panel styling; returns the series dict."""
    ax.plot(t, NB,  color=CELL_COLORS["NB"],  lw=2.0)
    ax.plot(t, GMC, color=CELL_COLORS["GMC"], lw=2.0)
    ax.plot(t, Im,  color=CELL_COLORS["Im"],  lw=2.0)
    ax.plot(t, Mat, color=CELL_COLORS["Mat"], lw=2.0)
    ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.set_xlim(t_span[0], t_span[1])
    ax.tick_params(labelsize=TICKLABEL_SIZE)
    return {"NB": NB, "GMC": GMC, "Im": Im, "Mat": Mat}


def add_row_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=ROWLABEL_SIZE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5))


def add_celltype_legend(fig, y_anchor):
    celltype_handles = [
        Line2D([0],[0], color=CELL_COLORS["NB"],  lw=2.0, label="NB"),
        Line2D([0],[0], color=CELL_COLORS["GMC"], lw=2.0, label="GMC"),
        Line2D([0],[0], color=CELL_COLORS["Im"],  lw=2.0, label="Im"),
        Line2D([0],[0], color=CELL_COLORS["Mat"], lw=2.0, label="Mat"),
    ]
    fig.legend(handles=celltype_handles, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, y_anchor), prop={'size': LEGEND_SIZE})


def save_figure(fig, out_dir, out_base):
    """Save fig as <out_dir>/<out_base>.svg with a tight bounding box."""
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'   # match DejaVu Sans
    mpl.rcParams['mathtext.rm'] = 'DejaVu Sans'

    # Keep text as real text in SVG (not paths)
    mpl.rcParams['svg.fonttype'] = 'none'

    # Fixed salt so SVG element ids are identical run to run
    mpl.rcParams['svg.hashsalt'] = 'fixed'

    # Render once to PNG before saving.
    fig.canvas.print_figure(io.BytesIO(), format="png", facecolor=fig.get_facecolor(),
                            edgecolor=fig.get_edgecolor(), dpi=fig.dpi, bbox_inches="tight")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_base)
    fig.savefig(f"{path}.svg", format="svg", bbox_inches="tight")
