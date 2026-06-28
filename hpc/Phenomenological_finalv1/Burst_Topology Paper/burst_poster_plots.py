#!/usr/bin/env python3
"""
burst_poster_plots.py -- poster-grade figures for network-burst runs.

PURE RENDER LAYER. No burst detection and no file discovery live here: the caller
hands in the spike arrays, the ALREADY-detected burst intervals (both detectors,
from burst_metrics), and the scalar metric dict, and this module only draws. That
keeps the science (detection / metrics in burst_metrics) and the presentation
(figures) fully decoupled -- a change to a detector never touches plotting, and a
restyle (burst_palettes) never touches the metrics. The only "compute" done here
is turning a spike train into a smoothed instantaneous-firing-rate (IFR) trace for
display, exactly as burst_plots.py already does via burst_metrics.population_rate.

WHAT IT DRAWS (one run -> up to 8 figure types, each in any palette)
--------------------------------------------------------------------
  hero        raster behind + filled NORMALIZED IFR in front, twin 0..1 axis.
              Reproduces the attached MEA-style example (raster + teal IFR).
  stacked     two panels: raster (scoring-burst spans shaded) over the absolute
              IFR [Hz/neuron] with the detection threshold theta and both
              detectors' spans, so detector (dis)agreement is visible.
  overlay     single panel: raster + IFR LINE on a twin axis + theta + spans +
              burst-onset markers. More analytical than hero, still one panel.
  heatmap     neurons reordered by burst-recruitment latency, time-binned spike
              counts as an image, IFR line overlaid. Shows recruitment structure.
  stereotypy  every detected burst aligned to its IFR peak and overlaid
              (each normalised to its own peak) + mean +/- std. Burst-shape
              consistency at a glance.
  returnmap   IBI Poincare map (IBI_n vs IBI_{n+1}) + burst-duration histogram.
              Regularity / variability descriptor.
  zoom        the representative burst (duration closest to the run median) over
              a configurable window (default 6 s, matching Mossink 2019 Fig 2g/h).
  composite   the drop-on-poster panel: hero on top; zoom + return map + a
              metrics-vs-Mossink-control box beneath.

  contact_sheet(runs)  a grid of mini-heroes for the top-K exemplars (driver use).

All sizing/coloring is delegated to a burst_palettes.PosterStyle, so this module
never hard-codes a color or a font size; the figures are designed at an A0 panel
width and saved as BOTH vector PDF (crisp at any final scale) and high-dpi PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")                        # headless / cluster-safe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import burst_metrics as bm
import burst_palettes as bp


# ===========================================================================
# Data carried into every figure function (uniform signature fn(D, style))
# ===========================================================================
@dataclass
class RunData:
    """Everything a figure needs about ONE run. Detection is already done."""
    spk_t: np.ndarray                  # spike times [s], k = 0..K-1
    spk_i: np.ndarray                  # spiking neuron index 0..n_neurons-1
    n_neurons: int                     # N_neurons
    t_rec: float                       # recording duration [s]
    pr_intervals: List[Tuple[float, float]]   # population-rate detector spans
    li_intervals: List[Tuple[float, float]]   # log-ISI/participation spans
    metrics: Dict[str, float]          # scalar_row from burst_metrics.compute_all
    cfg: bm.BurstConfig = field(default_factory=bm.BurstConfig)
    ifr_display_sigma_s: float = 0.05  # Gaussian std for the DISPLAY IFR envelope
    title: str = ""
    channel_label: str = "Neuron index"  # left-axis label (set "Channels" to taste)
    targets: Optional[Dict[str, Tuple[float, float]]] = None  # Mossink anchor

    @property
    def scoring_intervals(self) -> List[Tuple[float, float]]:
        return (self.li_intervals if self.cfg.scoring_detector == "logisi"
                else self.pr_intervals)

    @property
    def scoring_label(self) -> str:
        return ("log-ISI / participation" if self.cfg.scoring_detector == "logisi"
                else "population-rate")


# ===========================================================================
# Small render helpers (no detection)
# ===========================================================================
def _valid_rc(rc: Dict[str, object]) -> Dict[str, object]:
    """Keep only keys this matplotlib understands (PosterStyle.rc is mpl-free,
    so it may name a key an older mpl lacks -- drop those rather than crash)."""
    valid = set(plt.rcParams.keys())
    return {k: v for k, v in rc.items() if k in valid}


def _heat_cmap(pal: bp.Palette) -> LinearSegmentedColormap:
    """Light-to-dark teal colormap for activity images (0 spikes blends into
    paper; high counts saturate to the brand teal)."""
    return LinearSegmentedColormap.from_list(
        "brand_teal", [pal.heat_lo, pal.ifr_fill, pal.heat_hi], N=256)


def _display_ifr(D: RunData, sigma_s: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed per-neuron population rate r_pn(t) [Hz/neuron] for DISPLAY.
    sigma_s defaults to D.ifr_display_sigma_s (a cleaner envelope than the
    detector's own smoothing). Returns (t_centers, r_pn)."""
    sig = D.ifr_display_sigma_s if sigma_s is None else sigma_s
    return bm.population_rate(D.spk_t, D.n_neurons, D.t_rec, D.cfg.grid_dt_s, sig)


def _detector_ifr(D: RunData) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """r_pn(t) at the DETECTOR's smoothing (so theta and the detected spans line
    up), plus (mu, sd, theta) with theta = mu + k*sd."""
    t, r = bm.population_rate(D.spk_t, D.n_neurons, D.t_rec,
                              D.cfg.grid_dt_s, D.cfg.pr_smooth_sigma_s)
    mu, sd = float(r.mean()), float(r.std())
    theta = mu + D.cfg.pr_thresh_k * sd
    return t, r, mu, sd, theta


def _norm_to_max(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x)) if x.size else 0.0
    return x / m if m > 0 else x


def _representative_burst(intervals: List[Tuple[float, float]]
                          ) -> Optional[Tuple[float, float]]:
    """Interval whose duration is closest to the median duration."""
    if not intervals:
        return None
    durs = np.array([e - s for s, e in intervals])
    return intervals[int(np.argmin(np.abs(durs - np.median(durs))))]


def _recruitment_order(D: RunData) -> np.ndarray:
    """Neuron indices reordered so early-in-burst firers come first.

    For each neuron, median latency from the onset of the burst its in-burst
    spikes fall into; neurons that never fire inside a scoring burst are pushed
    to the end, ordered by descending total spike count. Reveals the recruitment
    gradient in the heatmap (index order in the raw data is arbitrary)."""
    n = D.n_neurons
    intervals = D.scoring_intervals
    counts = np.bincount(np.asarray(D.spk_i, dtype=np.int64),
                         minlength=n)[:n].astype(np.float64)
    if not intervals or D.spk_t.size == 0:
        return np.argsort(-counts, kind="stable")

    starts = np.array([s for s, _ in intervals])
    stops = np.array([e for _, e in intervals])
    lat_sum = np.zeros(n)
    lat_cnt = np.zeros(n)
    order = np.argsort(D.spk_t, kind="stable")
    ts = D.spk_t[order]
    isort = np.asarray(D.spk_i)[order]
    # assign each spike to the burst whose [start, stop] contains it
    bi = np.searchsorted(starts, ts, side="right") - 1
    valid = (bi >= 0) & (bi < len(intervals))
    valid &= np.where(valid, ts <= stops[np.clip(bi, 0, len(intervals) - 1)], False)
    for nid, tt, b, ok in zip(isort, ts, bi, valid):
        if ok:
            lat_sum[nid] += (tt - starts[b])
            lat_cnt[nid] += 1
    has = lat_cnt > 0
    med_lat = np.full(n, np.inf)
    med_lat[has] = lat_sum[has] / lat_cnt[has]      # mean latency (cheap proxy)
    # primary key: in-burst neurons by latency; trailing: silent by -count
    early = np.flatnonzero(has)
    early = early[np.argsort(med_lat[early], kind="stable")]
    late = np.flatnonzero(~has)
    late = late[np.argsort(-counts[late], kind="stable")]
    return np.concatenate([early, late]).astype(np.int64)


def _metrics_box_text(D: RunData) -> str:
    """Monospace run-vs-Mossink-DIV28 table for the metrics box."""
    cfg = D.cfg
    targets = D.targets or bm.MOSSINK_CONTROL_DIV28
    keymap = bm.scoring_keys(cfg)
    rows = [("burst/min", "burst_rate_per_min", "{:.2f}"),
            ("dur (s)",   "burst_dur_mean_s",   "{:.3f}"),
            ("IBI (s)",   "nibi_mean_s",         "{:.2f}"),
            ("CV IBI",    "nibi_cv",             "{:.2f}"),
            ("out-frac",  "frac_spikes_outside_burst", "{:.2f}"),
            ("FR (Hz)",   "mean_FR_Hz",          "{:.2f}")]
    lines = ["metric        run     Mossink ctrl"]
    for label, mk, fmt in rows:
        val = D.metrics.get(keymap[mk], float("nan"))
        tgt_m, tgt_s = targets[mk]
        try:
            vs = fmt.format(val)
        except (ValueError, TypeError):
            vs = "nan"
        lines.append("{:<12} {:>6}   {:g} +/- {:g}".format(label, vs, tgt_m, tgt_s))
    return "\n".join(lines)


def _style_right_axis(ax2, color: str) -> None:
    ax2.spines["right"].set_color(color)
    ax2.tick_params(axis="y", colors=color)
    ax2.yaxis.label.set_color(color)


def _shade(ax, intervals, color, alpha, ymin=0.0, ymax=1.0, zorder=1):
    for s, e in intervals:
        ax.axvspan(s, e, ymin=ymin, ymax=ymax, color=color, alpha=alpha,
                   lw=0, zorder=zorder)


# ===========================================================================
# Figure 1 -- HERO (the attached-example composition)
# ===========================================================================
def fig_hero(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig, ax = plt.subplots(figsize=style.figsize(2.25))

    t, r = _display_ifr(D)
    ifr = _norm_to_max(r)
    N = max(D.n_neurons, 1)

    # IFR mapped onto the raster's own y-axis (0..N) so the raster, drawn at a
    # higher zorder, always sits ON TOP of the fill (no cross-axes z-order fight).
    ax.fill_between(t, 0.0, ifr * N, color=pal.ifr_fill, alpha=0.55,
                    lw=0, zorder=1)
    ax.plot(t, ifr * N, color=pal.ifr_line, lw=style.lw_ifr, zorder=2)
    ax.scatter(D.spk_t, D.spk_i, s=style.marker_raster, c=pal.raster,
               marker=".", linewidths=0, zorder=3, rasterized=True)

    ax.set_xlim(0, D.t_rec)
    ax.set_ylim(0, N)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(D.channel_label)
    ax.set_title(D.title or "Raster + population IFR")
    for sp in ("top",):
        ax.spines[sp].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.0)                       # 0..N on ax maps to 0..1 here
    ax2.set_ylabel("Normalized IFR")
    ax2.spines["top"].set_visible(False)
    _style_right_axis(ax2, pal.ifr_line)
    return fig


# ===========================================================================
# Figure 2 -- STACKED (raster over absolute IFR with threshold + both detectors)
# ===========================================================================
def fig_stacked(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig = plt.figure(figsize=style.figsize(1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.10)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_r)

    N = max(D.n_neurons, 1)
    # panel 1: raster + scoring-burst spans
    _shade(ax_r, D.scoring_intervals, pal.span_li if D.cfg.scoring_detector == "logisi"
           else pal.span_pr, 0.18)
    ax_r.scatter(D.spk_t, D.spk_i, s=style.marker_raster, c=pal.raster,
                 marker=".", linewidths=0, rasterized=True, zorder=3)
    ax_r.set_ylim(0, N)
    ax_r.set_ylabel(D.channel_label)
    ax_r.set_title(D.title or "Raster and population rate")
    ax_r.tick_params(labelbottom=False)
    ax_r.spines["top"].set_visible(False)

    # panel 2: absolute IFR + theta + both detectors' spans
    t, r, mu, sd, theta = _detector_ifr(D)
    ax_p.plot(t, r, color=pal.ifr_line, lw=style.lw_ifr * 0.75, zorder=3)
    ax_p.axhline(theta, ls="--", lw=style.lw_thresh, color=pal.threshold, zorder=4)
    ax_p.text(0.004, 0.92,
              r"$\theta=\mu+%.1f\,\sigma$" % D.cfg.pr_thresh_k,
              transform=ax_p.transAxes, color=pal.threshold,
              fontsize=style.pt_annot, va="top", ha="left")
    # log-ISI spans as full-height washes; poprate spans as a thin floor ribbon
    _shade(ax_p, D.li_intervals, pal.span_li, 0.22)
    for s, e in D.pr_intervals:
        ax_p.axvspan(s, e, ymin=0.0, ymax=0.06, color=pal.span_pr, alpha=0.85, lw=0)
    ax_p.set_xlim(0, D.t_rec)
    ax_p.set_ylabel("Pop. rate (Hz/neuron)")
    ax_p.set_xlabel("Time (s)")
    ax_p.spines["top"].set_visible(False)

    # legend strip
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=pal.ifr_line, lw=style.lw_ifr * 0.75, label="IFR"),
        Line2D([0], [0], color=pal.threshold, lw=style.lw_thresh, ls="--",
               label=r"threshold $\theta$"),
        Patch(facecolor=pal.span_li, alpha=0.5, label="log-ISI burst"),
        Patch(facecolor=pal.span_pr, alpha=0.85, label="pop-rate burst"),
    ]
    ax_p.legend(handles=handles, loc="upper right", ncol=2,
                fontsize=style.pt_legend * 0.8, handlelength=1.4,
                columnspacing=1.0)
    return fig


# ===========================================================================
# Figure 3 -- OVERLAY (single panel, raster + IFR line + theta + spans + onsets)
# ===========================================================================
def fig_overlay(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig, ax = plt.subplots(figsize=style.figsize(2.25))
    N = max(D.n_neurons, 1)

    _shade(ax, D.scoring_intervals,
           pal.span_li if D.cfg.scoring_detector == "logisi" else pal.span_pr,
           0.16, zorder=1)
    ax.scatter(D.spk_t, D.spk_i, s=style.marker_raster, c=pal.raster,
               marker=".", linewidths=0, rasterized=True, zorder=3)
    # burst-onset markers along the top
    for s, _e in D.scoring_intervals:
        ax.plot([s], [N * 0.985], marker="v", ms=style.marker_raster * 1.1,
                color=pal.threshold, zorder=4, clip_on=False)
    ax.set_xlim(0, D.t_rec)
    ax.set_ylim(0, N)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(D.channel_label)
    ax.set_title(D.title or "Raster with IFR overlay")
    ax.spines["top"].set_visible(False)

    # IFR (detector smoothing) + theta on a twin axis, both mapped to 0..1
    t, r, mu, sd, theta = _detector_ifr(D)
    rmax = float(r.max()) if r.size else 1.0
    rmax = rmax if rmax > 0 else 1.0
    ax2 = ax.twinx()
    ax2.plot(t, r, color=pal.ifr_line, lw=style.lw_ifr, zorder=5)
    ax2.axhline(theta, ls="--", lw=style.lw_thresh, color=pal.threshold, zorder=5)
    ax2.set_ylim(0, rmax)
    ax2.set_ylabel("Pop. rate (Hz/neuron)")
    ax2.spines["top"].set_visible(False)
    _style_right_axis(ax2, pal.ifr_line)
    return fig


# ===========================================================================
# Figure 4 -- HEATMAP (recruitment-ordered activity image + IFR line)
# ===========================================================================
def fig_heatmap(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig, ax = plt.subplots(figsize=style.figsize(2.0))
    N = max(D.n_neurons, 1)

    # time-binned per-neuron spike counts, neurons reordered by recruitment
    dt = max(D.cfg.grid_dt_s * 8.0, 0.02)        # coarser bins read better at A0
    nb = max(int(np.ceil(D.t_rec / dt)), 1)
    order = _recruitment_order(D)
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)
    H = np.zeros((N, nb), dtype=np.float64)
    if D.spk_t.size:
        tb = np.clip((np.asarray(D.spk_t) / dt).astype(np.int64), 0, nb - 1)
        rr = rank[np.asarray(D.spk_i, dtype=np.int64)]
        np.add.at(H, (rr, tb), 1.0)

    im = ax.imshow(H, aspect="auto", origin="lower", cmap=_heat_cmap(pal),
                   extent=(0.0, D.t_rec, 0.0, N), interpolation="nearest",
                   rasterized=True, zorder=1)
    ax.set_xlim(0, D.t_rec)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(D.channel_label + " (recruitment order)")
    ax.set_title(D.title or "Recruitment-ordered activity")
    ax.spines["top"].set_visible(False)

    cb = fig.colorbar(im, ax=ax, pad=0.10, fraction=0.045)
    cb.set_label("spikes / %.0f ms bin" % (dt * 1000.0),
                 fontsize=style.pt_tick)
    cb.ax.tick_params(labelsize=style.pt_tick * 0.85)

    t, r = _display_ifr(D)
    ifr = _norm_to_max(r)
    ax2 = ax.twinx()
    ax2.plot(t, ifr, color=pal.ifr_line, lw=style.lw_ifr, zorder=3)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Normalized IFR")
    ax2.spines["top"].set_visible(False)
    _style_right_axis(ax2, pal.ifr_line)
    return fig


# ===========================================================================
# Figure 5 -- STEREOTYPY (burst-aligned IFR overlay + mean +/- std)
# ===========================================================================
def fig_stereotypy(D: RunData, style: bp.PosterStyle, window_s: float = 2.0):
    pal = style.palette
    fig, ax = plt.subplots(figsize=style.figsize(1.65))
    intervals = D.scoring_intervals

    t, r, _mu, _sd, _theta = _detector_ifr(D)
    dt = D.cfg.grid_dt_s
    half = max(int(round(window_s / dt)), 1)
    tau = np.arange(-half, half + 1) * dt
    snippets = []
    for s, e in intervals:
        i0 = int(np.floor(s / dt))
        i1 = int(np.ceil(e / dt))
        i0, i1 = max(i0, 0), min(i1, r.size - 1)
        if i1 < i0:
            continue
        p = i0 + int(np.argmax(r[i0:i1 + 1]))     # IFR peak inside the burst
        lo, hi = p - half, p + half + 1
        seg = np.full(tau.size, np.nan)
        a, b = max(lo, 0), min(hi, r.size)
        seg[(a - lo):(b - lo)] = r[a:b]
        pk = np.nanmax(seg)
        if pk and pk > 0:
            snippets.append(seg / pk)             # normalise to OWN peak (shape)

    if snippets:
        M = np.vstack(snippets)
        for row in M:
            ax.plot(tau, row, color=pal.ifr_fill, lw=style.lw_ifr * 0.35,
                    alpha=0.45, zorder=2)
        mean = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
        ax.fill_between(tau, mean - sd, mean + sd, color=pal.ifr_fill,
                        alpha=0.25, lw=0, zorder=1)
        ax.plot(tau, mean, color=pal.ifr_line, lw=style.lw_ifr, zorder=3,
                label="mean (n=%d)" % M.shape[0])
        ax.axvline(0.0, ls=":", lw=style.lw_thresh * 0.8, color=pal.threshold,
                   zorder=4)
        ax.legend(loc="upper right", fontsize=style.pt_legend * 0.85)
    else:
        ax.text(0.5, 0.5, "fewer than one detectable burst",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=style.pt_annot, color=pal.muted)

    ax.set_xlim(-window_s, window_s)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time from burst IFR peak (s)")
    ax.set_ylabel("IFR (norm. to own peak)")
    ax.set_title(D.title or "Burst stereotypy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ===========================================================================
# Figure 6 -- RETURN MAP (IBI Poincare + burst-duration histogram)
# ===========================================================================
def fig_returnmap(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig = plt.figure(figsize=style.figsize(1.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.32)
    ax_p = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1])

    intervals = sorted(D.scoring_intervals)
    starts = np.array([s for s, _ in intervals])
    stops = np.array([e for _, e in intervals])
    durs = (stops - starts) if intervals else np.array([])

    # --- Poincare of network IBIs (silent gap onset_{b+1} - offset_b) ---
    if len(intervals) >= 3:
        ibi = starts[1:] - stops[:-1]
        x, y = ibi[:-1], ibi[1:]
        cidx = np.arange(x.size)
        sc = ax_p.scatter(x, y, c=cidx, cmap=_heat_cmap(pal),
                          s=style.marker_raster * 9.0, edgecolors=pal.ink,
                          linewidths=0.6 * style.scale, zorder=3)
        lim = float(np.nanmax(ibi)) * 1.08 if ibi.size else 1.0
        lim = lim if lim > 0 else 1.0
        ax_p.plot([0, lim], [0, lim], ls="--", lw=style.lw_thresh,
                  color=pal.threshold, zorder=2, label="$y=x$")
        ax_p.set_xlim(0, lim)
        ax_p.set_ylim(0, lim)
        ax_p.legend(loc="upper left", fontsize=style.pt_legend * 0.85)
        cb = fig.colorbar(sc, ax=ax_p, pad=0.02, fraction=0.045)
        cb.set_label("burst order", fontsize=style.pt_tick * 0.9)
        cb.ax.tick_params(labelsize=style.pt_tick * 0.8)
    else:
        ax_p.text(0.5, 0.5, "need >= 3 bursts\nfor an IBI return map",
                  ha="center", va="center", transform=ax_p.transAxes,
                  fontsize=style.pt_annot, color=pal.muted)
    ax_p.set_xlabel(r"IBI$_{n}$ (s)")
    ax_p.set_ylabel(r"IBI$_{n+1}$ (s)")
    ax_p.set_title("IBI return map")
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)

    # --- burst-duration histogram ---
    if durs.size >= 1:
        nb = int(min(20, max(5, durs.size // 2)))
        ax_h.hist(durs, bins=nb, color=pal.ifr_fill, edgecolor=pal.ink,
                  linewidth=0.8 * style.scale, zorder=3)
        ax_h.axvline(float(np.median(durs)), ls="--", lw=style.lw_thresh,
                     color=pal.threshold, zorder=4,
                     label="median %.3f s" % float(np.median(durs)))
        ax_h.legend(loc="upper right", fontsize=style.pt_legend * 0.85)
    else:
        ax_h.text(0.5, 0.5, "no bursts", ha="center", va="center",
                  transform=ax_h.transAxes, fontsize=style.pt_annot,
                  color=pal.muted)
    ax_h.set_xlabel("Burst duration (s)")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Burst durations")
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    fig.suptitle(D.title or "Burst regularity", fontsize=style.pt_title,
                 fontweight="bold", color=pal.ink)
    return fig


# ===========================================================================
# Figure 7 -- ZOOM (representative single burst)
# ===========================================================================
def fig_zoom(D: RunData, style: bp.PosterStyle, zoom_window_s: float = 6.0):
    pal = style.palette
    fig, ax = plt.subplots(figsize=style.figsize(2.0))
    N = max(D.n_neurons, 1)
    rep = _representative_burst(D.scoring_intervals)

    if rep is None:
        ax.text(0.5, 0.5, "no network bursts detected", ha="center",
                va="center", transform=ax.transAxes, fontsize=style.pt_annot,
                color=pal.muted)
        ax.set_xlim(0, D.t_rec)
        ax.set_ylim(0, N)
    else:
        c = 0.5 * (rep[0] + rep[1])
        z0 = max(0.0, c - 0.5 * zoom_window_s)
        z1 = min(D.t_rec, c + 0.5 * zoom_window_s)
        m = (D.spk_t >= z0) & (D.spk_t <= z1)

        # IFR within the window, normalised to the window max, mapped to 0..N
        t, r = _display_ifr(D)
        wm = (t >= z0) & (t <= z1)
        tt, rr = t[wm], r[wm]
        rrn = _norm_to_max(rr)
        ax.fill_between(tt, 0.0, rrn * N, color=pal.ifr_fill, alpha=0.5, lw=0,
                        zorder=1)
        ax.plot(tt, rrn * N, color=pal.ifr_line, lw=style.lw_ifr, zorder=2)
        ax.axvspan(rep[0], rep[1], color=pal.span_li
                   if D.cfg.scoring_detector == "logisi" else pal.span_pr,
                   alpha=0.16, lw=0, zorder=0)
        ax.scatter(D.spk_t[m], D.spk_i[m], s=style.marker_raster * 2.4,
                   c=pal.raster, marker=".", linewidths=0, zorder=3,
                   rasterized=True)
        ax.set_xlim(z0, z1)
        ax.set_ylim(0, N)

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Normalized IFR")
        ax2.spines["top"].set_visible(False)
        _style_right_axis(ax2, pal.ifr_line)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(D.channel_label)
    ax.set_title(D.title or
                 ("Representative burst (%.1f s window)" % zoom_window_s))
    ax.spines["top"].set_visible(False)
    return fig


# ===========================================================================
# Figure 8 -- COMPOSITE (drop-on-poster: hero + zoom + return map + metrics)
# ===========================================================================
def fig_composite(D: RunData, style: bp.PosterStyle):
    pal = style.palette
    fig = plt.figure(figsize=style.figsize(1.32))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 2.0],
                          hspace=0.52, wspace=0.34)
    ax_h = fig.add_subplot(gs[0, :])             # hero spans the top
    ax_z = fig.add_subplot(gs[1, 0])             # zoom
    ax_p = fig.add_subplot(gs[1, 1])             # IBI return map
    ax_m = fig.add_subplot(gs[1, 2])             # metrics box
    N = max(D.n_neurons, 1)

    # ---- hero ----
    t, r = _display_ifr(D)
    ifr = _norm_to_max(r)
    ax_h.fill_between(t, 0.0, ifr * N, color=pal.ifr_fill, alpha=0.55, lw=0,
                      zorder=1)
    ax_h.plot(t, ifr * N, color=pal.ifr_line, lw=style.lw_ifr, zorder=2)
    ax_h.scatter(D.spk_t, D.spk_i, s=style.marker_raster, c=pal.raster,
                 marker=".", linewidths=0, zorder=3, rasterized=True)
    ax_h.set_xlim(0, D.t_rec)
    ax_h.set_ylim(0, N)
    ax_h.set_xlabel("Time (s)")
    ax_h.set_ylabel(D.channel_label)
    ax_h.set_title(D.title or "Network bursting overview")
    ax_h.spines["top"].set_visible(False)
    ax2 = ax_h.twinx()
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Norm. IFR")
    ax2.spines["top"].set_visible(False)
    _style_right_axis(ax2, pal.ifr_line)

    # ---- zoom ----
    rep = _representative_burst(D.scoring_intervals)
    if rep is not None:
        c = 0.5 * (rep[0] + rep[1])
        z0, z1 = max(0.0, c - 3.0), min(D.t_rec, c + 3.0)
        m = (D.spk_t >= z0) & (D.spk_t <= z1)
        wm = (t >= z0) & (t <= z1)
        rrn = _norm_to_max(r[wm])
        ax_z.fill_between(t[wm], 0.0, rrn * N, color=pal.ifr_fill, alpha=0.5,
                          lw=0, zorder=1)
        ax_z.scatter(D.spk_t[m], D.spk_i[m], s=style.marker_raster * 1.6,
                     c=pal.raster, marker=".", linewidths=0, zorder=3,
                     rasterized=True)
        ax_z.set_xlim(z0, z1)
    ax_z.set_ylim(0, N)
    ax_z.set_xlabel("Time (s)", fontsize=style.pt_tick)
    ax_z.set_ylabel(D.channel_label, fontsize=style.pt_tick)
    ax_z.set_title("Representative burst", fontsize=style.pt_tick)
    ax_z.spines["top"].set_visible(False)
    ax_z.spines["right"].set_visible(False)

    # ---- IBI return map ----
    intervals = sorted(D.scoring_intervals)
    starts = np.array([s for s, _ in intervals])
    stops = np.array([e for _, e in intervals])
    if len(intervals) >= 3:
        ibi = starts[1:] - stops[:-1]
        ax_p.scatter(ibi[:-1], ibi[1:], c=np.arange(ibi.size - 1),
                     cmap=_heat_cmap(pal), s=style.marker_raster * 7.0,
                     edgecolors=pal.ink, linewidths=0.5 * style.scale, zorder=3)
        lim = float(np.nanmax(ibi)) * 1.08
        lim = lim if lim > 0 else 1.0
        ax_p.plot([0, lim], [0, lim], ls="--", lw=style.lw_thresh,
                  color=pal.threshold, zorder=2)
        ax_p.set_xlim(0, lim)
        ax_p.set_ylim(0, lim)
    else:
        ax_p.text(0.5, 0.5, "need >= 3 bursts", ha="center", va="center",
                  transform=ax_p.transAxes, fontsize=style.pt_annot,
                  color=pal.muted)
    ax_p.set_xlabel(r"IBI$_{n}$ (s)", fontsize=style.pt_tick)
    ax_p.set_ylabel(r"IBI$_{n+1}$ (s)", fontsize=style.pt_tick)
    ax_p.set_title("IBI return map", fontsize=style.pt_tick)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)

    # ---- metrics box ----
    ax_m.axis("off")
    ax_m.text(0.0, 1.0, _metrics_box_text(D), transform=ax_m.transAxes,
              family="monospace", fontsize=style.pt_annot, va="top", ha="left",
              color=pal.ink,
              bbox=dict(boxstyle="round,pad=0.6", fc=pal.paper, ec=pal.ink,
                        lw=style.lw_spine))
    ax_m.set_title("Run vs Mossink control", fontsize=style.pt_tick)
    return fig


# ===========================================================================
# Contact sheet -- a grid of mini-heroes for the top-K exemplars
# ===========================================================================
def fig_contact_sheet(runs: Sequence[RunData], style: bp.PosterStyle,
                      ncols: int = 2):
    pal = style.palette
    n = len(runs)
    if n == 0:
        raise ValueError("contact sheet needs at least one run")
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    w_in = style.width_in()
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(w_in, w_in * 0.42 * nrows / ncols),
                             squeeze=False)
    for k, D in enumerate(runs):
        ax = axes[k // ncols][k % ncols]
        N = max(D.n_neurons, 1)
        t, r = _display_ifr(D)
        ifr = _norm_to_max(r)
        ax.fill_between(t, 0.0, ifr * N, color=pal.ifr_fill, alpha=0.55, lw=0,
                        zorder=1)
        ax.scatter(D.spk_t, D.spk_i, s=style.marker_raster * 0.5, c=pal.raster,
                   marker=".", linewidths=0, zorder=3, rasterized=True)
        ax.set_xlim(0, D.t_rec)
        ax.set_ylim(0, N)
        ax.set_title(D.title or ("run %d" % k), fontsize=style.pt_tick)
        ax.tick_params(labelsize=style.pt_tick * 0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # blank any unused cells
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Top-%d burst exemplars (closest to Mossink DIV28)" % n,
                 fontsize=style.pt_title, fontweight="bold", color=pal.ink)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ===========================================================================
# Registry + save / orchestration
# ===========================================================================
FIGURES = {
    "hero":       fig_hero,
    "stacked":    fig_stacked,
    "overlay":    fig_overlay,
    "heatmap":    fig_heatmap,
    "stereotypy": fig_stereotypy,
    "returnmap":  fig_returnmap,
    "zoom":       fig_zoom,
    "composite":  fig_composite,
}
ALL_FIGURES = ("hero", "stacked", "overlay", "heatmap",
               "stereotypy", "returnmap", "zoom", "composite")
HERO_ONLY = ("hero",)


def save_figure(fig, out_base: str, style: bp.PosterStyle,
                formats: Sequence[str] = ("pdf", "png")) -> List[str]:
    """Write fig to out_base.<fmt> for each requested format; close it."""
    paths = []
    for fmt in formats:
        p = f"{out_base}.{fmt}"
        fig.savefig(p, dpi=style.dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def render_run(D: RunData, style: bp.PosterStyle, out_dir: str, basename: str,
               which: Sequence[str] = ALL_FIGURES,
               formats: Sequence[str] = ("pdf", "png")) -> List[str]:
    """Render the requested figure types for one run under one palette.

    Files are written as <out_dir>/<basename>__<palette>__<figtype>.<fmt>.
    Returns the list of written paths.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    with plt.rc_context(_valid_rc(style.rc())):
        for name in which:
            if name not in FIGURES:
                continue
            fig = FIGURES[name](D, style)
            base = os.path.join(out_dir,
                                f"{basename}__{style.palette.name}__{name}")
            written += save_figure(fig, base, style, formats)
    return written


def render_contact_sheet(runs: Sequence[RunData], style: bp.PosterStyle,
                         out_dir: str, basename: str = "contact_sheet",
                         formats: Sequence[str] = ("pdf", "png")) -> List[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(_valid_rc(style.rc())):
        fig = fig_contact_sheet(runs, style)
        base = os.path.join(out_dir, f"{basename}__{style.palette.name}")
        return save_figure(fig, base, style, formats)


# ===========================================================================
# Smoke test  (python burst_poster_plots.py)
# ===========================================================================
def _synth_bursting_run(seed: int = 3) -> RunData:
    """A clean network-bursting spike train.

    Each volley recruits ~85% of neurons, and -- crucially -- every recruited
    neuron emits a short INTRA-CELLULAR burst (7 spikes at ~10 ms ISI) so BOTH
    the population-rate detector AND the stricter log-ISI/participation detector
    fire (the latter needs single cells to burst, not merely to spike once)."""
    rng = np.random.default_rng(seed)
    N, T = 80, 60.0
    burst_times = np.array([6.0, 12.5, 19.0, 26.0, 33.0, 40.5, 47.0, 54.0])
    spikes_per_cell = 11
    isi_s = 0.006
    t, ix = [], []
    for bt in burst_times:
        part = rng.choice(N, size=int(0.90 * N), replace=False)
        for nid in part:
            t0 = bt + rng.normal(0, 0.010)
            offs = np.arange(spikes_per_cell) * isi_s + rng.normal(
                0, 0.0006, spikes_per_cell)
            t.extend(t0 + offs)
            ix.extend([int(nid)] * spikes_per_cell)
    bg = int(0.4 * N * T)
    t.extend(rng.uniform(0, T, bg)); ix.extend(rng.integers(0, N, bg))
    spk_t = np.clip(np.asarray(t, float), 0, T)
    spk_i = np.asarray(ix, int)
    o = np.argsort(spk_t); spk_t, spk_i = spk_t[o], spk_i[o]

    cfg = bm.BurstConfig()
    row = bm.compute_all(spk_t, spk_i, N, T, cfg)
    metrics = bm.scalar_row(row)
    # mean_FR_Hz is detector-independent and normally injected from the sweep
    # sidecar / CSV; compute it here so the metrics box is complete in the test.
    metrics["mean_FR_Hz"] = float(spk_t.size) / float(N * T)
    return RunData(
        spk_t=spk_t, spk_i=spk_i, n_neurons=N, t_rec=T,
        pr_intervals=row["_pr_intervals"], li_intervals=row["_li_intervals"],
        metrics=metrics, cfg=cfg, title="smoke bursting run")


def _smoke_test() -> int:
    import os
    import tempfile
    print("Smoke test -- burst_poster_plots")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        ok = ok and bool(cond)

    D = _synth_bursting_run()
    n_pr, n_li = len(D.pr_intervals), len(D.li_intervals)
    check("synthetic run detected bursts (both detectors)",
          n_pr >= 4 and n_li >= 4, f"pr={n_pr} li={n_li}")

    # display IFR sane and normalisable to [0, 1]
    _t, r = _display_ifr(D)
    ifr = _norm_to_max(r)
    check("normalized display IFR within [0, 1]",
          float(ifr.min()) >= -1e-9 and float(ifr.max()) <= 1.0 + 1e-9,
          f"min={ifr.min():.3f} max={ifr.max():.3f}")

    # recruitment order is a permutation of 0..N-1
    order = _recruitment_order(D)
    check("recruitment order is a valid permutation",
          sorted(order.tolist()) == list(range(D.n_neurons)),
          f"len={order.size}")

    out = tempfile.mkdtemp(prefix="smoke_poster_")
    # render EVERY figure type in EVERY palette; assert non-empty PDF + PNG
    total_files = 0
    for pname in bp.DEFAULT_PALETTE_ORDER:
        style = bp.make_style(pname, panel_mm=380, dpi=120)  # low dpi = fast test
        paths = render_run(D, style, out, "smoke", which=ALL_FIGURES)
        # expect 2 files (pdf+png) per figure type
        exp = 2 * len(ALL_FIGURES)
        non_empty = [p for p in paths if os.path.getsize(p) > 1200]
        check(f"{pname}: rendered {exp} non-empty files",
              len(paths) == exp and len(non_empty) == exp,
              f"got {len(paths)} files, {len(non_empty)} non-empty")
        total_files += len(paths)

    # contact sheet over a few runs
    runs = [_synth_bursting_run(seed=s) for s in (1, 2, 3, 4)]
    for i, rd in enumerate(runs):
        rd.title = "#%d" % (i + 1)
    style = bp.make_style("teal_analogous", panel_mm=380, dpi=120)
    cs = render_contact_sheet(runs, style, out)
    check("contact sheet rendered (pdf+png, non-empty)",
          len(cs) == 2 and all(os.path.getsize(p) > 1200 for p in cs))

    # a degenerate (no-burst) run must not crash any figure
    Dn = RunData(spk_t=np.array([1.0, 2.0, 3.0]), spk_i=np.array([0, 1, 2]),
                 n_neurons=5, t_rec=10.0, pr_intervals=[], li_intervals=[],
                 metrics={}, cfg=bm.BurstConfig(), title="empty run")
    try:
        style = bp.make_style("teal_amber_comp", panel_mm=250, dpi=110)
        paths = render_run(Dn, style, out, "empty", which=ALL_FIGURES)
        crash = False
    except Exception as e:
        crash = True
        print("      degenerate render raised:", repr(e))
    check("degenerate no-burst run renders without crashing", not crash)

    print(f"  [info] wrote {total_files + 2} files to {out}")
    print("\nSmoke test:", "PASS" if ok else "FAIL")
    if not ok:
        print("  (artefacts kept at", out, ")")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
