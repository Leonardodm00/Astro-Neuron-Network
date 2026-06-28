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
    # --- astrocytes (the second population; optional, drawn as a band) ---
    spk_A_t: Optional[np.ndarray] = None   # astrocyte Ca2+ event times [s]
    spk_A_i: Optional[np.ndarray] = None   # astrocyte index 0..n_astro-1
    n_astro: int = 0                       # N_astrocytes (0 -> no astro band)

    @property
    def has_astro(self) -> bool:
        return (self.n_astro > 0 and self.spk_A_t is not None
                and len(self.spk_A_t) > 0)

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


def _draw_astro_band(ax, D: RunData, style: bp.PosterStyle,
                     astro_frac: float = 0.16,
                     x0: Optional[float] = None, x1: Optional[float] = None,
                     label: bool = False, marker_scale: float = 1.3) -> float:
    """Draw astrocyte Ca2+ events as a colored band ABOVE the neuron raster.

    The neuron raster occupies y in [0, N] and the IFR envelope is mapped to the
    same [0, N]; astrocytes get a SEPARATE band [N*1.02, N*(1.02+astro_frac)] in
    the palette's astro color, so the two populations never overlap. Returns the
    new y-axis top (== N if there are no astrocytes, so callers can use it
    unconditionally). x0/x1 optionally clip events to a time window (zoom panels).
    """
    pal = style.palette
    N = float(max(D.n_neurons, 1))
    if not D.has_astro:
        return N
    t = np.asarray(D.spk_A_t, dtype=np.float64)
    i = np.asarray(D.spk_A_i, dtype=np.float64)
    if x0 is not None and x1 is not None:
        m = (t >= x0) & (t <= x1)
        t, i = t[m], i[m]
    Na = max(int(D.n_astro),
             (int(np.max(D.spk_A_i)) + 1) if len(D.spk_A_i) else 1)
    band = N * astro_frac
    y_base = N * 1.02                       # small gap above the neuron region
    y_top = y_base + band
    if t.size:
        ay = y_base + (i / max(Na - 1, 1)) * band
        ax.scatter(t, ay, s=style.marker_raster * marker_scale, c=pal.astro,
                   marker=".", linewidths=0, zorder=4, rasterized=True)
    # faint separator between the two populations
    ax.axhline(N * 1.01, color=pal.muted, lw=style.lw_spine * 0.35,
               alpha=0.6, zorder=3)
    if label:
        ax.text((x0 if x0 is not None else 0.0), y_base + 0.5 * band,
                " astro", color=pal.astro, va="center", ha="left",
                fontsize=style.pt_tick * 0.7, zorder=5)
    return y_top


def _ifr_twin(ax, ax_ymax: float, n_neurons: int, color: str, label: str):
    """Configure a twin y-axis so its 0..1 range aligns with the neuron region
    [0, N] of the host axis even when an astro band has raised ax's ymax."""
    N = float(max(n_neurons, 1))
    ax2 = ax.twinx()
    ax2.set_ylim(0.0, ax_ymax / N)          # ax y=N maps to ax2 y=1.0
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel(label)
    ax2.spines["top"].set_visible(False)
    _style_right_axis(ax2, color)
    return ax2


# ===========================================================================
# Astrocyte Ca2+ burst detection helpers
# ===========================================================================

def _detect_astro_bursts(
        D: RunData,
        astro_participation_frac: float = 0.50,
        astro_smooth_sigma_s: float = 0.20,
        astro_merge_gap_s: float = 0.50,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
           np.ndarray, np.ndarray, float, float, float]:
    """Two-detector astrocyte Ca2+ network burst detection.

    Detector 1 -- population rate:
        Smooth the Ca2+ event train with astro_smooth_sigma_s (default 0.20 s;
        Ca2+ waves unfold over ~0.5-1 s so this is ~13x wider than the
        neuronal 15 ms sigma). Threshold at theta_A = mu_A + k*sigma_A and
        merge supra-threshold epochs closer than astro_merge_gap_s (default
        0.50 s; consecutive astrocyte activations within one wave can be
        0.3-0.8 s apart -- the neuronal 0.1 s merge gap would fragment them).

    Detector 2 -- participation fraction (ADAPTED for astrocytes):
        Within each merged poprate epoch, count unique astrocytes that fired
        at least one Ca2+ event. Keep epochs where fraction >=
        astro_participation_frac (default 0.50; no intra-cellular burst
        requirement -- astrocytes produce one event per wave).

    Parameters
    ----------
    astro_smooth_sigma_s : float
        Gaussian smoothing sigma for the threshold detector [s].
    astro_merge_gap_s : float
        Max gap between supra-threshold epochs that will be merged [s].
        Should be at least as long as the typical within-wave silence.
    astro_participation_frac : float
        Active-fraction threshold for the participation detector.
    """
    if not D.has_astro:
        return [], [], np.zeros(2), np.zeros(2), 0.0, 0.0, 0.0

    from scipy.ndimage import label as _ndlabel

    cfg = D.cfg
    spk_A_t = np.asarray(D.spk_A_t, dtype=np.float64)
    spk_A_i = np.asarray(D.spk_A_i, dtype=np.int64)
    Na = max(D.n_astro, 1)

    t, r_A = bm.population_rate(spk_A_t, Na, D.t_rec,
                                cfg.grid_dt_s, astro_smooth_sigma_s)
    mu_A = float(r_A.mean())
    sd_A = float(r_A.std())
    theta_A = mu_A + cfg.pr_thresh_k * sd_A

    above = (r_A > theta_A).astype(np.int8)
    labeled, n_lbl = _ndlabel(above)
    raw: List[List[float]] = []
    for lbl in range(1, n_lbl + 1):
        idxs = np.where(labeled == lbl)[0]
        raw.append([float(t[idxs[0]]), float(t[idxs[-1]])])

    # merge with the ASTRO-appropriate gap (wider than the neuronal 0.1 s)
    merged: List[List[float]] = []
    for s, e in sorted(raw):
        if merged and (s - merged[-1][1]) <= astro_merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    pr_intervals: List[Tuple[float, float]] = []
    for s, e in merged:
        if (e - s) < cfg.pr_min_dur_s:
            continue
        if int(np.sum((spk_A_t >= s) & (spk_A_t <= e))) < cfg.pr_min_spikes:
            continue
        pr_intervals.append((float(s), float(e)))

    part_intervals: List[Tuple[float, float]] = []
    for s, e in pr_intervals:
        mk = (spk_A_t >= s) & (spk_A_t <= e)
        n_active = int(np.unique(spk_A_i[mk]).size) if mk.any() else 0
        if (n_active / Na) >= astro_participation_frac:
            part_intervals.append((s, e))

    return pr_intervals, part_intervals, t, r_A, mu_A, sd_A, theta_A


def _astro_burst_metrics(D: RunData,
                         pr_intervals: List[Tuple[float, float]],
                         part_intervals: List[Tuple[float, float]]) -> Dict[str, float]:
    """Descriptive statistics for the astrocyte Ca2+ burst population.

    Uses part_intervals as the primary criterion (richer detection); falls back
    to pr_intervals if the participation filter removed everything. Computes:
      burst_rate_per_min, burst_dur_mean_s, ibi_mean_s, ibi_cv,
      mean_participation (fraction of astrocytes per burst),
      neuron_astro_lag_mean_s / _sd_s (lag from neuronal burst onset to
      astrocyte burst onset, capped at 30 s for biologically plausible lags).
    """
    intervals = part_intervals if part_intervals else pr_intervals
    nan = float("nan")
    m: Dict[str, float] = {
        "n_bursts_pr": float(len(pr_intervals)),
        "n_bursts_part": float(len(part_intervals)),
    }
    if not intervals:
        for k in ("burst_rate_per_min", "burst_dur_mean_s", "ibi_mean_s",
                  "ibi_cv", "mean_participation",
                  "neuron_astro_lag_mean_s", "neuron_astro_lag_sd_s"):
            m[k] = nan
        return m

    starts = np.array([s for s, _ in intervals])
    stops  = np.array([e for _, e in intervals])
    durs   = stops - starts

    m["burst_rate_per_min"] = len(intervals) / (D.t_rec / 60.0)
    m["burst_dur_mean_s"]   = float(np.mean(durs))

    if len(intervals) > 1:
        ibi = starts[1:] - stops[:-1]
        m["ibi_mean_s"] = float(np.mean(ibi))
        m["ibi_cv"] = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else nan
    else:
        m["ibi_mean_s"] = m["ibi_cv"] = nan

    spk_A_t = np.asarray(D.spk_A_t, dtype=np.float64)
    spk_A_i = np.asarray(D.spk_A_i, dtype=np.int64)
    Na = max(D.n_astro, 1)
    parts = []
    for s, e in intervals:
        mk = (spk_A_t >= s) & (spk_A_t <= e)
        n_active = int(np.unique(spk_A_i[mk]).size) if mk.any() else 0
        parts.append(n_active / Na)
    m["mean_participation"] = float(np.mean(parts))

    # neuron -> astrocyte lag: for each astro burst, find the most recent
    # PRECEDING neuronal burst onset (lag must be positive and < 30 s).
    n_ivls = D.scoring_intervals
    if n_ivls:
        n_starts = np.array([s for s, _ in n_ivls])
        lags = []
        for as_ in starts:
            before = n_starts[n_starts < as_]
            if before.size:
                lag = float(as_ - before[-1])
                if 0.0 < lag < 30.0:
                    lags.append(lag)
        if lags:
            m["neuron_astro_lag_mean_s"] = float(np.mean(lags))
            m["neuron_astro_lag_sd_s"]   = float(np.std(lags))
        else:
            m["neuron_astro_lag_mean_s"] = m["neuron_astro_lag_sd_s"] = nan
    else:
        m["neuron_astro_lag_mean_s"] = m["neuron_astro_lag_sd_s"] = nan

    return m


def _astro_metrics_box_text(astro_m: Dict[str, float], D: RunData) -> str:
    """Monospace astrocyte descriptive stats + neuron vs astrocyte comparison."""
    nan = float("nan")
    keymap = bm.scoring_keys(D.cfg)
    nm = D.metrics

    def _f(v, fmt="{:.2f}"):
        try:
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return " nan"

    lines = [
        "--- Astrocyte Ca2+ bursts ---",
        "n detected (poprate)   %s" % int(astro_m.get("n_bursts_pr", 0)),
        "n detected (>=50%% part) %s" % int(astro_m.get("n_bursts_part", 0)),
        "rate (bursts/min)      %s" % _f(astro_m.get("burst_rate_per_min")),
        "mean duration (s)      %s" % _f(astro_m.get("burst_dur_mean_s"), "{:.3f}"),
        "mean IBI (s)           %s" % _f(astro_m.get("ibi_mean_s")),
        "CV IBI                 %s" % _f(astro_m.get("ibi_cv")),
        "mean participation     %s" % _f(astro_m.get("mean_participation")),
        "",
        "--- Neuron vs Astrocyte ---",
        "%-14s  neuron   astro" % "metric",
        "%-14s  %-7s  %s" % (
            "rate/min",
            _f(nm.get(keymap.get("burst_rate_per_min", ""), nan), "{:.2f}"),
            _f(astro_m.get("burst_rate_per_min"))),
        "%-14s  %-7s  %s" % (
            "dur (s)",
            _f(nm.get(keymap.get("burst_dur_mean_s", ""), nan), "{:.3f}"),
            _f(astro_m.get("burst_dur_mean_s"), "{:.3f}")),
        "%-14s  %-7s  %s" % (
            "IBI (s)",
            _f(nm.get(keymap.get("nibi_mean_s", ""), nan)),
            _f(astro_m.get("ibi_mean_s"))),
        "N->A lag (s)   %s +/- %s" % (
            _f(astro_m.get("neuron_astro_lag_mean_s")),
            _f(astro_m.get("neuron_astro_lag_sd_s"))),
    ]
    nan = float("nan")
    return "\n".join(lines)


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

    # astrocytes as a colored band above the neuron raster (auto if present)
    y_top = _draw_astro_band(ax, D, style, label=D.has_astro)

    ax.set_xlim(0, D.t_rec)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(D.channel_label)
    ax.set_title(D.title or "Raster + population IFR")
    for sp in ("top",):
        ax.spines[sp].set_visible(False)

    ax2 = _ifr_twin(ax, y_top, D.n_neurons, pal.ifr_line, "Normalized IFR")
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
# Figure 10 -- ASTRO  (dedicated astrocyte Ca2+ burst analysis composite)
# ===========================================================================
def fig_astro(D: RunData, style: bp.PosterStyle,
              astro_participation_frac: float = 0.50,
              stereotypy_window_s: float = 6.0) -> "plt.Figure":
    """Dedicated astrocyte Ca2+ network burst analysis figure.

    Returns 'no data' placeholder when D.has_astro is False, so callers
    never need to gate on it.

    Layout (2 rows x 3 cols, height_ratios=[3, 2.5]):

      Row 0 (full width)  -- HERO+COUPLING
        Astrocyte raster (pal.astro) + normalised Ca2+ rate fill/line
        (pal.astro / pal.astro_line) + neuronal IFR overlaid as a thin
        dashed line (pal.ifr_line, for coupling/lag visualisation) +
        detected burst spans (participation = full-height wash,
        poprate = floor ribbon).

      Row 1, col 0        -- ANALYTICAL
        Absolute Ca2+ rate [Hz/astrocyte] + detection threshold
        theta_A = mu_A + k*sigma_A (dashed, pal.threshold) + both
        detectors' spans (mirrors the neuronal fig_stacked lower panel).

      Row 1, col 1        -- STEREOTYPY
        Each detected Ca2+ burst aligned to its own rate peak, normalised
        to its own peak amplitude, overlaid + mean +/- std.  Mirrors
        fig_stereotypy but applied to the astrocyte population.

      Row 1, col 2        -- METRICS BOX
        Astrocyte descriptive stats (rate, duration, IBI, CV, participation)
        + side-by-side neuron vs astrocyte comparison + mean N->A lag.

    Parameters
    ----------
    astro_participation_frac : float
        Fraction of astrocytes that must be active within a poprate-detected
        epoch for it to be kept by the participation detector.  Default 0.50
        (more lenient than the neuronal 0.80 because astrocyte coupling is
        typically less all-or-nothing than neuronal participation).
    stereotypy_window_s : float
        Total window around each Ca2+ burst peak [s] for the stereotypy panel.
    """
    pal = style.palette

    # ---- degenerate: no astrocyte data -----------------------------------
    if not D.has_astro:
        fig, ax = plt.subplots(figsize=style.figsize(2.0))
        ax.text(0.5, 0.5,
                "No astrocyte data in RunData\n"
                "(populate spk_A_t, spk_A_i, n_astro)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=style.pt_annot, color=pal.muted)
        ax.axis("off")
        fig.suptitle(D.title or "Astrocyte Ca2+ burst analysis",
                     fontsize=style.pt_title, fontweight="bold", color=pal.ink)
        return fig

    Na = max(D.n_astro, 1)
    spk_A_t = np.asarray(D.spk_A_t, dtype=np.float64)
    spk_A_i = np.asarray(D.spk_A_i, dtype=np.int64)

    # ---- detection -------------------------------------------------------
    pr_ivls, part_ivls, t_A, r_A, mu_A, sd_A, theta_A = _detect_astro_bursts(
        D, astro_participation_frac=astro_participation_frac,
        astro_smooth_sigma_s=0.20, astro_merge_gap_s=0.50)
    primary_ivls = part_ivls if part_ivls else pr_ivls

    # ---- display rates (smoother sigma for visual appeal) ----------------
    t_Ad, r_Ad = bm.population_rate(
        spk_A_t, Na, D.t_rec, D.cfg.grid_dt_s, D.ifr_display_sigma_s)
    r_Ad_norm = _norm_to_max(r_Ad)

    # ---- neuronal IFR for coupling overlay (display sigma) ---------------
    t_N, r_N = _display_ifr(D)
    r_N_norm = _norm_to_max(r_N)

    # ---- metrics ---------------------------------------------------------
    astro_m = _astro_burst_metrics(D, pr_ivls, part_ivls)

    # ---- layout ----------------------------------------------------------
    fig = plt.figure(figsize=style.figsize(1.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 2.5],
                          hspace=0.70, wspace=0.40)
    ax_hero = fig.add_subplot(gs[0, :])      # top full-width
    ax_anal = fig.add_subplot(gs[1, 0])     # bottom left
    ax_ster = fig.add_subplot(gs[1, 1])     # bottom centre
    ax_box  = fig.add_subplot(gs[1, 2])     # bottom right
    sub_title_size = style.pt_label * 0.80  # smaller than hero suptitle

    # ===== PANEL A: hero + coupling =======================================
    # burst spans: participation = full-height wash; poprate = floor ribbon
    for s, e in part_ivls:
        ax_hero.axvspan(s, e, color=pal.astro, alpha=0.13, lw=0, zorder=0)
    for s, e in pr_ivls:
        ax_hero.axvspan(s, e, ymin=0.0, ymax=0.04,
                        color=pal.astro, alpha=0.70, lw=0)

    # Ca2+ normalised rate fill + outline
    ax_hero.fill_between(t_Ad, 0.0, r_Ad_norm * Na,
                         color=pal.astro, alpha=0.45, lw=0, zorder=1)
    ax_hero.plot(t_Ad, r_Ad_norm * Na, color=pal.astro_line,
                 lw=style.lw_ifr, zorder=2, label="Ca$^{2+}$ rate")

    # neuronal IFR overlay (thin dashed line in the neuronal palette colour)
    ax_hero.plot(t_N, r_N_norm * Na, color=pal.ifr_line,
                 lw=style.lw_ifr * 0.65, ls="--", zorder=3,
                 label="Neuronal IFR")

    # astrocyte raster
    ax_hero.scatter(spk_A_t, spk_A_i, s=style.marker_raster,
                    c=pal.astro, marker=".", linewidths=0,
                    alpha=0.75, zorder=4, rasterized=True)

    ax_hero.set_xlim(0, D.t_rec)
    ax_hero.set_ylim(0, Na)
    ax_hero.set_xlabel("Time (s)")
    ax_hero.set_ylabel("Astrocyte index")
    # title carried by suptitle (includes burst count); hero just labels the axes
    ax_hero.spines["top"].set_visible(False)
    ax_hero.legend(loc="upper right", fontsize=style.pt_legend * 0.85,
                   ncol=2, handlelength=1.6)

    # twin 0..1 for the normalised Ca2+ rate (both traces share this scale)
    ax2_h = ax_hero.twinx()
    ax2_h.set_ylim(0, 1.0)
    ax2_h.set_ylabel("Norm. rate")
    ax2_h.spines["top"].set_visible(False)
    _style_right_axis(ax2_h, pal.astro_line)

    # ===== PANEL B: analytical (absolute rate + threshold + spans) ========
    for s, e in part_ivls:
        ax_anal.axvspan(s, e, color=pal.astro, alpha=0.18, lw=0)
    for s, e in pr_ivls:
        ax_anal.axvspan(s, e, ymin=0.0, ymax=0.05,
                        color=pal.astro, alpha=0.80, lw=0)

    ax_anal.plot(t_A, r_A, color=pal.astro_line,
                 lw=style.lw_ifr * 0.75, zorder=3)
    ax_anal.axhline(theta_A, ls="--", lw=style.lw_thresh,
                    color=pal.threshold, zorder=4)
    ax_anal.text(0.004, 0.94,
                 r"$\theta_A=\mu_A+%.1f\,\sigma_A$" % D.cfg.pr_thresh_k,
                 transform=ax_anal.transAxes, color=pal.threshold,
                 fontsize=style.pt_annot, va="top")
    ax_anal.set_xlim(0, D.t_rec)
    ax_anal.set_xlabel("Time (s)")
    ax_anal.set_ylabel("Rate (Hz/cell)")
    ax_anal.set_title("Ca$^{2+}$ rate + threshold", fontsize=sub_title_size)
    ax_anal.spines["top"].set_visible(False)
    ax_anal.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles_b = [
        Line2D([0],[0], color=pal.astro_line, lw=style.lw_ifr*0.75,
               label="Ca$^{2+}$ rate"),
        Line2D([0],[0], color=pal.threshold, lw=style.lw_thresh, ls="--",
               label=r"$\theta_A$"),
        Patch(facecolor=pal.astro, alpha=0.5, label="participation"),
        Patch(facecolor=pal.astro, alpha=0.8, label="pop-rate"),
    ]
    ax_anal.legend(handles=handles_b, loc="upper right", ncol=2,
                   fontsize=style.pt_legend * 0.68, handlelength=1.1,
                   columnspacing=0.8)

    # ===== PANEL C: Ca2+ burst stereotypy ================================
    dt = D.cfg.grid_dt_s
    half = max(int(round(0.5 * stereotypy_window_s / dt)), 1)
    tau = np.arange(-half, half + 1) * dt
    snippets = []
    for s, e in primary_ivls:
        i0 = max(int(np.floor(s / dt)), 0)
        i1 = min(int(np.ceil(e / dt)), r_A.size - 1)
        if i1 <= i0:
            continue
        pk = i0 + int(np.argmax(r_A[i0:i1 + 1]))
        lo, hi = pk - half, pk + half + 1
        seg = np.full(tau.size, np.nan)
        a, b = max(lo, 0), min(hi, r_A.size)
        seg[(a - lo):(b - lo)] = r_A[a:b]
        pv = float(np.nanmax(seg))
        if pv > 0:
            snippets.append(seg / pv)

    if snippets:
        M = np.vstack(snippets)
        for row in M:
            ax_ster.plot(tau, row, color=pal.astro,
                         lw=style.lw_ifr * 0.35, alpha=0.40, zorder=2)
        mn = np.nanmean(M, axis=0)
        sd_s = np.nanstd(M, axis=0)
        ax_ster.fill_between(tau, mn - sd_s, mn + sd_s,
                             color=pal.astro, alpha=0.20, lw=0, zorder=1)
        ax_ster.plot(tau, mn, color=pal.astro_line,
                     lw=style.lw_ifr, zorder=3,
                     label="mean (n=%d)" % M.shape[0])
        ax_ster.axvline(0.0, ls=":", lw=style.lw_thresh * 0.8,
                        color=pal.threshold, zorder=4)
        ax_ster.legend(loc="upper right", fontsize=style.pt_legend * 0.82)
    else:
        ax_ster.text(0.5, 0.5, "< 1 detected\nastrocyte burst",
                     ha="center", va="center", transform=ax_ster.transAxes,
                     fontsize=style.pt_annot, color=pal.muted)

    ax_ster.set_xlim(-0.5 * stereotypy_window_s, 0.5 * stereotypy_window_s)
    ax_ster.set_ylim(0, 1.05)
    ax_ster.set_xlabel("Time from Ca$^{2+}$ peak (s)")
    ax_ster.set_ylabel("Rate (norm.)")
    ax_ster.set_title("Ca$^{2+}$ burst stereotypy", fontsize=sub_title_size)
    ax_ster.spines["top"].set_visible(False)
    ax_ster.spines["right"].set_visible(False)

    # ===== PANEL D: metrics box ==========================================
    ax_box.axis("off")
    ax_box.text(0.02, 0.98, _astro_metrics_box_text(astro_m, D),
                transform=ax_box.transAxes,
                family="monospace", fontsize=style.pt_annot * 0.82,
                va="top", ha="left", color=pal.ink,
                bbox=dict(boxstyle="round,pad=0.55", fc=pal.paper,
                          ec=pal.astro, lw=style.lw_spine))
    ax_box.set_title("Burst metrics", color=pal.astro_line, fontsize=sub_title_size)

    fig.subplots_adjust(left=0.09, right=0.93, top=0.90, bottom=0.08)
    fig.suptitle(
        (D.title or "Astrocyte Ca$^{2+}$ analysis")
        + "  [%d pr / %d part bursts]" % (len(pr_ivls), len(part_ivls)),
        fontsize=style.pt_label, fontweight="bold", color=pal.ink)
    return fig


# ===========================================================================
# Registry + save / orchestration
# ===========================================================================
FIGURES = {
    "hero":          fig_hero,
    "stacked":       fig_stacked,
    "overlay":       fig_overlay,
    "heatmap":       fig_heatmap,
    "stereotypy":    fig_stereotypy,
    "returnmap":     fig_returnmap,
    "zoom":          fig_zoom,
    "composite":     fig_composite,
    "burst_gallery": None,         # filled below after the function is defined
    "astro":         fig_astro,    # dedicated astrocyte figure (auto-added when has_astro)
}
ALL_FIGURES = ("hero", "stacked", "overlay", "heatmap",
               "stereotypy", "returnmap", "zoom", "composite")
HERO_ONLY    = ("hero",)
ASTRO_FIGURES = ("astro",)         # rendered automatically when D.has_astro


# ===========================================================================
# Figure 9 -- BURST GALLERY  (multiple bursts, selectable + styleable)
# ===========================================================================
# Not in ALL_FIGURES because it needs an explicit burst-selection choice.
# Call it directly: fig_burst_gallery(D, style, ...)
# or via render_run with which=("burst_gallery",) after setting the kwargs you
# want as attributes on D (see _gallery_kwarg helper below).
# ===========================================================================
def fig_burst_gallery(
        D: RunData,
        style: bp.PosterStyle,
        # ---- BURST SELECTION (choose one strategy) -----------------------
        burst_indices: Optional[List[int]] = None,
        # ^^ explicit list of integer indices into D.scoring_intervals.
        #    e.g. [0, 2, 5] picks the 1st, 3rd, and 6th detected burst.
        #    When provided, n_bursts and select_by are ignored.
        n_bursts: int = 6,
        # ^^ how many bursts to show when burst_indices is None
        select_by: str = "peak",
        # ^^ criterion when burst_indices is None:
        #      "peak"           - highest IFR peak (most synchronous)
        #      "duration"       - longest burst first
        #      "order"          - first N in temporal order (onset time)
        #      "representative" - N closest to the median burst duration
        # ---- APPEARANCE --------------------------------------------------
        zoom_window_s: float = 4.0,
        # ^^ time window centred on each burst's IFR peak [s]
        normalize_ifr: bool = True,
        # ^^ True  -> each panel's IFR is divided by its own max (shape)
        #    False -> absolute Hz/neuron (amplitude comparison across panels)
        show_raster: bool = True,
        # ^^ overlay the spike raster behind the IFR envelope
        show_astro: Optional[bool] = None,
        # ^^ draw the astrocyte event band above each panel's raster.
        #    None -> auto (on when D carries astrocyte data); True/False force it.
        color_bursts: bool = False,
        # ^^ False -> every panel uses the palette's standard ifr_fill/ifr_line
        #    True  -> cycle through a small set of perceptually distinct hues
        #             so each burst panel reads as a separate "trace"
        ncols: int = 3,
        # ^^ columns in the panel grid (rows computed automatically)
) -> "plt.Figure":
    """Gallery of individually zoomed burst panels for a single run.

    BURST SELECTION
    ---------------
    ``burst_indices`` is the most direct control: pass an explicit list of
    integer indices into ``D.scoring_intervals`` and exactly those bursts are
    shown, in the order you provide.  For example::

        # show the 1st, 4th, and 7th detected burst
        fig = fig_burst_gallery(D, style, burst_indices=[0, 3, 6])

    When ``burst_indices`` is None the function selects ``n_bursts`` bursts by
    the criterion ``select_by``:

      * ``"peak"``           -- highest IFR peak inside the burst; shows the
                                most synchronised events, useful for posters.
      * ``"duration"``       -- longest bursts first.
      * ``"order"``          -- temporal order; first N by onset time.
      * ``"representative"`` -- the N bursts whose durations are closest to the
                                median duration; a shape-stability gallery.

    APPEARANCE
    ----------
    ``zoom_window_s``   physical width of each panel in seconds.
    ``normalize_ifr``   True = each panel's IFR normalised to its own max
                        (shape comparison); False = absolute Hz/neuron
                        (amplitude comparison -- rarer on a poster but useful
                        for showing burst-to-burst amplitude variability).
    ``show_raster``     add the spike raster behind the IFR fill.
    ``color_bursts``    False (default) keeps every panel in the palette's
                        standard IFR colour so the gallery reads as one
                        coherent block; True cycles a small set of hues so
                        each burst is visually distinct (useful when you
                        paste panels at different poster positions and need
                        a legend-free colour code).
    """
    pal = style.palette
    intervals = D.scoring_intervals
    if not intervals:
        fig, ax = plt.subplots(figsize=style.figsize(2.0))
        ax.text(0.5, 0.5, "no bursts detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=style.pt_annot, color=pal.muted)
        ax.axis("off")
        fig.suptitle(D.title or "Burst gallery",
                     fontsize=style.pt_title, fontweight="bold", color=pal.ink)
        return fig

    t_full, r_full = _display_ifr(D)

    # ---- resolve which burst indices to show ----------------------------
    if burst_indices is not None:
        idx = [int(i) for i in burst_indices
               if 0 <= int(i) < len(intervals)]
    else:
        n_bursts = max(1, n_bursts)
        if select_by == "peak":
            # score each burst by the max IFR inside its span
            scores = []
            for k, (s, e) in enumerate(intervals):
                mask = (t_full >= s) & (t_full <= e)
                peak = float(r_full[mask].max()) if mask.any() else 0.0
                scores.append((peak, k))
            scores.sort(key=lambda x: -x[0])
            idx = [k for _, k in scores[:n_bursts]]
        elif select_by == "duration":
            durs = [(e - s, k) for k, (s, e) in enumerate(intervals)]
            durs.sort(key=lambda x: -x[0])
            idx = [k for _, k in durs[:n_bursts]]
        elif select_by == "representative":
            durs = np.array([e - s for s, e in intervals])
            med = float(np.median(durs))
            order = np.argsort(np.abs(durs - med), kind="stable")
            idx = list(order[:n_bursts].astype(int))
        else:                              # "order" (temporal)
            idx = list(range(min(n_bursts, len(intervals))))

    if not idx:
        idx = [0]

    # per-panel font scale: each cell is 1/ncols of the full panel width,
    # so scale point sizes gently (same exponent as PosterStyle itself).
    cell_scale = (1.0 / max(ncols, 1)) ** 0.35

    # ---- per-burst colour cycle (used when color_bursts=True) -----------
    # Four hues derived from the palette that are perceptually separable
    # (the palette's own ifr_fill, accent2, threshold, and a fourth
    # constructed as their midpoint in RGB).  Abuse of colour roles flagged.
    _c_fill = [pal.ifr_fill, pal.accent2, pal.threshold,
               "#%02x%02x%02x" % tuple(
                   int((a + b) // 2) for a, b in zip(
                       (int(pal.ifr_fill.lstrip("#")[i:i+2], 16)
                        for i in (0, 2, 4)),
                       (int(pal.threshold.lstrip("#")[i:i+2], 16)
                        for i in (0, 2, 4))))]
    # NOTE: this re-uses role colours for cycle positions 2+ (flagged here).
    _c_line = [pal.ifr_line, pal.ink, pal.ink, pal.ink]

    # ---- grid layout -----------------------------------------------------
    n = len(idx)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    w_in = style.width_in()
    # Minimum cell height ensures suptitle + panel title + data + x-axis label
    # all fit. Single-row layouts need more headroom relative to their total
    # height because the 40pt suptitle costs ~0.55" and the axes are very wide.
    cell_h_min = 3.0 if nrows == 1 else 2.0
    cell_h_in = max(w_in * 0.38 / ncols, cell_h_min)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w_in, cell_h_in * nrows),
        squeeze=False)

    half = zoom_window_s * 0.5
    N = max(D.n_neurons, 1)
    # derived point sizes for each small panel (not the figure-level rc sizes)
    pt_title_cell = style.pt_tick * cell_scale * 1.1
    pt_tick_cell  = style.pt_tick * cell_scale * 0.85
    pt_label_cell = style.pt_tick * cell_scale * 0.80

    # resolve astro display (auto when the run carries astrocyte data)
    draw_astro = D.has_astro if show_astro is None else bool(show_astro)

    # absolute mode compares amplitude ACROSS panels, so all panels must share
    # one scale: the global IFR max over every selected burst window.
    global_ifr_max = 1.0
    if not normalize_ifr:
        for bk in idx:
            sb, eb = intervals[bk]
            wmask = (t_full >= sb) & (t_full <= eb)
            if wmask.any():
                global_ifr_max = max(global_ifr_max, float(r_full[wmask].max()))

    for panel, burst_k in enumerate(idx):
        ax = axes[panel // ncols][panel % ncols]
        s_burst, e_burst = intervals[burst_k]

        # centre on IFR peak inside the burst
        win_mask = (t_full >= s_burst) & (t_full <= e_burst)
        if win_mask.any():
            peak_t = float(t_full[win_mask][np.argmax(r_full[win_mask])])
        else:
            peak_t = 0.5 * (s_burst + e_burst)
        z0 = max(0.0, peak_t - half)
        z1 = min(D.t_rec, peak_t + half)

        # IFR in window. Map onto the neuron region [0, N] in BOTH modes; the
        # twin axis carries the units. normalize_ifr -> per-panel max (shape);
        # else -> shared global max (cross-panel amplitude comparison).
        wm = (t_full >= z0) & (t_full <= z1)
        tt, rr = t_full[wm], r_full[wm]
        if normalize_ifr:
            ref = float(rr.max()) if rr.size and rr.max() > 0 else 1.0
            twin_top, twin_lbl = 1.0, "Norm. IFR"
        else:
            ref = global_ifr_max
            twin_top, twin_lbl = global_ifr_max, "Hz/neuron"
        disp = rr / ref if ref > 0 else rr           # in [0, 1] (or <=1 for abs)

        # colours for this panel
        ci = panel % len(_c_fill)
        fill_c = _c_fill[ci] if color_bursts else pal.ifr_fill
        line_c = _c_line[ci] if color_bursts else pal.ifr_line

        # IFR envelope on the neuron-region scale (0..N)
        ax.fill_between(tt, 0.0, disp * N, color=fill_c,
                        alpha=0.55, lw=0, zorder=1)
        ax.plot(tt, disp * N, color=line_c,
                lw=style.lw_ifr * cell_scale, zorder=2)
        ax.axvspan(s_burst, e_burst, color=pal.span_li
                   if D.cfg.scoring_detector == "logisi" else pal.span_pr,
                   alpha=0.13, lw=0, zorder=0)

        if show_raster:
            m = (D.spk_t >= z0) & (D.spk_t <= z1)
            ax.scatter(D.spk_t[m], D.spk_i[m],
                       s=style.marker_raster * cell_scale,
                       c=pal.raster, marker=".", linewidths=0,
                       zorder=3, rasterized=True)

        # astrocyte band above the raster (clipped to this panel's window)
        if draw_astro:
            y_axis_top = _draw_astro_band(
                ax, D, style, x0=z0, x1=z1,
                label=(panel % ncols == 0), marker_scale=cell_scale * 1.2)
        else:
            y_axis_top = N

        ax.set_xlim(z0, z1)
        ax.set_ylim(0, y_axis_top)
        ax.set_title("#%d  t=%.1fs  dur=%.3fs" % (
                     burst_k + 1, s_burst, e_burst - s_burst),
                     fontsize=pt_title_cell, color=pal.ink, pad=2)
        ax.tick_params(labelsize=pt_tick_cell)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # IFR right-axis aligned so host y=N maps to twin y=twin_top
        ax2 = ax.twinx()
        ax2.set_ylim(0, (y_axis_top / N) * twin_top)
        ax2.set_yticks(np.linspace(0, twin_top, 5))
        ax2.spines["top"].set_visible(False)
        _style_right_axis(ax2, line_c)
        ax2.tick_params(labelsize=pt_tick_cell * 0.9)
        if panel % ncols == ncols - 1 or panel == len(idx) - 1:
            ax2.set_ylabel(twin_lbl, fontsize=pt_label_cell, color=line_c)

        # x/y axis labels only on border panels
        if panel // ncols == nrows - 1:
            ax.set_xlabel("Time (s)", fontsize=pt_label_cell)
        if panel % ncols == 0:
            ax.set_ylabel(D.channel_label, fontsize=pt_label_cell)

    # blank unused cells
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    sel_label = (("indices %s" % burst_indices) if burst_indices is not None
                 else ("top-%d by %s" % (n, select_by)))
    fig.suptitle((D.title or "Burst gallery") + "  [%s]" % sel_label,
                 fontsize=style.pt_label, fontweight="bold", color=pal.ink,
                 y=1.0, va="top")
    # Reserve space for suptitle (pt_label, not pt_title, since internal panel
    # titles already carry per-burst annotations; a full pt_title would compete).
    top_margin = {1: 0.80, 2: 0.84}.get(nrows, 0.90)
    fig.subplots_adjust(top=top_margin, hspace=0.55, wspace=0.38)
    return fig


FIGURES["burst_gallery"] = fig_burst_gallery


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

    # synthetic astrocytes: sparse Ca2+ events, a fraction of them LAGGING the
    # network bursts by ~0.3-1.2 s (astrocytes respond after neuronal volleys).
    Na = 40
    at, ai = [], []
    burst_times = np.array([6.0, 12.5, 19.0, 26.0, 33.0, 40.5, 47.0, 54.0])
    for bt in burst_times:
        # 80% recruitment, tighter window (0.30-0.70 s lag) so the merge gap
        # can capture the full population response as one epoch
        responders = rng.choice(Na, size=int(0.80 * Na), replace=False)
        for aid in responders:
            at.append(bt + rng.uniform(0.30, 0.70)); ai.append(int(aid))
    n_bg_a = int(0.3 * Na)                       # sparse spontaneous events
    at.extend(rng.uniform(0, T, n_bg_a)); ai.extend(rng.integers(0, Na, n_bg_a))
    spk_A_t = np.clip(np.asarray(at, float), 0, T)
    spk_A_i = np.asarray(ai, int)

    return RunData(
        spk_t=spk_t, spk_i=spk_i, n_neurons=N, t_rec=T,
        pr_intervals=row["_pr_intervals"], li_intervals=row["_li_intervals"],
        metrics=metrics, cfg=cfg, title="smoke bursting run",
        spk_A_t=spk_A_t, spk_A_i=spk_A_i, n_astro=Na)


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

    # burst gallery: every selection mode + astro band on/off
    D2 = _synth_bursting_run()
    check("fixture carries astrocyte data", D2.has_astro,
          f"n_astro={D2.n_astro} events={0 if D2.spk_A_t is None else len(D2.spk_A_t)}")
    gal_ok = True
    style = bp.make_style("teal_analogous", panel_mm=380, dpi=110)
    with plt.rc_context(_valid_rc(style.rc())):
        for kw in (dict(n_bursts=6, select_by="peak", ncols=3),
                   dict(burst_indices=[0, 2, 4], color_bursts=True),
                   dict(n_bursts=3, select_by="duration", normalize_ifr=False),
                   dict(n_bursts=4, select_by="representative", show_astro=False),
                   dict(n_bursts=4, show_astro=True)):
            try:
                fig = fig_burst_gallery(D2, style, **kw)
                p = os.path.join(out, "gal_%d.png" % abs(hash(str(kw))))
                fig.savefig(p, dpi=80); plt.close(fig)
                gal_ok = gal_ok and os.path.getsize(p) > 1200
            except Exception as e:
                gal_ok = False
                print("      gallery raised for", kw, "->", repr(e))
    check("burst gallery: all selection modes + astro on/off render", gal_ok)

    # astro band must lift the hero's y-axis above N (band drawn above neurons)
    with plt.rc_context(_valid_rc(style.rc())):
        fig = fig_hero(D2, style)
        top = fig.axes[0].get_ylim()[1]
        plt.close(fig)
    check("hero y-axis extends above N when astro present (band drawn)",
          top > D2.n_neurons, f"ymax={top:.1f} N={D2.n_neurons}")

    # dedicated astrocyte figure: detection + all four panels
    with plt.rc_context(_valid_rc(style.rc())):
        pr_ivls, part_ivls, _t, _r, _mu, _sd, _th = _detect_astro_bursts(D2)
        check("astro poprate detector fires on fixture",
              len(pr_ivls) >= 4, f"pr_ivls={len(pr_ivls)}")
        check("astro participation detector fires on fixture (>=50%% threshold)",
              len(part_ivls) >= 4, f"part_ivls={len(part_ivls)}")
        fig = fig_astro(D2, style)
        p = os.path.join(out, "astro_full.png")
        fig.savefig(p, dpi=80); plt.close(fig)
        check("fig_astro renders non-empty PNG", os.path.getsize(p) > 1200)
        # no-astro guard
        Dn2 = RunData(spk_t=np.array([1.0]), spk_i=np.array([0]),
                      n_neurons=5, t_rec=10.0, pr_intervals=[], li_intervals=[],
                      metrics={}, cfg=bm.BurstConfig())
        try:
            fig = fig_astro(Dn2, style); plt.close(fig); crash_a = False
        except Exception as e:
            crash_a = True; print("      fig_astro no-data crash:", e)
        check("fig_astro no-astro-data placeholder renders without crash",
              not crash_a)

    print(f"  [info] wrote {total_files + 2} files to {out}")
    print("\nSmoke test:", "PASS" if ok else "FAIL")
    if not ok:
        print("  (artefacts kept at", out, ")")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
