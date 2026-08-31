#!/usr/bin/env python3
"""
burst_plots.py -- render-only layer for the burst gallery.

NO detection logic and NO file discovery live here: the caller passes in the
already-detected burst intervals (from burst_metrics) and the spike arrays, and
this module only draws. That keeps the science (detection / metrics) and the
presentation (figures) fully decoupled, so a change to a detector never touches
plotting and a restyle never touches the metrics.

Each gallery figure has three stacked panels:
  (1) full-run population raster (neuron index vs time), scoring-detector bursts
      shaded;
  (2) per-neuron population rate r_pn(t) [Hz/neuron] with the threshold line and
      both detectors' burst spans shaded (so detector disagreement is visible);
  (3) a zoom on the "representative" burst -- the scoring-detector burst whose
      duration is closest to that run's median burst duration -- over a
      configurable window (default 6 s, matching the 6 s burst zoom panels in
      Mossink et al. 2019, Fig. 2g/h).

A text box reports the run's scored metrics next to the Mossink control DIV28
anchor so a reader can see at a glance why the run was selected.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np

import burst_metrics as bm


def _representative_burst(intervals: List[Tuple[float, float]]
                          ) -> Optional[Tuple[float, float]]:
    """The interval whose duration is closest to the median duration."""
    if not intervals:
        return None
    durs = np.array([e - s for s, e in intervals])
    return intervals[int(np.argmin(np.abs(durs - np.median(durs))))]


def _annotation(metrics: Dict[str, float], cfg: bm.BurstConfig,
                targets: Dict[str, Tuple[float, float]]) -> str:
    keymap = bm.scoring_keys(cfg)
    lines = ["metric          run     Mossink-ctrl(DIV28)"]
    fmt = {
        "burst_rate_per_min":        ("burst/min", "{:.2f}"),
        "burst_dur_mean_s":          ("dur (s)",   "{:.3f}"),
        "nibi_mean_s":               ("IBI (s)",   "{:.2f}"),
        "nibi_cv":                   ("CV IBI",    "{:.2f}"),
        "frac_spikes_outside_burst": ("out-frac",  "{:.2f}"),
        "mean_FR_Hz":                ("FR (Hz)",   "{:.2f}"),
    }
    for mk, (label, f) in fmt.items():
        val = metrics.get(keymap[mk], float("nan"))
        tgt_m, tgt_s = targets[mk]
        try:
            vs = f.format(val)
        except (ValueError, TypeError):
            vs = "nan"
        lines.append(f"{label:<14} {vs:>6}   {tgt_m:g} +/- {tgt_s:g}")
    return "\n".join(lines)


def plot_run(spk_t: np.ndarray, spk_i: np.ndarray, n_neurons: int, t_rec: float,
             pr_intervals: List[Tuple[float, float]],
             li_intervals: List[Tuple[float, float]],
             metrics: Dict[str, float], out_path: str,
             cfg: Optional[bm.BurstConfig] = None,
             targets: Optional[Dict[str, Tuple[float, float]]] = None,
             zoom_window_s: float = 6.0, title: str = "",
             dpi: int = 130) -> str:
    """Render the 3-panel gallery figure for one run. Returns out_path.

    pr_intervals / li_intervals are the population-rate and log-ISI/participation
    burst spans respectively; the scoring detector (cfg.scoring_detector) decides
    which spans drive the raster shading and the zoom selection.
    """
    cfg = cfg or bm.BurstConfig()
    targets = targets or bm.MOSSINK_CONTROL_DIV28
    scoring = li_intervals if cfg.scoring_detector == "logisi" else pr_intervals

    t_centers, r_pn = bm.population_rate(
        spk_t, n_neurons, t_rec, cfg.grid_dt_s, cfg.pr_smooth_sigma_s)
    mu, sd = float(r_pn.mean()), float(r_pn.std())
    theta = mu + cfg.pr_thresh_k * sd

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 3], hspace=0.32)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_r)
    ax_z = fig.add_subplot(gs[2])

    # ---- panel 1: full raster -------------------------------------------
    ax_r.scatter(spk_t, spk_i, s=1.2, c="k", marker=".", linewidths=0,
                 rasterized=True)
    for s, e in scoring:
        ax_r.axvspan(s, e, color="tab:orange", alpha=0.18, lw=0)
    ax_r.set_ylabel("neuron index")
    ax_r.set_ylim(-1, n_neurons)
    ax_r.set_xlim(0, t_rec)
    ttl = title or "burst gallery run"
    ax_r.set_title(ttl + "   (shaded = "
                   + ("log-ISI/participation" if cfg.scoring_detector == "logisi"
                      else "population-rate") + " network bursts)")

    # ---- panel 2: population rate + threshold + both detectors ----------
    ax_p.plot(t_centers, r_pn, lw=0.8, c="tab:blue")
    ax_p.axhline(theta, ls="--", lw=0.9, c="tab:red")
    ax_p.text(0.005, 0.9, "theta = mean + %.1f*std" % cfg.pr_thresh_k,
              transform=ax_p.transAxes, fontsize=7, color="tab:red", va="top")
    for s, e in li_intervals:
        ax_p.axvspan(s, e, color="tab:orange", alpha=0.22, lw=0)
    for s, e in pr_intervals:
        ax_p.axvspan(s, e, ymin=0.0, ymax=0.06, color="tab:green", alpha=0.6, lw=0)
    ax_p.set_ylabel("pop rate (Hz/neuron)")
    ax_p.set_xlabel("time (s)")
    ax_p.set_xlim(0, t_rec)

    # ---- panel 3: zoom on representative burst --------------------------
    rep = _representative_burst(scoring)
    if rep is not None:
        c = 0.5 * (rep[0] + rep[1])
        z0, z1 = c - 0.5 * zoom_window_s, c + 0.5 * zoom_window_s
        z0, z1 = max(0.0, z0), min(t_rec, z1)
        m = (spk_t >= z0) & (spk_t <= z1)
        ax_z.scatter(spk_t[m], spk_i[m], s=3.0, c="k", marker=".", linewidths=0)
        ax_z.axvspan(rep[0], rep[1], color="tab:orange", alpha=0.18, lw=0)
        ax_z.set_xlim(z0, z1)
        ax_z.set_title("representative burst zoom (%.1f s window)" % zoom_window_s)
    else:
        ax_z.text(0.5, 0.5, "no network bursts detected",
                  ha="center", va="center", transform=ax_z.transAxes)
    ax_z.set_ylabel("neuron index")
    ax_z.set_ylim(-1, n_neurons)
    ax_z.set_xlabel("time (s)")

    # ---- annotation box --------------------------------------------------
    fig.text(0.62, 0.012, _annotation(metrics, cfg, targets),
             family="monospace", fontsize=7.5,
             bbox=dict(boxstyle="round", fc="0.96", ec="0.6"))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
