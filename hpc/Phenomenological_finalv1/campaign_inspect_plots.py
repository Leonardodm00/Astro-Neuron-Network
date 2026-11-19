#!/usr/bin/env python3
"""
campaign_inspect_plots.py -- the figures of campaign_inspect.py
(specs/SPEC.md Block 14). It draws the counts that
campaign_inspect.summarize() returns and computes nothing scientific.

    write_all(summary, out_dir) -> [paths]
        fig_rate_hist.png         runs per 0.05-decade bin of the mean rate
        fig_frac_active_hist.png  runs per bin of the share of active neurons
        fig_class_shares.png      the four activity classes
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CLASS_ORDER = ("silent", "active", "high", "saturated")
CLASS_COLOURS = {"silent": "#8d99ae", "active": "#2d9a3d", "high": "#f4a259",
                 "saturated": "#b5174e"}


def _bars(ax, edges, counts, colour):
    edges = np.asarray(edges, dtype=float)
    ax.bar(edges[:-1], np.asarray(counts), width=np.diff(edges), align="edge",
           color=colour, edgecolor="none")


def plot_rate_hist(summary: Dict, path: Path) -> Path:
    h = summary["histograms"]["log10_rate"]
    th = summary["thresholds"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _bars(ax, h["edges"], h["counts"], "#3a86ff")
    marks = (("r_high %g Hz" % th["r_high_hz"], th["r_high_hz"], "#f4a259"),
             ("r_sat %g Hz" % th["r_sat_hz"], th["r_sat_hz"], "#b5174e"),
             ("1/t_ref %.0f Hz" % summary["rate_ceiling_hz"], summary["rate_ceiling_hz"], "k"))
    for label, x, colour in marks:
        ax.axvline(np.log10(x), color=colour, ls="--", lw=1.0, label=label)
    ax.set_xlabel("log10 of the mean rate per neuron [Hz]")
    ax.set_ylabel("runs")
    ax.set_title("%d runs with at least one spike; %d with none (not drawn); "
                 "%d below, %d above the range"
                 % (int(np.sum(h["counts"])), h["n_zero_rate"], h["n_below_range"],
                    h["n_above_range"]), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_frac_active_hist(summary: Dict, path: Path) -> Path:
    h = summary["histograms"]["frac_active"]
    a_min = summary["thresholds"]["a_min"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _bars(ax, h["edges"], h["counts"], "#2d9a3d")
    ax.axvline(a_min, color="#8d99ae", ls="--", lw=1.0,
               label="a_min %g: silent to the left" % a_min)
    ax.set_xlabel("share of neurons above 0.1 Hz (frac_active)")
    ax.set_ylabel("runs")
    ax.set_title("%d classified runs" % summary["runs_classified"], fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_shares(summary: Dict, path: Path) -> Path:
    cls = summary["classes"]
    n = [cls[c]["n"] for c in CLASS_ORDER]
    total = max(sum(n), 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(CLASS_ORDER))
    ax.barh(y, n, color=[CLASS_COLOURS[c] for c in CLASS_ORDER])
    for yi, ni in zip(y, n):
        ax.text(ni, yi, "  %d (%.1f %%)" % (ni, 100.0 * ni / total), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("runs")
    ax.set_xlim(0, total * 1.25)
    ax.set_title("activity classes of %d runs" % sum(n), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_all(summary: Dict, out_dir) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [plot_rate_hist(summary, out / "fig_rate_hist.png"),
            plot_frac_active_hist(summary, out / "fig_frac_active_hist.png"),
            plot_class_shares(summary, out / "fig_class_shares.png")]
