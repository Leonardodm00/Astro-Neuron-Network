#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_campaign_report.py
===================================================================
Cross-campaign aggregator for the CAdEx neuron-astrocyte SBI sweeps.

This is a THIN aggregation layer on top of campaign_report.py. It does NOT
re-implement any per-iter / per-topo feature extraction: it delegates each
campaign to campaign_report.analyze_campaign() (the single, already-validated
source of truth), then assembles a cohort-level comparison from the per-campaign
roll-ups those calls return.

TWO FULLY-SEPARATE OUTPUT LAYERS
--------------------------------
  (A) Per-campaign layer  (unchanged from campaign_report.py)
        <campaign_i>/report/campaign_report.md
        <campaign_i>/report/campaign_summary.json
        <campaign_i>/report/figures/*.png
  (B) Cohort layer        (new; written under --out)
        <out>/multi_campaign_report.md      cross-campaign comparison (verbose)
        <out>/multi_campaign_summary.json   machine-readable cohort roll-up
        <out>/figures/*.png                 overlay / grouped comparison figures

COHORT-LAYER CONTENT (all four requested comparison facets)
-----------------------------------------------------------
  1. Campaign-keyed metrics table  (useful-yield, one-spike-artifact frac,
     corr(sigma,logFR), corr(I_inj,logFR), mean in-degree K, regime mix, ...).
  2. Overlay / grouped figures  (grouped regime-fraction bars; per-campaign
     useful-yield bars; sigma- and I_inj-predictivity dot plots across campaigns;
     mean in-degree vs useful-yield scatter).
  3. Pooled correlation with `campaign` as a grouping factor: reports both the
     distribution of WITHIN-campaign Spearman rho (already computed per campaign
     by campaign_report._roll_up, hence free of cross-campaign confounds) and a
     COHORT-LEVEL rho across campaign medians, so a reader can see whether the
     driver->rate relationship is consistent within campaigns vs across them.
  4. Per-campaign verdict deltas: each campaign's headline metrics expressed as
     a signed deviation from the cohort median, with the known-failure-mode
     verdict lines (reused verbatim from campaign_report._verdict_lines).

DISCOVERY
---------
A "campaign" is operationally any directory that (a) matches campaign_* / sweep_*,
or (b) directly or recursively contains topo_* subdirectories. We pass each such
directory unchanged to campaign_report.analyze_campaign(), whose own
_discover_topologies() already tolerates both the flat (out_dir/topo_*) and the
nested (sweep_*_task*/topo_*) on-disk layouts.

Run on HPC
----------
    python multi_campaign_report.py /scratch/<user>/sweeps_parent
    python multi_campaign_report.py PARENT --out ./cohort_report
    python multi_campaign_report.py PARENT --glob 'campaign_*' --topo-sample 50
    python multi_campaign_report.py PARENT --max-iters-per-topo 200 --jobs 4
    python multi_campaign_report.py --smoke-test       # synthetic self-check

Notes
-----
* Pure numpy / scipy / matplotlib + stdlib; reuses campaign_report.py wholesale.
* Per-campaign analysis can run in parallel processes (--jobs N). Each campaign
  writes its own report independently, so failures are isolated and COUNTED,
  never fatal to the cohort run.
===================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import campaign_report as cr
except ImportError as exc:                                       # pragma: no cover
    raise SystemExit(
        "multi_campaign_report.py must sit next to campaign_report.py "
        "(could not import it): " + str(exc)
    )

REGIME_ORDER = ['one_spike_artifact', 'silent', 'low', 'useful', 'seizing']


# ======================================================================
# 1. Discovery
# ======================================================================
def _dir_has_topo(d: Path) -> bool:
    for pat in ('topo_*', 'sweep_*_task*/topo_*', '**/topo_*'):
        for p in d.glob(pat):
            if p.is_dir():
                return True
    return False


def discover_campaigns(parent: Path, glob_pat: str = 'campaign_*') -> List[Path]:
    parent = parent.resolve()
    for pat in (glob_pat, 'sweep_*'):
        cands = sorted(c for c in parent.glob(pat)
                       if c.is_dir() and _dir_has_topo(c))
        if cands:
            return cands
    cands = sorted(c for c in parent.iterdir()
                   if c.is_dir() and _dir_has_topo(c))
    if cands:
        return cands
    if _dir_has_topo(parent):
        return [parent]
    return []


# ======================================================================
# 2. Per-campaign delegation
# ======================================================================
def _run_one_campaign(campaign: str,
                      max_iters_per_topo: int,
                      topo_sample: int) -> Tuple[str, Optional[Dict], Optional[str]]:
    cpath = Path(campaign)
    try:
        summary = cr.analyze_campaign(
            cpath, cpath / 'report',
            max_iters_per_topo=max_iters_per_topo,
            topo_sample=topo_sample,
        )
        return (campaign, summary, None)
    except Exception as exc:
        return (campaign, None, f'{type(exc).__name__}: {exc}')


def analyze_all_campaigns(campaigns: List[Path],
                          max_iters_per_topo: int = 0,
                          topo_sample: int = 0,
                          jobs: int = 1
                          ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str]]]:
    ok: List[Tuple[str, Dict]] = []
    failed: List[Tuple[str, str]] = []
    if jobs and jobs > 1 and len(campaigns) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_run_one_campaign, str(c),
                              max_iters_per_topo, topo_sample): c
                    for c in campaigns}
            for fut in as_completed(futs):
                cpath, summary, err = fut.result()
                (ok if err is None else failed).append(
                    (cpath, summary if err is None else err))
    else:
        for c in campaigns:
            cpath, summary, err = _run_one_campaign(
                str(c), max_iters_per_topo, topo_sample)
            (ok if err is None else failed).append(
                (cpath, summary if err is None else err))
    ok.sort(key=lambda t: t[0])
    failed.sort(key=lambda t: t[0])
    return ok, failed


# ======================================================================
# 3. Cohort aggregation
# ======================================================================
@dataclass
class CampaignRow:
    name: str
    path: str
    n_iters: int
    n_topologies: int
    useful_frac: float
    one_spike_artifact_frac: float
    fr_median: float
    corr_sigma_logFR: float
    corr_iinj_logFR: float
    mean_in_degree: float
    frac_zero_in: float
    largest_component_frac: float
    seizing_frac: float
    silent_frac: float


def _row_from_summary(name: str, path: str, roll: Dict) -> CampaignRow:
    rf = roll.get('regime_fractions', {})
    topo = roll.get('topology', {})
    return CampaignRow(
        name=name, path=path,
        n_iters=int(roll.get('n_iters', 0)),
        n_topologies=int(roll.get('n_topologies', 0)),
        useful_frac=float(rf.get('useful', 0.0)),
        one_spike_artifact_frac=float(roll.get('one_spike_artifact_frac', 0.0)),
        fr_median=float(roll.get('fr_median', 0.0)),
        corr_sigma_logFR=float(roll.get('corr_sigma_logFR', float('nan'))),
        corr_iinj_logFR=float(roll.get('corr_iinj_logFR', float('nan'))),
        mean_in_degree=float(topo.get('mean_in_degree', float('nan'))),
        frac_zero_in=float(topo.get('frac_zero_in', float('nan'))),
        largest_component_frac=float(topo.get('mean_largest_component_frac', float('nan'))),
        seizing_frac=float(rf.get('seizing', 0.0)),
        silent_frac=float(rf.get('silent', 0.0)),
    )


def _nanmedian(xs: List[float]) -> float:
    a = np.array([x for x in xs if x is not None and np.isfinite(x)], float)
    return float(np.median(a)) if a.size else float('nan')


def _pooled_corr_across_campaigns(rows: List[CampaignRow]) -> Dict:
    out: Dict = {}
    for drv, key in (('sigma', 'corr_sigma_logFR'),
                     ('I_inj', 'corr_iinj_logFR')):
        within = [getattr(r, key) for r in rows]
        within_finite = [v for v in within if v is not None and np.isfinite(v)]
        out[drv] = dict(
            within_campaign_rho=within,
            within_campaign_rho_median=_nanmedian(within),
            within_campaign_rho_iqr=(
                [float(np.percentile(within_finite, 25)),
                 float(np.percentile(within_finite, 75))]
                if len(within_finite) >= 2 else [float('nan'), float('nan')]),
            n_campaigns_finite=len(within_finite),
        )
    fr_med = np.array([r.fr_median for r in rows], float)
    valid_fr = fr_med > 0
    out['between_campaign'] = {}
    try:
        from scipy.stats import spearmanr
        kin = np.array([r.mean_in_degree for r in rows], float)
        m = valid_fr & np.isfinite(kin)
        if m.sum() >= 3:
            rho = spearmanr(kin[m], np.log(fr_med[m]))[0]
            out['between_campaign']['rho_meanKin_logFRmedian'] = float(rho)
        else:
            out['between_campaign']['rho_meanKin_logFRmedian'] = float('nan')
        out['between_campaign']['n'] = int(m.sum())
    except Exception:
        out['between_campaign']['rho_meanKin_logFRmedian'] = float('nan')
        out['between_campaign']['n'] = 0
    return out


def build_cohort(ok: List[Tuple[str, Dict]]):
    rows: List[CampaignRow] = []
    for path, summary in ok:
        roll = summary.get('roll_up', {})
        rows.append(_row_from_summary(Path(path).name, path, roll))
    cohort_median = dict(
        useful_frac=_nanmedian([r.useful_frac for r in rows]),
        one_spike_artifact_frac=_nanmedian([r.one_spike_artifact_frac for r in rows]),
        corr_sigma_logFR=_nanmedian([r.corr_sigma_logFR for r in rows]),
        corr_iinj_logFR=_nanmedian([r.corr_iinj_logFR for r in rows]),
        mean_in_degree=_nanmedian([r.mean_in_degree for r in rows]),
        fr_median=_nanmedian([r.fr_median for r in rows]),
        seizing_frac=_nanmedian([r.seizing_frac for r in rows]),
    )
    cohort = dict(
        n_campaigns=len(rows),
        total_iters=int(sum(r.n_iters for r in rows)),
        total_topologies=int(sum(r.n_topologies for r in rows)),
        rows=[r.__dict__ for r in rows],
        cohort_median=cohort_median,
        grouped_correlation=_pooled_corr_across_campaigns(rows),
    )
    return cohort, rows


# ======================================================================
# 4. Cohort figures
# ======================================================================
def _regime_frac(row: CampaignRow, regime: str) -> float:
    if regime == 'useful':
        return row.useful_frac
    if regime == 'one_spike_artifact':
        return row.one_spike_artifact_frac
    if regime == 'seizing':
        return row.seizing_frac
    if regime == 'silent':
        return row.silent_frac
    if regime == 'low':
        return max(0.0, 1.0 - (row.useful_frac + row.one_spike_artifact_frac
                               + row.seizing_frac + row.silent_frac))
    return 0.0


def make_cohort_figures(rows: List[CampaignRow], fig_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    if not rows:
        return made
    names = [r.name for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows) + 3), 4.5))
    cols = ['#999999', '#332288', '#88ccee', '#117733', '#cc6677']
    bottom = np.zeros(len(rows))
    for reg, col in zip(REGIME_ORDER, cols):
        vals = np.array([_regime_frac(r, reg) for r in rows])
        ax.bar(x, vals, bottom=bottom, color=col, label=reg, width=0.8)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('fraction of sims'); ax.set_ylim(0, 1)
    ax.set_title('Regime mix per campaign (stacked)')
    ax.legend(fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(); p = fig_dir / 'cohort_regime_mix.png'
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); made.append(str(p))

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows) + 3), 4.5))
    w = 0.4
    ax.bar(x - w / 2, [r.useful_frac for r in rows], width=w,
           color='#117733', label='useful-band yield')
    ax.bar(x + w / 2, [r.one_spike_artifact_frac for r in rows], width=w,
           color='#999999', label='one-spike artifact')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('fraction'); ax.set_ylim(0, 1)
    ax.set_title('Useful-band yield vs startup artifact per campaign')
    ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / 'cohort_yield_vs_artifact.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows) + 3), 4.5))
    cs = np.array([r.corr_sigma_logFR for r in rows], float)
    ci = np.array([r.corr_iinj_logFR for r in rows], float)
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(x, cs, 'o-', color='#cc6677', label=r'$\rho(\sigma,\log FR\mid$campaign$)$')
    ax.plot(x, ci, 's-', color='#4477aa', label=r'$\rho(I_{inj},\log FR\mid$campaign$)$')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('within-campaign Spearman rho'); ax.set_ylim(-1, 1)
    ax.set_title('Driver -> rate predictivity per campaign (~0 = recurrence-driven)')
    ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / 'cohort_driver_predictivity.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    kin = np.array([r.mean_in_degree for r in rows], float)
    frm = np.array([r.fr_median for r in rows], float)
    m = np.isfinite(kin) & (frm > 0)
    if m.sum() >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(kin[m], frm[m], s=40, color='#882255')
        for xi, yi, nm in zip(kin[m], frm[m], np.array(names)[m]):
            ax.annotate(nm, (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_yscale('log'); ax.set_xlabel('mean in-degree K (per campaign)')
        ax.set_ylabel('median mean_FR (Hz)')
        ax.set_title('Between-campaign: wiring density vs network rate')
        fig.tight_layout(); p = fig_dir / 'cohort_Kin_vs_rate.png'
        fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))
    return made


# ======================================================================
# 5. Cohort markdown
# ======================================================================
def _fmt(v, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    return f'{v:.{nd}f}'


def _signed(v, ref, nd: int = 2) -> str:
    if not (np.isfinite(v) and np.isfinite(ref)):
        return 'n/a'
    return f'{(v - ref):+.{nd}f}'


def write_cohort_markdown(out: Path, parent: Path, cohort: Dict,
                          rows: List[CampaignRow],
                          ok: List[Tuple[str, Dict]],
                          failed: List[Tuple[str, str]],
                          figs: List[str], elapsed: float) -> None:
    L: List[str] = []
    A = L.append
    cm = cohort['cohort_median']
    gc = cohort['grouped_correlation']

    A(f"# Cohort report — `{parent.name}`  ({cohort['n_campaigns']} campaigns)\n")
    A(f"_Generated by multi_campaign_report.py in {elapsed:.1f}s. Per-campaign "
      f"reports are written separately under each `<campaign>/report/`; this file "
      f"is the cross-campaign comparison layer only._\n")

    A("## 1. Cohort verdict\n")
    A(f"- campaigns analysed: **{cohort['n_campaigns']}**  "
      f"(total {cohort['total_iters']} sims across "
      f"{cohort['total_topologies']} topologies)")
    A(f"- cohort-median useful-band yield: **{cr._pct(cm['useful_frac'])}**; "
      f"cohort-median startup-artifact fraction: **{cr._pct(cm['one_spike_artifact_frac'])}**")
    A(f"- within-campaign predictivity (median rho): "
      f"sigma **{_fmt(gc['sigma']['within_campaign_rho_median'])}**, "
      f"I_inj **{_fmt(gc['I_inj']['within_campaign_rho_median'])}**  "
      f"(both ~0 => recurrence-driven across the cohort)")
    A(f"- between-campaign wiring trend: "
      f"rho(mean K_in, log FR_median | campaign) = "
      f"**{_fmt(gc['between_campaign']['rho_meanKin_logFRmedian'])}** "
      f"over {gc['between_campaign']['n']} campaigns")
    if failed:
        A(f"- **{len(failed)} campaign(s) failed** and were excluded (see section 6).")
    A("")

    A("## 2. Campaign-keyed metrics table\n")
    A("| campaign | sims | topos | useful | artifact | silent | seizing | "
      "FR_med(Hz) | rho(sig) | rho(Iinj) | K_in | %zero_in | comp_frac |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r.name} | {r.n_iters} | {r.n_topologies} | "
          f"{cr._pct(r.useful_frac)} | {cr._pct(r.one_spike_artifact_frac)} | "
          f"{cr._pct(r.silent_frac)} | {cr._pct(r.seizing_frac)} | "
          f"{_fmt(r.fr_median, 3)} | {_fmt(r.corr_sigma_logFR)} | "
          f"{_fmt(r.corr_iinj_logFR)} | {_fmt(r.mean_in_degree, 1)} | "
          f"{cr._pct(r.frac_zero_in) if np.isfinite(r.frac_zero_in) else 'n/a'} | "
          f"{_fmt(r.largest_component_frac)} |")
    A("")

    A("## 3. Driver->rate predictivity with `campaign` as grouping factor\n")
    for drv in ('sigma', 'I_inj'):
        g = gc[drv]; iqr = g['within_campaign_rho_iqr']
        A(f"- **{drv}**: within-campaign rho({drv}, log FR | campaign=c) median "
          f"**{_fmt(g['within_campaign_rho_median'])}** "
          f"(IQR [{_fmt(iqr[0])}, {_fmt(iqr[1])}]) across "
          f"{g['n_campaigns_finite']} campaigns with a finite estimate.")
    A(f"- **between-campaign** trend (each campaign = one observation): "
      f"rho(mean K_in, log median FR) = "
      f"**{_fmt(gc['between_campaign']['rho_meanKin_logFRmedian'])}** "
      f"(n={gc['between_campaign']['n']}).")
    A("\n> Boundary note: a true fixed-effect pooled within-campaign rho needs "
      "per-iter driver/FR samples, which the per-campaign summary discards after "
      "roll-up. The within-campaign rho DISTRIBUTION above is the confound-free "
      "object; the between-campaign rho isolates the cross-campaign trend.\n")

    A("## 4. Per-campaign verdict deltas (vs cohort median)\n")
    A("| campaign | Δuseful | Δartifact | Δrho(sig) | Δrho(Iinj) | ΔK_in | Δseizing |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r.name} | {_signed(r.useful_frac, cm['useful_frac'])} | "
          f"{_signed(r.one_spike_artifact_frac, cm['one_spike_artifact_frac'])} | "
          f"{_signed(r.corr_sigma_logFR, cm['corr_sigma_logFR'])} | "
          f"{_signed(r.corr_iinj_logFR, cm['corr_iinj_logFR'])} | "
          f"{_signed(r.mean_in_degree, cm['mean_in_degree'], 1)} | "
          f"{_signed(r.seizing_frac, cm['seizing_frac'])} |")
    A("")
    A("Per-campaign failure-mode verdicts (reused from campaign_report):\n")
    for path, summary in ok:
        A(f"### `{Path(path).name}`")
        for v in cr._verdict_lines(summary.get('roll_up', {})):
            A(v)
        A("")

    if figs:
        A("## 5. Cohort figures\n")
        for f in figs:
            A(f"- `figures/{Path(f).name}`")
        A("")

    A("## 6. Failed / skipped campaigns\n")
    if failed:
        for path, err in failed:
            A(f"- `{Path(path).name}` — {err}")
    else:
        A("- none")
    A("")
    A("## 7. Per-campaign report locations\n")
    for path, _ in ok:
        A(f"- `{path}/report/campaign_report.md`")
    A("")
    out.write_text("\n".join(L))


# ======================================================================
# 6. Driver
# ======================================================================
def analyze_cohort(parent: Path, out_dir: Path,
                   glob_pat: str = 'campaign_*',
                   max_iters_per_topo: int = 0,
                   topo_sample: int = 0,
                   jobs: int = 1) -> Dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    campaigns = discover_campaigns(parent, glob_pat)
    if not campaigns:
        raise SystemExit(
            f"no campaigns found under {parent} "
            f"(looked for {glob_pat!r}, 'sweep_*', and any dir containing topo_*)")
    ok, failed = analyze_all_campaigns(
        campaigns, max_iters_per_topo=max_iters_per_topo,
        topo_sample=topo_sample, jobs=jobs)
    cohort, rows = build_cohort(ok)
    figs = make_cohort_figures(rows, fig_dir)
    elapsed = time.time() - t0
    write_cohort_markdown(out_dir / 'multi_campaign_report.md', parent,
                          cohort, rows, ok, failed, figs, elapsed)
    summary = dict(
        parent=str(parent), n_campaigns_found=len(campaigns),
        n_ok=len(ok), n_failed=len(failed),
        failed=[{'campaign': p, 'error': e} for p, e in failed],
        elapsed_s=elapsed, figures=[Path(f).name for f in figs],
        cohort=cohort,
        per_campaign_reports=[str(Path(p) / 'report' / 'campaign_report.md')
                              for p, _ in ok],
    )
    (out_dir / 'multi_campaign_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# ======================================================================
# 7. Smoke test
# ======================================================================
def _build_synth_campaign(camp: Path, rng, *, Nn=200, T=60.0, K=15, specs=None) -> None:
    td = camp / 'sweep_cpu_task0000' / 'topo_00000'
    td.mkdir(parents=True)
    n_syn = Nn * K
    S_i = rng.integers(0, Nn, n_syn).astype(np.int32)
    S_j = rng.integers(0, Nn, n_syn).astype(np.int32)
    keep = S_i != S_j
    S_i, S_j = S_i[keep], S_j[keep]
    N_pos = rng.uniform(0, 700, (Nn, 2))
    np.savez_compressed(td / 'topology.npz', N_pos=N_pos, S_i=S_i, S_j=S_j,
                        A_pos=np.empty((0, 2)))
    (td / 'topology_meta.json').write_text(json.dumps(
        dict(Nn=Nn, Na=0, T=T, mode='Neuronal', topo_idx=0,
             conn_rule='weibull', p0_conn=0.6, d0_conn=100.0, beta_conn=1.0,
             c_max=700.0)))
    if specs is None:
        specs = [('silent', 0.0), ('one_spike', 1.0),
                 ('useful', 1.5), ('seize', 8.0)]
    for k, (name, rate) in enumerate(specs):
        params = np.zeros(35)
        params[cr.IDX['Sigma']] = rng.uniform(1, 12)
        params[cr.IDX['I_inj']] = rng.uniform(1, 8)
        params[cr.IDX['g_ampa']] = rng.uniform(0.05, 1.0)
        if name == 'one_spike':
            spk_t = rng.uniform(0, 0.05, Nn); spk_i = np.arange(Nn)
        elif rate == 0.0:
            spk_t = np.empty(0); spk_i = np.empty(0, int)
        else:
            n = int(rate * Nn * T)
            spk_t = np.sort(rng.uniform(0, T, n))
            spk_i = rng.integers(0, Nn, n)
        np.savez_compressed(td / f'iter_{k:05d}.npz',
                            spk_N_t=spk_t.astype(np.float32),
                            spk_N_i=spk_i.astype(np.int32),
                            spk_A_t=np.empty(0, np.float32),
                            spk_A_i=np.empty(0, np.int32),
                            params=params, theta=params)


def _smoke_test() -> int:
    import tempfile
    print("[smoke] building synthetic PARENT with two distinct campaigns ...")
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    parent = tmp / 'sweeps_parent'; parent.mkdir()
    _build_synth_campaign(parent / 'campaign_A', rng, K=15)
    _build_synth_campaign(parent / 'campaign_B', rng, K=4,
                          specs=[('silent', 0.0), ('silent', 0.0),
                                 ('one_spike', 1.0), ('useful', 1.2)])
    summary = analyze_cohort(parent, tmp / 'cohort_report', jobs=1)
    c = summary['cohort']; rows = c['rows']
    print(f"[smoke] campaigns_found={summary['n_campaigns_found']} "
          f"ok={summary['n_ok']} failed={summary['n_failed']} "
          f"total_iters={c['total_iters']}")
    for r in rows:
        print(f"        {r['name']}: useful={r['useful_frac']:.2f} "
              f"K_in={r['mean_in_degree']:.1f} "
              f"artifact={r['one_spike_artifact_frac']:.2f}")
    by = {r['name']: r for r in rows}
    cohort_md = tmp / 'cohort_report' / 'multi_campaign_report.md'
    cohort_js = tmp / 'cohort_report' / 'multi_campaign_summary.json'
    ok = (summary['n_campaigns_found'] == 2 and summary['n_ok'] == 2
          and summary['n_failed'] == 0 and c['n_campaigns'] == 2
          and c['total_iters'] == 8
          and 'campaign_A' in by and 'campaign_B' in by
          and by['campaign_A']['mean_in_degree'] > by['campaign_B']['mean_in_degree']
          and cohort_md.exists() and cohort_js.exists()
          and (parent / 'campaign_A' / 'report' / 'campaign_report.md').exists()
          and (parent / 'campaign_B' / 'report' / 'campaign_report.md').exists()
          and len(summary['figures']) >= 3)
    print("[smoke]", "PASS" if ok else "FAIL")
    print(f"[smoke] cohort report at {cohort_md}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('parent', nargs='?',
                    help='parent directory containing campaign_* subdirectories')
    ap.add_argument('--out', default=None,
                    help='cohort output dir (default <parent>/cohort_report)')
    ap.add_argument('--glob', default='campaign_*',
                    help="glob for campaign dirs under parent (default 'campaign_*')")
    ap.add_argument('--max-iters-per-topo', type=int, default=0)
    ap.add_argument('--topo-sample', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    if args.smoke_test:
        return _smoke_test()
    if not args.parent:
        ap.error('parent path required (or use --smoke-test)')
    parent = Path(args.parent).resolve()
    if not parent.is_dir():
        ap.error(f'not a directory: {parent}')
    out_dir = Path(args.out) if args.out else parent / 'cohort_report'
    summary = analyze_cohort(parent, out_dir, glob_pat=args.glob,
                             max_iters_per_topo=args.max_iters_per_topo,
                             topo_sample=args.topo_sample, jobs=args.jobs)
    c = summary['cohort']
    print(f"\n[multi_campaign_report] {c['n_campaigns']} campaigns / "
          f"{c['total_iters']} sims total")
    print(f"  cohort-median useful yield: {100*c['cohort_median']['useful_frac']:.0f}%")
    gc = c['grouped_correlation']
    print(f"  within-campaign median rho: "
          f"sigma={gc['sigma']['within_campaign_rho_median']:+.2f} "
          f"I_inj={gc['I_inj']['within_campaign_rho_median']:+.2f}")
    if summary['n_failed']:
        print(f"  FAILED campaigns: {summary['n_failed']}")
    print(f"  cohort report : {out_dir/'multi_campaign_report.md'}")
    print(f"  cohort summary: {out_dir/'multi_campaign_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
