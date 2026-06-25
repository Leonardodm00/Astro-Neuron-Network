#!/usr/bin/env python3
"""
test_burst_metrics.py -- self-contained smoke test for burst_metrics.py.

Builds synthetic population spike trains whose burst structure is known by
construction, runs both detectors, and asserts the recovered scalar metrics
match the planted ground truth within tolerance. Also checks the two
detector-independent irregularity descriptors against closed-form cases and a
negative (asynchronous-Poisson) control.

HOW TO RUN
----------
    cd <dir containing burst_metrics.py>
    python test_burst_metrics.py            # prints PASS/FAIL per check, exits 1 on any FAIL

No campaign data, no plotting, no network access required: everything is
generated in-process from a fixed seed, so the test is deterministic and
reviewable line-by-line.
"""
import sys

import numpy as np

import burst_metrics as bm


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------
def make_bursting_population(n_neurons=60, t_rec=60.0, n_bursts=6,
                             first_onset=5.0, period=10.0, burst_dur=0.40,
                             spikes_per_neuron_in_burst=12, bg_rate_hz=0.05,
                             seed=0):
    """Population with n_bursts synchronous network bursts on a regular period.

    Ground truth:
      onsets        = first_onset + period * arange(n_bursts)
      duration      ~ burst_dur (each neuron fires spikes_per_neuron_in_burst
                      spikes on a near-regular grid across [onset, onset+burst_dur]
                      with small jitter, so every internal ISI stays well below
                      the 100 ms intra-burst cap and the single-cell run is not
                      fragmented)
      participation ~ 1.0 (all neurons fire in every burst)
      nibi gap      = period - burst_dur
      plus a sparse asynchronous Poisson background at bg_rate_hz per neuron.
    """
    rng = np.random.default_rng(seed)
    onsets = first_onset + period * np.arange(n_bursts)
    K = spikes_per_neuron_in_burst
    base_grid = (np.arange(K) + 0.5) * (burst_dur / K)     # centers, in [0, burst_dur]
    jit = 0.2 * (burst_dur / K)                            # +/- 20% of the spacing
    t_list, i_list = [], []
    for nid in range(n_neurons):
        for on in onsets:
            ts = on + base_grid + rng.uniform(-jit, jit, size=K)
            t_list.append(ts)
            i_list.append(np.full(ts.size, nid))
        n_bg = rng.poisson(bg_rate_hz * t_rec)
        if n_bg:
            ts = rng.uniform(0.0, t_rec, size=n_bg)
            t_list.append(ts)
            i_list.append(np.full(ts.size, nid))
    spk_t = np.concatenate(t_list).astype(np.float64)
    spk_i = np.concatenate(i_list).astype(np.int64)
    order = np.argsort(spk_t)
    truth = dict(n_bursts=n_bursts, period=period, burst_dur=burst_dur,
                 nibi_gap=period - burst_dur,
                 burst_rate_per_min=n_bursts / (t_rec / 60.0))
    return spk_t[order], spk_i[order], n_neurons, t_rec, truth


def make_async_poisson(n_neurons=60, t_rec=60.0, rate_hz=2.0, seed=1):
    """Asynchronous Poisson population -- should yield ~0 network bursts."""
    rng = np.random.default_rng(seed)
    t_list, i_list = [], []
    for nid in range(n_neurons):
        n = rng.poisson(rate_hz * t_rec)
        ts = rng.uniform(0.0, t_rec, size=n)
        t_list.append(ts)
        i_list.append(np.full(ts.size, nid))
    spk_t = np.concatenate(t_list).astype(np.float64)
    spk_i = np.concatenate(i_list).astype(np.int64)
    order = np.argsort(spk_t)
    return spk_t[order], spk_i[order], n_neurons, t_rec


# ---------------------------------------------------------------------------
# Assertion harness
# ---------------------------------------------------------------------------
_FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def close(a, b, rtol=0.15, atol=0.0):
    if a is None or b is None:
        return False
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_bursting():
    spk_t, spk_i, N, T, truth = make_bursting_population()
    cfg = bm.BurstConfig()
    row = bm.compute_all(spk_t, spk_i, N, T, cfg)

    for pref, label in (("pr_", "poprate"), ("li_", "logisi")):
        nb = row[pref + "n_bursts"]
        check(f"{label}: burst count == 6 (got {nb:.0f})", abs(nb - truth["n_bursts"]) <= 1)
        check(f"{label}: burst_rate_per_min ~ {truth['burst_rate_per_min']:.1f}",
              close(row[pref + "burst_rate_per_min"], truth["burst_rate_per_min"], rtol=0.25),
              f"got {row[pref + 'burst_rate_per_min']:.2f}")
        check(f"{label}: burst_dur_mean_s ~ {truth['burst_dur']:.2f}",
              close(row[pref + "burst_dur_mean_s"], truth["burst_dur"], rtol=0.5, atol=0.15),
              f"got {row[pref + 'burst_dur_mean_s']:.3f}")
        check(f"{label}: nibi_mean_s ~ {truth['nibi_gap']:.1f}",
              close(row[pref + "nibi_mean_s"], truth["nibi_gap"], rtol=0.25),
              f"got {row[pref + 'nibi_mean_s']:.2f}")
        check(f"{label}: most spikes inside bursts (frac_outside < 0.25)",
              row[pref + "frac_spikes_outside_burst"] < 0.25,
              f"got {row[pref + 'frac_spikes_outside_burst']:.3f}")

    check("logisi: participation_mean > 0.8 (all neurons fire per burst)",
          row["li_participation_mean"] > 0.8, f"got {row['li_participation_mean']:.3f}")
    check("cross-check Jaccard > 0.5 (detectors agree)",
          row["cross_check_jaccard"] > 0.5, f"got {row['cross_check_jaccard']:.3f}")
    check("frac_active == 1.0 (every neuron fires)",
          close(row["frac_active"], 1.0, rtol=0.0, atol=1e-9), f"got {row['frac_active']:.3f}")


def test_negative_control():
    spk_t, spk_i, N, T = make_async_poisson()
    cfg = bm.BurstConfig()
    row = bm.compute_all(spk_t, spk_i, N, T, cfg)
    check("async control: participation detector finds ~0 network bursts",
          row["li_n_bursts"] <= 1, f"got {row['li_n_bursts']:.0f}")
    # poprate may flag a few chance fluctuations; require it stays small
    check("async control: poprate detector burst rate is low (< 10/min)",
          row["pr_burst_rate_per_min"] < 10.0, f"got {row['pr_burst_rate_per_min']:.2f}")


def test_irregularity_closed_form():
    # Perfectly regular train: CV_ISI == 0, adaptation index A == 0.
    t = np.arange(0.0, 10.0, 0.1)          # constant 100 ms ISI
    i = np.zeros(t.size, dtype=np.int64)
    d = bm.single_cell_irregularity(t, i, n_neurons=1)
    check("regular train: cv_isi_mean ~ 0", close(d["cv_isi_mean"], 0.0, rtol=0.0, atol=1e-6),
          f"got {d['cv_isi_mean']:.3e}")
    check("regular train: adapt_index_mean ~ 0",
          close(d["adapt_index_mean"], 0.0, rtol=0.0, atol=1e-6), f"got {d['adapt_index_mean']:.3e}")

    # Decelerating train: ISIs strictly increase -> A > 0.
    isi = np.linspace(0.05, 0.5, 20)
    t2 = np.concatenate([[0.0], np.cumsum(isi)])
    i2 = np.zeros(t2.size, dtype=np.int64)
    d2 = bm.single_cell_irregularity(t2, i2, n_neurons=1)
    check("decelerating train: adapt_index_mean > 0",
          d2["adapt_index_mean"] > 0.0, f"got {d2['adapt_index_mean']:.3f}")


def test_literature_distance_ranking():
    # Build three runs: one near the Mossink anchor, two off it. The near one
    # must get the smallest distance.
    cfg = bm.BurstConfig()
    near = make_bursting_population(period=10.0, burst_dur=0.4, n_bursts=6)
    far_fast = make_bursting_population(period=2.0, burst_dur=0.4, n_bursts=28, t_rec=60.0)
    far_long = make_bursting_population(period=20.0, burst_dur=3.0, n_bursts=3, t_rec=60.0)

    rows = []
    for (st, si, n, t, _truth) in (near, far_fast, far_long):
        r = bm.scalar_row(bm.compute_all(st, si, n, t, cfg))
        r["mean_FR_Hz"] = 3.0          # hold FR at the anchor so bursts drive the ranking
        rows.append(r)
    d = bm.literature_distance(rows, cfg, scale="campaign")
    check("literature distance: anchor-like run ranks closest",
          int(np.argmin(d)) == 0, f"distances={np.round(d, 3).tolist()}")


def main():
    print("=" * 64)
    print("burst_metrics smoke test")
    print("=" * 64)
    test_bursting()
    print("-" * 64)
    test_negative_control()
    print("-" * 64)
    test_irregularity_closed_form()
    print("-" * 64)
    test_literature_distance_ranking()
    print("=" * 64)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} FAILED -> {_FAILS}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
