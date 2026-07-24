# Virtual MEA pipeline

Synthesises 9-channel virtual multi-electrode-array (MEA) recordings from
Astro-Neuron-Network HPC campaign output (neuron positions + spike times),
detects spikes on those recordings, and saves detected events alongside the
run's parameter vector. Optionally writes diagnostic plots so a result can be
sanity-checked without re-deriving it from the numbers.

For the physics/math behind every design choice below, see
[`THEORY.md`](./THEORY.md). This document is the interface reference: what
each file does, what it expects, what it produces, and how to run it.

Files affected / added:
- `eap_template_library.py` — biophysical EAP waveform generator (new)
- `mea_probe.py` — probe geometry + distance-scaling weights (new)
- `mea_synthesis.py` — trace synthesis + noise calibration (new)
- `mea_detection.py` — spike detection + ground-truth matching (new)
- `mea_plots.py` — diagnostic plots (new, optional: needs matplotlib)
- `process_campaign.py` — campaign walker / HPC entry point (new)
- `submit_mea.sh` — PBS submission wrapper, single campaign (new)
- `build_mea_manifest.py`, `submit_mea_array.sh`, `launch_mea_array.sh` —
  PBS array submission across many campaigns as ONE combined job (new)
- `launch_mea_per_campaign.sh` — submit ONE SEPARATE PBS job per campaign
  (thin wrapper around `launch_mea_array.sh`; new)
- `smoke_test_eap_library.py`, `smoke_test_mea_pipeline.py`,
  `smoke_test_mea_plots.py` — correctness tests (new)

---

## Contents

1. Overview
2. Dependencies
3. Input contract
4. Output contract
5. Configuration reference
6. Module API reference
7. Usage
8. Diagnostic plots
9. HPC submission
10. Known limitations
11. References

---

## 1. Overview

```
campaign/topo_k/topology.npz  ─┐
campaign/topo_k/*.json        ─┼──► mea_probe.py ──► W(n,e) weights (once per topo)
                                │
eap_library.npz  ──────────────┼──► mea_synthesis.py ──► 9-channel raw traces
                                │         (per iter, uses W + spike times)
campaign/topo_k/iter_n.npz  ───┘
                                          │
                                          ▼
                                  mea_detection.py ──► detected events
                                          │             + ground-truth match
                                          ▼
                          out/topo_k/mea_iter_n.npz  (final output)
                                          │
                                          ▼ (optional, --plots)
                          out/topo_k/plots/...        (diagnostic PNGs)
```

`process_campaign.py` is the only entry point you need to run; it drives the
other modules and handles the once-per-topology / per-iteration split.

---

## 2. Dependencies

Pure `numpy` + `scipy`. No Brian2, no compiler, no GPU. `matplotlib` is
**optional**, used only for diagnostic plots (`--plots`, see §8) and for
`eap_template_library.py --plot`. If it is absent the pipeline prints one
warning and continues, producing identical numerical output.

```bash
pip install numpy scipy --break-system-packages   # if not already present
pip install matplotlib --break-system-packages    # optional, for --plots
```

---

## 3. Input contract

`process_campaign.py --campaign <DIR>` expects `<DIR>` to directly contain
one or more `topo_*/` subdirectories (i.e. point it at a `sweep_*_task*`
directory, not the `campaign_<TAG>` root — submit one job per sweep directory,
or loop over them).

Per `topo_*/`:

| file | required keys | notes |
|---|---|---|
| `topology.npz` | `N_pos` : `(Nn, 2)` float, µm | neuron positions |
| `topology_meta.json` (or `job_args.json`) | `c_max` | arena side length, µm; falls back to inferring from `N_pos` extent if absent |
| `iter_*.npz` (one or more) | `spk_N_t` : `(K,)` float32, seconds<br>`spk_N_i` : `(K,)` int32<br>`params` : `(36,)` float64<br>`theta`, `conn_prob`, `topo_idx`, `seed_run` | passed through to output unchanged |

---

## 4. Output contract

One file per input iteration: `<out>/topo_<k>/mea_iter_<n>.npz`

| key | shape / dtype | meaning |
|---|---|---|
| `det_ch` | `(M,)` int32 | detected-event electrode channel, `0..8` |
| `det_t` | `(M,)` float32 | detected event (trough) time, seconds |
| `det_amp` | `(M,)` float32 | detected event amplitude, µV (signed) |
| `det_src_neuron` | `(M,)` int32 | ground-truth-matched neuron id, or `-1` (unmatched / false positive) |
| `det_dt_to_truth` | `(M,)` float32 | `\|t_det - t_true\|`, seconds, or `nan` if unmatched |
| `sigma` | `(9,)` float32 | per-channel robust noise estimate, µV |
| `electrode_centers` | `(9,2)` float32 | probe coordinates, µm |
| `params` | `(36,)` float64 | pass-through from the input iter file |
| `theta`, `conn_prob`, `topo_idx`, `iter_idx`, `seed_run` | — | pass-through / provenance |
| `fs`, `simtime` | float | recording sample rate (Hz), simulated duration (s) |
| `meta_json` | str | full pipeline configuration used for this run (reproducibility) |
| `traces` | `(9, T)` float32 | **only present** if `--save_traces_first` is set, and only for iteration 0 of each topology |

Raw traces are not saved by default — at campaign scale this is the dominant
storage cost. Use `--save_traces_first` to keep a small validation subset (one
trace per topology). Diagnostic PNGs, if enabled, are written alongside the
`.npz` files under `<out>/topo_<k>/plots/` (see §8) and are not part of this
`.npz` contract.

---

## 5. Configuration reference

All parameters are exposed as `process_campaign.py` CLI flags. Defaults shown.

### Template library

| flag | default | meaning |
|---|---|---|
| `--library` | `eap_library.npz` | path to a pre-built library (`eap_template_library.py`); generated on the fly with defaults if missing |
| `--n_templates` | `30` | library size (only used if generating on the fly) |

### Probe geometry

| flag | default | meaning |
|---|---|---|
| `--n_side` | `3` | electrodes per side of the grid (`3` = 3x3 = 9 electrodes, matching the specified probe; not validated beyond `3`) |
| `--pitch` | `60.0` | electrode centre-to-centre pitch, µm |
| `--edge` | `25.0` | electrode edge length, µm |
| `--n_sub` | `4` | sub-grid resolution per electrode (`n_sub x n_sub` sub-sites) |

### Scaling law

| flag | default | meaning |
|---|---|---|
| `--n_dec` | `2.0` | decay exponent; `2` = dipole-like `1/r^2` (default), `1` = inverse-distance `1/r` |
| `--r_ref` | `30.0` | reference distance, µm |
| `--r_min` | `10.0` | near-field floor, µm |
| `--snr_ref` | `15.0` | target SNR at `r_ref`; sets `A_ref = snr_ref * target_noise_uv` |
| `--gamma` | `0.5` | reach-cutoff fraction of noise; sets `r_max` |
| `--target_noise_uv` | `5.0` | target post-filter robust noise level, µV |

### Recording / synthesis

| flag | default | meaning |
|---|---|---|
| `--fs` | `10110.09` | recording sample rate, Hz |

### Detection

| flag | default | meaning |
|---|---|---|
| `--band_lo` | `300.0` | band-pass low edge, Hz |
| `--band_hi` | `3000.0` | band-pass high edge, Hz |
| `--k` | `5.0` | threshold multiplier (x robust sigma) |
| `--refractory_ms` | `2.0` | lockout/dead-time window; also merges band-pass ringing into one event per spike |
| `--polarity` | `neg` | `neg` \| `pos` \| `both` |

### Diagnostics

| flag | default | meaning |
|---|---|---|
| `--plots` | off | write diagnostic PNGs (see §8) |
| `--plot_window_s` | `2.0` | width of the plotted time window, s (auto-centred on densest activity) |
| `--plot_max_true_rows` | `120` | max neurons shown in the ground-truth raster |

### Execution

| flag | default | meaning |
|---|---|---|
| `--workers` | `1` | worker processes; parallelism is **one topology per worker** (see below) — set this to the core count of your allocation, e.g. `--workers 48` for `-l select=1:ncpus=48` |
| `--save_traces_first` | off | keep raw `(9,T)` traces for iteration 0 of each topology |
| `--limit_topos` | none | process only the first N `topo_*` directories (debugging / dry runs) |
| `--limit_iters` | none | process only the first N iterations of each topology (debugging / dry runs) |
| `--self_test` | — | run the built-in synthetic end-to-end check and exit |

**Parallelization, precisely.** `walk_campaign` dispatches one
`multiprocessing` worker **per topology** (`spawn` context), not per
iteration — a topology's full geometry/weight setup and every one of its
iterations run inside a single worker, sequentially. This is the right
granularity for real campaigns: `HPC_main_sweep.py` defaults to
`--n_topologies 10000` per sweep, so a real sweep directory has far more
topologies than any node has cores, and each topology is fully independent
work (no shared state needed once its weight matrix is built), which is the
condition process-level parallelism wants. Two things follow directly from
this:

- `--workers` above the number of `topo_*` directories found leaves the extra
  workers idle for that run; the pipeline detects this and prints a warning
  rather than doing it silently.
- Every worker pins its own BLAS/OpenMP backend to a single thread
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` are all set to
  `1` via `os.environ.setdefault` as the first thing `process_campaign.py`
  does, before numpy is imported anywhere). This matters because numpy here
  is commonly linked against OpenBLAS, whose default thread count is the
  number of *visible* CPUs — without this, `N` worker processes could each
  also try to spawn up to that many BLAS threads, oversubscribing the node
  and making more workers *slower*, not faster. If you have deliberately set
  these variables yourself (e.g. to run `--workers` below the core count and
  leave a couple of BLAS threads per worker on purpose), `setdefault` means
  your value is respected, not overwritten. The active values are printed at
  the start of every run so this is verifiable from the job log rather than
  something you have to trust:

  ```
  [mea] 4000 topo(s) found, dispatching over 48 worker(s) (1 topo/worker; BLAS pinned: OMP=1 OPENBLAS=1 MKL=1)
  ```

---

## 6. Module API reference

Every signature below is copied from source, not paraphrased; if a signature
in your checked-out copy differs, the code is the source of truth and this
section is stale.

### `eap_template_library.py`

```python
generate_template_library(n_templates=30, seed=0,
                          sim_cfg: SimConfig = None,
                          tpl_cfg: TemplateConfig = None) -> dict
    # -> {templates: (L, n_hr) float64, dt_hr_ms: float, trough_idx: (L,) int64,
    #     params: (L, 5) float64, param_names: [...], meta: {...}}

resample_template_to_fs(w, dt_hr_ms, fs_hz) -> (w_resampled, trough_idx_resampled)

save_library(lib, path)          # -> path
load_library(path)               # -> dict, same shape as generate_template_library's output
```

`SimConfig` (HH integration: `t_total_ms`, `t_on_ms`, `I_app`, `dt_hr_ms`,
`rtol`, `atol`) and `TemplateConfig` (`surrogate`:
`'capacitive'|'ionic_sum'|'neg_d2V'`, `pre_ms`, `post_ms`, `n_avg_spikes`,
`v_thresh_mv`) are dataclasses with sensible defaults; override by
constructing and passing explicitly. `HHParams` holds one drawn parameter set
(`gNa, gKd, VT, gM, taumax, gL, EL, Cm, ENa, EK`).

CLI: `python eap_template_library.py --out eap_library.npz --n_templates 30
[--seed 0] [--surrogate capacitive] [--I_app 7.0] [--plot]`.

### `mea_probe.py`

```python
ProbeConfig(n_side=3, pitch=60.0, edge=25.0, n_sub=4)
ScalingConfig(A_ref=75.0, r_ref=30.0, r_min=10.0, n_dec=2.0)

make_probe(c_max, cfg: ProbeConfig = None) -> dict
    # -> {centers: (E,2), sub_sites: (E, n_sub^2, 2), cfg, center_xy: (2,)}

compute_weights(neuron_xy, probe, scfg: ScalingConfig = None) -> W   # (Nn, E) float, uV
r_max_reach(scfg: ScalingConfig, gamma, sigma_noise) -> float        # um
reach_mask(W, floor) -> (Nn,) bool
```

### `mea_synthesis.py`

```python
assign_templates(Nn, n_templates, seed=0) -> (Nn,) int64

prepare_library_at_fs(lib, fs_hz) -> dict          # {wf: [(T_tpl,) array, ...], trough_idx: (L,) int64}

calibrate_noise_rms(fs_hz, band, target_post_uv, order=4, seed=12345,
                    n_probe=None) -> float           # PRE-filter white-noise RMS [uV]

synthesize_traces(spk_t, spk_i, W, lib_fs, tmpl_assign, fs_hz, simtime_s,
                  noise_rms, reach=None, seed=0, return_clean=False)
    # -> (traces, t)                    if return_clean is False
    # -> (traces, t, clean)             if return_clean is True
    # traces, clean : (E, T) float, uV  |  t : (T,) float, seconds
    # NB: returns a TUPLE, not a bare array -- unpack it (see process_iter below)

bandpass(x, fs_hz, lo_hz, hi_hz, order=4) -> filtered array   # sosfiltfilt, zero-phase; same shape as x
```

### `mea_detection.py`

```python
DetectConfig(lo_hz=300.0, hi_hz=3000.0, order=4, k=5.0,
            refractory_ms=2.0, polarity='neg')

robust_sigma(x) -> float                             # median(|x|) / 0.6745

detect(traces, fs_hz, cfg: DetectConfig = None, prefiltered=False) -> dict
    # -> {ch: (M,) int, t: (M,) float [s], amp: (M,) float [uV],
    #     sigma: (E,) float [uV], filtered: (E,T) float [uV]}

match_to_truth(det, W, spk_t, spk_i, fs_hz, window_ms=1.5) -> (src_neuron, dt_to_truth)
    # src_neuron : (M,) int, ground-truth-matched neuron id or -1
    # dt_to_truth: (M,) float, seconds, or nan if unmatched
```

### `mea_plots.py`

Every function takes plain arrays only (no npz, no config objects) and
returns the path it wrote. `HAVE_MPL` is `False` when matplotlib is absent; a
call into any plotting function raises `RuntimeError` in that case, which is
exactly why `process_campaign.py` checks `HAVE_MPL` before calling in.

```python
# per-topology geometry
plot_probe_layout(neuron_xy, probe, W=None, r_max=None, c_max=None, out_dir='.') -> path
plot_weight_decay(neuron_xy, probe, W, scaling, r_max=None, noise_floor=None, out_dir='.') -> path
plot_template_library(templates, dt_hr_ms, out_dir='.') -> path

# per-iteration signal
plot_stacked_traces(traces, fs_hz, t_window=None, sigma=None, out_dir='.') -> path
plot_traces_with_detections(filtered, fs_hz, det_t, det_ch, det_amp, sigma, k,
                            t_window=None, src_neuron=None, out_dir='.') -> path
plot_raster(det_t, det_ch, n_el, simtime_s, src_neuron=None,
            spk_t=None, spk_i=None, visible_neurons=None,
            t_window=None, out_dir='.') -> path
plot_zoom(filtered, fs_hz, det_t, det_ch, det_amp, sigma, k,
          channel=None, width_ms=60.0, out_dir='.') -> path
plot_detected_waveforms(filtered, fs_hz, det_t, det_ch, n_el,
                        pre_ms=1.0, post_ms=2.0, max_per_ch=300, out_dir='.') -> path
plot_amplitude_histogram(det_amp, det_ch, sigma, k, n_el, out_dir='.') -> path
plot_detection_summary(det_ch, src_neuron, sigma, n_el, dt_to_truth=None, out_dir='.') -> path

# orchestrators (what process_campaign.py actually calls)
save_topo_diagnostics(out_dir, neuron_xy, probe, W, scaling, r_max=None,
                      noise_floor=None, templates=None, dt_hr_ms=None) -> [paths]   # 3 files
save_iter_diagnostics(out_dir, traces, filtered, fs_hz, det, src_neuron,
                      dt_to_truth, k, simtime_s, spk_t=None, spk_i=None,
                      visible_neurons=None, window_s=2.0) -> [paths]                # 7 files

# helper
busiest_window(det_t, simtime_s, width_s=2.0) -> (t0, t1)
```

### `process_campaign.py`

```python
PipelineConfig(**kwargs)
    # every CLI flag in section 5 as a keyword, with the same defaults;
    # .scaling(), .probe_cfg(), .detect_cfg() build the sub-configs above;
    # .to_dict() round-trips to/from JSON (this is what meta_json stores)

read_c_max(topo_dir, campaign_dir, N_pos) -> float
    # topology_meta.json, else job_args.json, else inferred from N_pos extent

process_iter(iter_npz, probe, W, reach, lib_fs, tmpl_assign, cfg,
            noise_rms, out_path, topo_idx, save_traces=False,
            plot_dir=None) -> int
    # synthesizes + detects + matches ONE iteration, writes out_path,
    # optionally writes signal plots into plot_dir; returns the number
    # of detected events

process_topo(topo_dir, campaign_dir, out_topo_dir, lib, cfg,
            save_traces_first=False, limit=None) -> (topo_idx, n_iters_found, n_iters_done)
    # builds geometry/weights/template-assignment ONCE, then loops
    # process_iter over every iter_*.npz in topo_dir; writes topology-level
    # plots once if cfg.plots is set; failures are logged to
    # <out_topo_dir>/_failures.log rather than aborting the topology

walk_campaign(campaign_dir, out_dir, lib_path, cfg, workers=1,
             save_traces_first=False, limit_iters=None, limit_topos=None) -> list
    # discovers every topo_* under campaign_dir and dispatches process_topo,
    # sequentially if workers<=1 else over a spawn-context multiprocessing
    # Pool (one worker process per topology); prints a one-line summary,
    # a startup line confirming the active BLAS thread-pinning (see section 5
    # "Parallelization, precisely"), and a warning if workers exceeds the
    # number of topologies found; returns the list of per-topology
    # (topo_idx, n_found, n_done) results

resolve_library(path) -> (lib, path)
    # loads path if it exists, else generates and saves a default library there

self_test(tmpdir=None) -> None
    # builds a small synthetic campaign, runs walk_campaign on it (with
    # plots enabled), and asserts 7 correctness checks; raises on failure
```

---

## 7. Usage

### Generate / preview a template library

```bash
python eap_template_library.py --out eap_library.npz --n_templates 30 --plot
```

### Run the smoke tests (do this first, especially after any edit)

```bash
python smoke_test_eap_library.py        # 10 checks, HH model + template library
python smoke_test_mea_pipeline.py       # 10 checks, probe + synthesis + detection
python smoke_test_mea_plots.py          # 9 checks, all diagnostic plots
python process_campaign.py --self_test  # 7 checks, full pipeline on a synthetic topology
```

All four must print `PASSED` before trusting output from a real campaign.

### Process a real campaign directory

```bash
python process_campaign.py \
    --campaign /path/to/campaign_TAG/sweep_c48_task0000 \
    --out      /path/to/mea_out/sweep_c48_task0000 \
    --library  eap_library.npz \
    --workers  32
```

Common overrides:

```bash
python process_campaign.py --campaign <DIR> --out <DIR> \
    --n_dec 1                 # inverse-distance instead of dipole-like
    --snr_ref 20 --k 4.5      # e.g. to mirror a specific real-rig convention
    --refractory_ms 1.0       # shorter lockout (more collisions resolved as distinct)
    --save_traces_first       # keep a raw-trace validation subset
    --plots                   # write diagnostic PNGs (see section 8)
```

A quick dry run on a small slice of a real campaign, before committing to a
full submission:

```bash
python process_campaign.py --campaign <DIR> --out <DIR>_dryrun \
    --limit_topos 1 --limit_iters 3 --plots
```

---

## 8. Diagnostic plots

Pass `--plots` to write PNG diagnostics. They are **off by default** (a full
campaign would otherwise emit hundreds of thousands of images). When on:
geometry plots are drawn **once per topology**, signal plots **only for the
first iteration** of each topology.

```bash
python process_campaign.py --campaign <DIR> --out <DIR> --plots
```

Layout:

```
<out>/topo_<k>/plots/
    01_probe_layout.png          geometry: culture, neurons, electrodes, r_max
    02_weight_decay.png          W vs distance + analytic scaling law
    03_template_library.png      the EAP template bank
    iter_<n>/
        04_stacked_traces_raw.png    9 raw channels, stacked
        05_traces_detections.png     band-passed + thresholds + detections
        06_raster.png                detected raster + ground-truth raster
        07_zoom.png                  single channel, waveform resolution
        08_detected_waveforms.png    snippet overlay per channel + mean
        09_amplitude_hist.png        amplitude distribution vs threshold
        10_detection_summary.png     TP/FP counts, sigma, timing error
```

What to look at first, and what a problem looks like (full rationale for each
plot, i.e. exactly what claim it is evidence for, is in `THEORY.md` §8b):

| plot | reading it |
|---|---|
| `02_weight_decay` | Scatter must lie **on** the analytic curve (slight upward lift below ~30 µm from sub-site averaging is expected). Off-curve = wrong exponent, floor, or units. |
| `05_traces_detections` | Green = matched to a true spike, red = unmatched (false positive). A field of red means the threshold is too low. |
| `07_zoom` | You should see the biphasic EAP shape and exactly **one** marker per spike. Several markers on one waveform = lockout too short. |
| `08_detected_waveforms` | A clean biphasic mean out of the cloud = real spikes. A shapeless hairball = detecting noise. |
| `09_amplitude_hist` | Amplitudes piling up **at** the threshold line = scraping the noise floor; raise `--k`. |

Relevant flags: `--plot_window_s` (width of the plotted time window, default
`2.0` s — the window is auto-centred on the densest activity, not on t=0) and
`--plot_max_true_rows` (cap on neurons shown in the ground-truth raster,
default `120`).

matplotlib is an **optional** dependency: if it is missing, the pipeline
prints one warning and continues, producing all numerical output as normal. A
failure inside plotting itself is also caught and logged as a warning — it
happens strictly *after* the iteration's `.npz` has already been written, so a
plotting bug can never cost a completed computation.

---

## 9. HPC submission

Two ways to run, depending on scope:

- **One campaign/sweep directory** -> `submit_mea.sh`, a single PBS job (9.1).
- **Many campaigns/sweeps, non-interactively, in one command** -> `build_mea_manifest.py` + `submit_mea_array.sh` + `launch_mea_array.sh`, a PBS **array** job (9.2).

### 9.1 Single campaign: `submit_mea.sh`

Post-processing only (numpy + scipy, no Brian2, no compilation), so it needs
far less walltime than the simulation sweep itself.

```bash
qsub -l select=1:ncpus=48 \
     -v CAMPAIGN=/scratch/USER/campaign_TAG/sweep_c48_task0000,\
OUT=/scratch/USER/mea_out/sweep_c48_task0000,\
LIB=/scratch/USER/eap_library.npz \
     submit_mea.sh
```

Edit the environment-activation lines at the top of `submit_mea.sh` to match
your cluster's module/conda setup (same environment used for the simulation
sweep is sufficient — no extra packages are required unless you also want
`--plots`, in which case add matplotlib to that environment).

### 9.2 Many campaigns, non-interactively: the array launcher

For running the pipeline across many `sweep_*_task*` directories — one or
more campaigns — **without submitting each one by hand and without an
interactive session**. This follows the same worker/launcher split as this
repo's own `submit_sweep_mixed.sh` / `launch_campaign.sh`, and uses the same
PBS Pro job-array mechanism (`qsub -J`, `$PBS_ARRAY_INDEX`) that
`launch_campaign.sh` already uses for the simulation sweep itself.

**Three files, three roles:**

| file | role |
|---|---|
| `build_mea_manifest.py` | discovers every `sweep_*_task*` directory (across one or more campaigns) that actually contains `topo_*`, and writes a numbered manifest |
| `submit_mea_array.sh` | the array **worker** — reads exactly one manifest line, selected by `$PBS_ARRAY_INDEX`, and runs `process_campaign.py` on just that directory |
| `launch_mea_array.sh` | the **launcher** — builds the manifest, then submits one `qsub -J` array job (or a plain job if only one work unit is found, since PBS Pro rejects a single-element array — same convention as `launch_campaign.sh`) |

You normally only ever run `launch_mea_array.sh`; the other two are called by
it. One command, non-interactive, covers everything under one or more
campaign roots:

```bash
./launch_mea_array.sh \
    --out-root /path/to/mea_out \
    --glob '/davinci-1/home/USER/ANN/Phenomenological/Main/campaign_*'
```

(Quote the `--glob` pattern so your shell doesn't expand it before the script
sees it — this lets the script pick up campaigns created after you type the
command, and matches every `campaign_*` directory, not just the ones that
exist right now.) Or name specific campaigns explicitly (repeatable):

```bash
./launch_mea_array.sh \
    --out-root /path/to/mea_out \
    --campaign-root /path/campaign_cadex_rho1300v3 \
    --campaign-root /path/campaign_cadex_rho2000v1
```

**Options** (`./launch_mea_array.sh --help` for the full list):

| flag | default | meaning |
|---|---|---|
| `--out-root` | required | output root; layout is `<out-root>/<campaign_dir_name>/<sweep_task_dir_name>` |
| `--campaign-root` | — | a campaign directory to search (repeatable) |
| `--glob` | — | a shell glob matching multiple campaign directories (repeatable; quote it) |
| `--lib` | `./eap_library.npz` | built **once, synchronously, before submitting** — so array members never race to generate it |
| `--queue` | `cpu` | PBS queue |
| `--ncpus` | `48` | cores per array member -> forwarded as `--workers` to `process_campaign.py` |
| `--walltime` | `06:00:00` | per array member |
| `--concurrency` | `20` | the `%N` throttle in `-J "0-M%N"`; keep within your fairshare/queue limits |
| `--skip-done` | off | omit work units whose output already has a complete `mea_manifest.json` — use this to top up a campaign after new sweeps finish, without reprocessing what's already done |
| `--extra-args "..."` | — | forwarded verbatim to every array member's `process_campaign.py` call, e.g. `--extra-args "--plots"` |
| `--dry-run` | off | build the manifest and print the `qsub` command without submitting — use this to sanity-check the plan first |

**Always `--dry-run` first** on a new campaign layout — it costs nothing and
shows you exactly how many work units were found and the array range that
will be submitted:

```bash
./launch_mea_array.sh --out-root /path/to/mea_out \
    --glob '/path/to/Main/campaign_*' --dry-run
```

Monitor with the job ID `launch_mea_array.sh` prints:

```bash
qstat -t <JID>          # per-array-member status
qstat -u $USER           # everything you have queued/running
```

Each array member's stdout/stderr (via `#PBS -k eo`) will show which
`campaign`/`out` pair it was assigned and its worker count — useful for
tracing a specific failed array index back to a specific sweep directory.

### 9.3 One separate job per campaign: `launch_mea_per_campaign.sh`

`launch_mea_array.sh` (9.2) flattens every campaign's work units into a
**single combined** array job. If instead you want **one independent PBS job
per campaign** — separate job IDs, so you can monitor, cancel, or re-run one
campaign without touching the others — use `launch_mea_per_campaign.sh`. It
is a thin loop around `launch_mea_array.sh`: for each campaign it calls that
script once, with `--campaign-root` pointing at just that one campaign, so
every array-vs-plain-job decision, BLAS pinning, and manifest-building step
is the same already-tested logic from 9.2, just applied per campaign instead
of once overall.

```bash
# auto-discover every campaign_* directory directly under --parent,
# and submit one job for each:
./launch_mea_per_campaign.sh \
    --parent /davinci-1/home/USER/ANN/Phenomenological/Main \
    --out-root /path/to/mea_out

# or name specific campaigns explicitly (repeatable), e.g. a subset:
./launch_mea_per_campaign.sh \
    --campaign-root /path/campaign_cadex_rho1300v1 \
    --campaign-root /path/campaign_cadex_rho1300v2 \
    --out-root /path/to/mea_out
```

Every other option (`--lib`, `--queue`, `--ncpus`, `--walltime`,
`--concurrency`, `--skip-done`, `--extra-args`, `--dry-run`) matches
`launch_mea_array.sh` exactly and is forwarded unchanged to every
per-campaign call — `--concurrency`, for instance, becomes a **per-campaign**
throttle rather than one shared across everything. `--dry-run` prints the
planned `qsub` command for every campaign without submitting anything, and a
summary table of campaign -> job ID (or "not submitted", under `--dry-run`)
is printed at the end.

---

## 10. Known limitations

(Full derivations and physical justification in `THEORY.md` §9.)

- **Lockout collisions.** Two distinct spikes on the same electrode within
  `--refractory_ms` are reported as a single event. This is an inherent limit
  of single-channel threshold detection, not a bug; tune `--refractory_ms` per
  the expected burst statistics of your cultures.
- **Template homogeneity.** RS-exc cortical action potentials are genuinely
  similar in shape, so the default library's cross-template correlation is
  high (~0.9999). Widen the parameter draws in `eap_template_library.py` (or
  add explicit width/amplitude jitter) for a harder waveform-diversity
  benchmark.
- **No absolute physical calibration.** Amplitudes are SNR-relative, not
  µV-from-first-principles (`THEORY.md` §4.3). This does not affect detection
  behaviour but means `det_amp` values should not be compared to real-rig µV
  readings without an explicit external calibration step.
- **Independent per-parameter template sampling.** The HH parameter library
  draws each of the five fitted parameters independently rather than
  preserving their empirical cell-to-cell correlation (`THEORY.md` §3.3).
- **Diagnostic plots are a display convenience.** The auto-selected plotting
  window is biased toward the busiest period of the recording by design; no
  statistic should be read off a figure instead of the saved `.npz` arrays
  (`THEORY.md` §8b).

---

## 11. References

- Pospischil M, Toledo-Rodriguez M, Monier C, Piwkowska Z, Bal T, Frégnac Y,
  Markram H, Destexhe A (2008). Minimal Hodgkin-Huxley type models for
  different classes of cortical and thalamic neurons. *Biol Cybern* 99:427-441.
  — HH model and RS-exc parameter distribution (project knowledge base, full
  text).
- Kleefstra-syndrome MEA study (project knowledge base, full text) — detection
  threshold convention (±4.5 SD, 100 Hz high-pass, 10 kHz sampling).
- Astrocyte-EV MEA study (project knowledge base, full text) — detection
  threshold convention (6x SD, 200-3000 Hz band-pass).
- Quiroga RQ, Nadasdy Z, Ben-Shaul Y (2004) — commonly cited origin of the
  median-absolute-deviation noise estimator used here. **Not verified against
  the paper's full text in this project** (no PubMed/bioRxiv tool was
  reachable from this environment); treat this attribution as a convention,
  not a confirmed citation, until checked.
