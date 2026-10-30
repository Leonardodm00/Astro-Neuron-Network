# SPEC -- Astro-Neuron-Network (hpc/): refractory hold, refractory audit, LindaColloDUP launch package

Single source of truth for what the covered code must do. The code follows
this file; when the two disagree, fix whichever is wrong and record why in the
decision log. The integration tester reads only the repository, so every
constraint the code must satisfy is stated here, including those that come
from the author's decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears; quantifiers ("for each neuron
i", "for all t in ...") are stated explicitly.

Drafted from the code on 2026-10-30. An entry says `Confirmed: inferred`
until the author has checked it against what was intended; a constraint that
comes from an explicit decision of the author says `Confirmed: yes` with the
date of the decision.

Last reviewed at commit: 4aa3fdb (first draft at 562a9fd; Block 5, the repository
invariant of Block 1 and section 5.1 added on 2026-10-30 on top of 5f21a3d;
Block 5's REFUSED rule for an unrecognised CAdEx line added on top of 4aa3fdb;
its GIT CLONE status and queued-job hold added on top of 4e12a40)

## 0. Coverage

- `hpc/refractory_audit/` (the audit and its `checks/`)
- `hpc/Phenomenological_finalv1/LindaColloDUP/` (launch package and its `checks/`)
- `hpc/range_check/` (20 s runs over a setup's prior boxes, Block 6)
- the refractory handling of every CAdEx neuron model in the repository
  (today `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py`): in
  `Neuronal_Network`, the dV/dt line and the threshold, reset and refractory
  arguments of the neuron `NeuronGroup` -- not the rest of those files

Everything else in the repository is not specified yet and is out of scope
for the tester.

## 1. Scientific goal

The sweep pipeline simulates cultured neuron-astrocyte networks (Brian2,
cpp_standalone; CAdEx neurons of Gorski et al. 2021, Neural Comput 33:41) to
build training sets for simulation-based inference. The covered code
(a) holds each neuron's membrane potential at the reset value during its
refractory period, as the cited model defines it; (b) measures, for campaigns
already run without that hold, how many runs lost neurons to the refractory
runaway; (c) launches the neurons-only LindaColloDUP campaign on davinci.
Consumers: the author (audit verdicts, campaign outputs) and the SBI training
(campaign outputs).

## 2. Mathematical formulation

For each neuron i = 0, ..., Nn - 1 (`Neuronal_Network`):

    dV_i/dt = noise_i + [ gL (EL - V_i) + gL DeltaT exp((V_i - VT)/DeltaT)
                          + g_A,i (EA - V_i) + I_inj b_i - I_syn,i ] / Cm
    dg_A,i/dt = ( gbar_A / (1 + exp((V_A - V_i)/Delta_A)) - g_A,i ) / tau_A
    noise_i = sigma sqrt(2 gL_ref / Cm) xi_i / sqrt(dt), xi_i ~ N(0, 1) drawn
              each step and held constant over the step

Spike of neuron i at time t_s,i: V_i(t_s,i) > V_D while neuron i is not
refractory. Reset at t_s,i: V_i <- V_R and g_A,i <- g_A,i + delta_gA.
Refractory hold, for each neuron i and each of its spike times t_s,i:

    V_i(t) = V_R   for all t in (t_s,i, t_s,i + t_ref],   t_ref = 3 ms

implemented as the Brian2 flag `(unless refractory)` on the dV/dt line;
g_A,i, the synaptic variables and the noise generator keep evolving
(Gorski et al. 2021, Sec. 2: V "remains at this value during a refractory
period"). Brian2 evaluates the threshold as `(V > VD) and not_refractory`;
without the hold V_i integrates during t_ref with no threshold able to stop
it, and a neuron pushed past the unstable fixed point of the current balance
diverges to inf/NaN and never spikes again.

Refractory-runaway screen (`refractory_scan.classify_run`), for one run of
simulated duration T [s] with spike times t_k [s] and neuron indices i_k:

    count_i        = number of finite spikes k with i_k = i (0 <= i_k < Nn)
    last_i         = max over those k of t_k (-inf if count_i = 0)
    spiked_i       = (count_i > 0)
    early_i        = spiked_i and last_i <= t_early        (evaluated only if T >= 5 t_early)
    early_strict_i = spiked_i and last_i <= t_early_strict (same condition)
    tau_late       = max(late_min_s, late_frac T)
    late_i         = spiked_i and t_early < last_i <= T - tau_late and
                     #{k spiking : last_k > last_i + late_gap_s}
                         >= max(1, ceil(late_active_frac (Nn - 1)))

A run is "affected" when it has at least one early neuron (the headline,
calibrated rule) and "fully silenced early" when every spiking neuron is early.

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| V_i | membrane potential of neuron i | Brian2 state, float64 | mV |
| g_A,i | adaptation conductance of neuron i | state, float64 | nS |
| V_R, V_D, VT, EL, EA, V_A | reset, detection limit (-40 mV), soft threshold, leak and adaptation reversal, adaptation half-activation | scalars per neuron | mV |
| DeltaT, Delta_A | spike-initiation and adaptation slopes | scalars | mV |
| gL, gL_ref, gbar_A, delta_gA | leak, fixed noise-reference leak, maximal subthreshold adaptation, spike-triggered increment | scalars | nS |
| Cm | membrane capacitance | scalar | pF |
| tau_A | adaptation time constant | scalar | ms |
| I_inj b_i | frozen per-neuron bias, b_i ~ U(-0.5, 0.5) drawn once at build | scalar per neuron | pA |
| I_syn,i | synaptic current (inward negative) | state | pA |
| sigma | noise amplitude (stationary SD of the passive membrane) | scalar | mV |
| t_ref | refractory period | scalar, 3 ms | ms |
| t_s,i | a spike time of neuron i | float | s |
| T | simulated time of one run | float | s |
| Nn | number of neurons | int | -- |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| `spk_N_t` | `topo_*/iter_*.npz` of a sweep task | (K,) | float32 | s | from t = 0 of the run; non-finite times dropped by the screen |
| `spk_N_i` | same file | (K,) | int32 | -- | 0-based neuron index, aligned with `spk_N_t` |
| `job_args.json` | task folder `sweep_<queue>_task<NNNN>/` | dict | JSON | -- | `simtime` [s], `Nn`, `lib_dir` (folder of the `ASD_fun_BD_cpp.py` the task ran) |
| `iter_*.json` | next to each npz | dict | JSON | -- | `param_names` (column names of the parameter vector) |
| `<campaign>_runs.csv` | refractory_scan output | one row per run | CSV | s for times | columns: task, topo, iter, status, T, Nn, nn_source, n_spikes, n_spiking, n_early, n_early_strict, n_late, n_suspect, frac_suspect, net_silent_early, t_last_any, conn_prob, then par_<name> |
| `<campaign>_summary.json` | refractory_scan output | dict | JSON | percent | includes runs_early, pct_runs_early, ci95_early (Wilson 95 %), runs_any_suspect, runs_late, pct_runs_net_silent_early, pct_neurons_early |

## 4. Global conventions

- Units: Brian2 units in the model; seconds in the saved spike files; mV, nS,
  pF, pA, ms in documentation and checks.
- Randomness: the pipeline's seeds are unchanged by the covered code; the
  screen's sampling uses `numpy.random.default_rng(--seed)`, default seed 1.
- Binding decisions. The project's decision log is not available where this
  file was written, so the decisions are recorded in the project doc
  `claude/LindaColloDUP_Campaign_Operations.md`:
  - 2026-10-30 [author]: fix the refractory runaway: the refractory hold of
    section 2 in every repository copy of the CAdEx network and in
    LindaColloDUP (`REFRACTORY_CLAMP=yes`). Not applied in place to the
    cluster's `MEAMA/` (its campaigns run there). Superseded for `MEAMA/` by
    the next entry: MEAMA is rerun clamped, and Block 5 patches its folder
    once its arrays are stopped.
  - 2026-10-30 [author]: every project is rerun with the clamped model --
    Giulia's simulations, MEAMA, LindaColloDUP and the rho1300 family (the
    100-electrode project, confirmed by the author on 2026-10-30) -- so every
    project folder must hold a clamped `ASD_fun_BD_cpp.py` before its rerun
    (section 5.1, Block 5). Reruns use new campaign tags; data from the
    unclamped model is never pooled with clamped runs.
  - 2026-10-30 [author]: the campaigns run with the unclamped model are kept
    apart from the clamped reruns and deleted later; where they are is
    recorded in the project doc `claude/Unclamped_Campaigns_Inventory.md`.
    Clamped tags carry `clamp` in their stem, and any tool that reads several
    campaigns at once is given the clamped campaigns explicitly (for
    `multi_campaign_report.py`, `--glob` with the clamped stem; its default
    `campaign_*` takes every campaign in the folder).
  - 2026-10-30 [author]: the rho1300 rerun is Neuronal, as the eight
    `campaign_cadex_rho1300v*` (637 neurons, no astrocytes).
  - 2026-10-30 [author]: before relaunching, the activity produced over the
    prior boxes is measured with 20 s runs (Block 6) -- now for Giulia's
    setup and LindaColloDUP's, each as an independent 48-core job; the boxes
    of MEAMA and of the rho1300 family are retuned later (section 8).
  - 2026-10-29 [author]: LindaColloDUP runs MODE Neuronal; freezes Cm,
    I_inj, VR, VA, DeltaA, EC50_ampa, EC50_nmda (and the inert alpha_syn,
    O_G, Omega_G); SIMTIME 180 s; RHO_TOTAL 1300 cells/mm^2; everything else
    as MEAMA.

## 5. Blocks

### Block 1: refractory hold in the CAdEx neuron

- Module: `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py`, `Neuronal_Network`.
- Change: the dV/dt line ends with ` (unless refractory)`; four comment lines
  above it. Nothing else.
- Inputs / outputs: unchanged (same parameters in; spike trains out).
- Library calls relied on: Brian2 equation flag `unless refractory`;
  `NeuronGroup(threshold='V > VD', reset='V = VR; gA += delta_gA', refractory=t_ref)`.
- Custom code: none.
- Test oracles: (1) paired runs (`hpc/refractory_audit/checks/check_refractory_blowup.py
  --allow_clamped_source`, LindaColloDUP prior, seed 2026, 4 runs x 5 s):
  0 NaN neurons in every run (the same runs as shipped: 141, 159, 159, 116 of
  159); (2) spike counts equal to those of the same runs with the flag added
  by `check_refractory_blowup.py --clamp` (exact equality: same equations,
  same seeds); (3) `apply_refractory_clamp.py --check` reports the file as
  already clamped and the audit's CODE section reports it CLAMPED.
- Repository invariant: in every model file (a file named
  `ASD_fun_BD_cpp.py`, the module the workers import), the dV/dt line that
  contains the CAdEx term `exp((V-VT)/DeltaT)` ends with ` (unless refractory)`,
  and no Hodgkin-Huxley model file carries the flag (`hpc/ASD_fun_BD_cpp.py`
  must stay unflagged: holding V would break an HH neuron). Standalone checks
  that mirror the model use the clamped equation by default
  (`LindaColloDUP/checks/check_rs_reset_adaptation.py`; `--unclamped` gives
  the old model). Test fixtures that deliberately hold the unclamped line are
  exempt: `ASD_LINE` in `smoke_test_refractory_scan.py`, `_FIXTURE` in
  `apply_refractory_clamp.py`, `_CADEX` in `clamp_cadex_copies.py`. Oracle:
  `python3 hpc/refractory_audit/clamp_cadex_copies.py <repository root>` exits
  0 and lists exactly the two files above as CLAMPED and
  `hpc/ASD_fun_BD_cpp.py` as SKIP.
- Data flow: every campaign launched from a copy of these files.
- Confirmed: yes 2026-10-30 (decision in section 4).
- Status: smoke-tested 2026-10-30 (skipped: the cluster's 37-D copies, which
  are not in the repository; runs longer than 5 s).

### Block 2: refractory_scan, the spike-train screen

- Module: `hpc/refractory_audit/refractory_scan.py` (numpy only).
- Public API: `classify_run(spk_t, spk_i, Nn: int, T: float, cfg: dict | None = None) -> dict`;
  `scan_campaign(campaign_dir, out_dir, sample, seed, workers, cfg)`;
  CLI `python refractory_scan.py CAMPAIGN_DIR [...] --out DIR [--sample N] [--seed S] [--workers W]`.
- Inputs: spk_t (K,) float [s]; spk_i (K,) int, 0-based; Nn; T [s];
  cfg overrides of DEFAULT_CFG.
- Outputs: classify_run returns length-Nn arrays count (int), last (float,
  s, -inf if no spike), spiked, early, early_strict, late (bool), and the
  scalars tau_late (s) and early_evaluated (bool). scan_campaign writes the
  CSV and summary files of section 3 and a text summary.
- Parameters: DEFAULT_CFG t_early = 1.0 s, t_early_strict = 0.1 s,
  late_frac = 0.5, late_min_s = 20 s, late_active_frac = 0.10,
  late_gap_s = 1.0 s; `--sample` 1000 (0 = all runs), `--seed` 1, `--workers` 4.
- Custom code: the rules themselves; no library implements this screen.
- Test oracles: `smoke_test_refractory_scan.py` (41 checks: classify_run unit
  cases, a synthetic campaign, the driver end to end with a fake HOME, the
  bytes of the six package files); calibration against Brian2's NaN record
  (`checks/calibrate_refractory_scan.py`, 5 s runs): early-rule sensitivity
  0.964, 0.964 and 0.910 in three run sets, 0 false positives among 6,531
  healthy neurons. The late rule is logic-tested only.
- Data flow: reads sweep outputs (section 3); feeds Block 3.
- Confirmed: inferred (rules and thresholds proposed in the conversation of
  2026-10-30 and calibrated; the author has not confirmed the thresholds).
- Status: smoke-tested 2026-10-30; on davinci only the smoke test has run.

### Block 3: the audit driver and its PBS job

- Module: `hpc/refractory_audit/check_refractory_nan.sh` (login node) and
  `hpc/refractory_audit/submit_refractory_scan.sh` (PBS).
- CLI: `bash check_refractory_nan.sh [--sample N] [--workers W] [--out DIR] [CAMPAIGN_DIR ...]`;
  default campaigns, under `/davinci-1/home/ldellamea/ANN/Phenomenological/`:
  `MEAMA/campaign_meama_rho1600_n01` and
  `Main/Giulia_Astro/campaign_cadex_hhgap_v1`, `_v2`, `_v4`, `_v5`.
- Outputs: `refractory_audit_<UTC>/` next to the script, holding `audit.log`
  (sections 1 CODE, 2 LOGS, 3 DATA, VERDICT) and Block 2's files per
  campaign. Reads campaigns and job logs only; writes nothing else.
- `submit_refractory_scan.sh`: `qsub -q cpu`, 16 cores, 12 h; `CAMPAIGNS`
  colon-separated through `qsub -v`; Block 2 with `--sample 0`; output
  `refractory_audit_full_<UTC>/` in the submission folder.
- Test oracles: smoke test T3 (fake HOME; CODE clamped / not clamped /
  missing, LOGS table, DATA, VERDICT, unknown option refused).
- Confirmed: inferred.
- Status: smoke-tested 2026-10-30.

### Block 4: LindaColloDUP launch package

- Module: `hpc/Phenomenological_finalv1/LindaColloDUP/`: `launch_LCDUP.sh`,
  `submit_LCDUP.sh`, `apply_freeze_patch.py`, `smoke_test_freeze.py`,
  `apply_refractory_clamp.py`, `APPLY_LCDUP.txt` (procedure). Deployed on
  davinci in `Phenomenological/LindaColloDUP/`, with the pipeline copied
  from the cluster's 37-D `MEAMA/`, which is not in the repository.
- `launch_LCDUP.sh` settings (environment variable = default): MODE =
  Neuronal; SWEEP_GROUP = neuron_synapse; RHO_TOTAL = 1300; ASTRO_FRAC = 0;
  C_MAX = 350 um; SIMTIME = 180 s; FREEZE =
  Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G;
  CONN_RULE = flat; CONN_PERIODIC = 0; CONN_PROB_LO/HI = 0.1/0.6;
  STOA_GATE_LO/HI = 1.0/1.0; N_TOPOLOGIES_WORKER = 32; K_PARAMS = 1;
  WALLTIME = 24:00:00; TARGET = 1000000; HEADROOM_PCT = 200; CAMPAIGN_TAG =
  lcdup15_rho1300_neur_n01; CONDA_ENV = brian_final; DRYRUN = 0.
- Guards of `launch_LCDUP.sh`: exit 2 missing pipeline or package files;
  exit 3 SIZING_CONFIRMED not set, or REFRACTORY_CLAMP unset or not yes/no;
  exit 4 REFRACTORY_CLAMP disagreeing with the file, MODE, ASTRO_FRAC, gate,
  conn_prob bounds, FREEZE syntax, existing campaign tag; exit 5 seed
  allocator missing.
- `apply_freeze_patch.py`: adds `--freeze` to the folder's own
  `HPC_main_sweep.py` (3 insertions, marker `[LindaColloDUP patch 2026-10-29]`);
  `--check`, `--self-test`; refuses CR bytes, partial or foreign patches,
  missing or duplicate anchors.
- `apply_refractory_clamp.py`: Block 1's flag on the folder's own
  `ASD_fun_BD_cpp.py` (marker `[LindaColloDUP clamp 2026-10-29]`); `--check`,
  `--self-test`; exit 0 patched or already clamped, 1 refused.
- Test oracles: `smoke_test_freeze.py` (35 checks); the two patchers'
  `--self-test` (the clamp's includes a Brian2 parse of the patched
  equations); the dry run (Nn = 159, two qsub lines `-J 0-651%2`); the worker
  probe of `APPLY_LCDUP.txt` step 6 (NaN warnings 0 when clamped).
- Data flow: launches campaigns from the patched folder copies.
- Confirmed: yes 2026-10-29 for the settings the author decided (freeze
  list, SIMTIME, RHO_TOTAL, the rest as MEAMA) and 2026-10-30 for
  REFRACTORY_CLAMP = yes; inferred for VA = -50 mV and VR = -55 mV, and for
  I_inj = 4.0 pA (both open, section 8).
- Status: smoke-tested 2026-10-29 in the sandbox on 36-D copies (skipped:
  the 37-D files and davinci).

### Block 5: clamp_cadex_copies, the cluster-wide clamp

- Module: `hpc/refractory_audit/clamp_cadex_copies.py` (standard library only;
  Brian2 only in its self-test, optional).
- CLI: `python3 clamp_cadex_copies.py [ROOT ...] [--apply] [--force-running]
  [--recent-hours H] [--max-depth D] [--self-test]`; default ROOT
  `/davinci-1/home/ldellamea/ANN`, H = 6 h, D = 6.
- Finds: every file named `ASD_fun_BD_cpp.py` under the roots (the module
  name every worker imports from its `lib_dir`), descending at most D levels
  and skipping folders named `campaign_*`, `scratch*`, `mea_out*`, hidden
  folders and `__pycache__`.
- Classifies each file: SKIP (the CAdEx term `exp((V-VT)/DeltaT)`, spaces
  ignored, appears nowhere in the file), CLAMPED (one CAdEx dV/dt line ending
  in the flag), NOT CLAMPED (one CAdEx dV/dt line without flags), REFUSED (CR
  bytes, not UTF-8, more than one CAdEx dV/dt line, other flags on it, the
  marker `[refractory clamp]` present without the flag, or the term present
  but carried by no single-line `dV/dt = ... : volt`). An unrecognised CAdEx
  copy is therefore REFUSED, never SKIP.
- A NOT CLAMPED file tracked by a git clone (git `ls-files --error-unmatch`
  succeeds in its folder; without git, a `.git` entry in its folder or above)
  becomes GIT CLONE, in both modes, and is never edited: a local edit would
  block every later `git pull` there; the remedy is `git pull` in that clone.
- Queued jobs: `qstat -u $USER`, then `qstat -f <id>` per job (lines rejoined
  where qstat wraps them with newline + TAB); a job belongs to a folder when
  its `Variable_List` has `PBS_O_WORKDIR` or `REPO_DIR` equal to that folder
  (real paths). Report mode names such jobs in the file's row. Without
  qstat, or for a job whose record names neither variable, a WARNING line is
  printed and the verdict line points to it; the check then does not hold
  anything (fail open, loudly).
- With `--apply`, for each NOT CLAMPED file: HELD (left untouched) if a job
  of the user still listed by qstat belongs to its folder, or if a
  `campaign_*` folder next to it, or one of that folder's children, was
  modified less than H hours ago, and `--force-running` is not given;
  otherwise backup `<file>.bak-clamp-<YYYYmmdd_HHMMSS>`, insert one comment
  line holding the marker above the dV/dt line, append ` (unless refractory)`,
  write atomically keeping the permissions, re-read, re-classify (must be
  CLAMPED), syntax-check; on any failure restore the backup and report
  FAILED; else PATCHED. Nothing else in the file changes.
- Output: one row per file (status, age of the newest campaign output in its
  folder, path, the dV/dt line or the reason) and a verdict line.
- Exit status: 0 every CAdEx copy is clamped (after patching, with
  `--apply`); 3 report mode with at least one NOT CLAMPED copy; 1 any
  REFUSED, HELD, FAILED or GIT CLONE row, a missing root, or no CAdEx copy
  at all.
- Custom code: the scan and the patch rule; no library does this. Same patch
  rule as `hpc/Phenomenological_finalv1/LindaColloDUP/apply_refractory_clamp.py`,
  with the CAdEx-term condition added and a neutral marker.
- Test oracles: `--self-test` (28 checks: classification of each case, the
  patch adds only the comment line and the flag, HH untouched, HELD and
  `--force-running`, backups, idempotence, exit codes, a copy tracked by a
  git clone, qstat parsing with a wrapped `Variable_List` and a job without
  one, HELD by a queued job, no qstat, Brian2 parse of the patched
  equations); `smoke_test_clamp_cadex_copies.py` (18 checks: a mock of the
  cluster folders built from the repository's real model files, report exit
  3, apply exit 0, byte-exact diff, one backup per patched file, second apply
  a no-op; bytes; a stub `qstat` on PATH: a queued job holds its folder, none
  left lets the patch through).
- Confirmed: yes 2026-10-30 for the goal (every project folder runs the
  clamped model, section 4); inferred for the H = 6 h hold rule, the pruned
  folder names, the git-clone rule and the queued-job rule (the real
  `qstat -f` layout on davinci has not been read).
- Status: smoke-tested 2026-10-30 in the sandbox (skipped: davinci).

### Block 6: range_check, 20 s runs over a setup's prior boxes

- Purpose: before a campaign is relaunched with the refractory hold, measure
  the network activity its prior produces, through the setup's own sweep
  code: one-axis sweeps around the nominal vector at three wiring densities,
  and a sample of the joint prior; 20 s per run.
- Modules (`hpc/range_check/`): `range_check.py` (driver), `range_analysis.py`
  (per-run metrics and summaries), `range_plots.py` (figures only),
  `submit_range_check.sh` (PBS job, one node, 48 cores, one setup per job),
  `smoke_test_range_check.py`, `README_RANGE_CHECK.txt`.
- Driver: `python3 range_check.py --rc-phase {oat,prior} --rc-out DIR
  [--rc-like JOB_ARGS] [--rc-conn P] [--rc-points K] [--rc-topologies R]
  [--rc-k-prior k] [--rc-simtime T] [--rc-seed S] <HPC_main_sweep.py
  arguments>`. It imports `HPC_main_sweep` from the `--lib_dir` among the
  pass-through arguments (the setup's own folder: 36-D or 37-D, with or
  without the freeze patch) and calls its `main()` with the pass-through
  arguments followed by `--out_dir`, `--simtime T` (default 20 s),
  `--n_topologies`, `--n_params_per_worker`, `--seed_master`, and for `oat`
  also `--conn_prob_lo P --conn_prob_hi P`. Later arguments win.
  - `oat`: free axes = `resolve_sweep_group(--sweep_group)` minus the axes of
    `--freeze` (through the module's own `_apply_freeze` when present). For
    each free axis j with lo_j < hi_j, K values (default 7) of theta_j =
    linspace(lo_j, hi_j, K) in the sweep's theta coordinate
    (`PARAM_BOUNDS_THETA`: ln on `LOG_PARAMS` axes, natural units otherwise),
    every other axis at the nominal theta, `natural_to_theta(NOMINAL_PARAMS)`
    taken after the freeze. The list is padded with nominal vectors to
    n_workers x k, k = ceil(len / n_workers), n_workers from the module's
    `_resolve_n_workers`. In the parent process only, the module's
    `sample_theta` is replaced by a function that returns that list in
    order for every topology; this is the only change to the production
    path. R topologies (default 2) at conn_prob = P. `design.json` records
    the nominal theta, each point's axis and value, K, k, n_workers.
  - `prior`: unchanged production sampling: T topologies (default 8) x
    n_workers x k (default 1) draws with the setup's own conn_prob box.
  - `--rc-like`: rebuilds the base argument list from a campaign's
    `job_args.json` through the module's own parser (each key that is a
    parser destination; store_true options as flags when true; None values
    skipped; never out_dir, scratch_root, lib_dir, seed_master, n_workers,
    n_topologies, n_params_per_worker, simtime, rebuild_manifest_only);
    explicit arguments after it override.
  - Refuses (exit non-zero, nothing run) when the setup's model does not end
    its CAdEx dV/dt line with `(unless refractory)`, unless
    `--rc-allow-unclamped`; refuses an output folder that already holds runs.
- Output: `DIR/oat_cp<P>/` and `DIR/prior/` in the standard task layout
  (`job_args.json`, `topo_*/iter_*.npz` + `.json`, `manifest.json`), plus
  `design.json`.
- Analysis (`range_analysis.py DIR`): per run, the neuron spikes in the
  window [t_skip, T] (t_skip default 2 s, shifted to start at 0) go through
  `burst_metrics.compute_all` (the repository's network-burst panel) for
  the burst rate (log-ISI detector, `li_burst_rate_per_min`), CV of ISI and
  adaptation index; mean rate = spikes in the window / (Nn (T - t_skip));
  active fraction = neurons with rate > 0.1 Hz in the window; astrocyte
  event rate from the sidecar. Each `oat` run is labelled by its position:
  iter_idx < n_design is design point iter_idx, the rest is the nominal
  padding, and the stored theta must equal that vector exactly, otherwise
  the run is `unmatched` (position keeps a design point that coincides with
  the nominal vector apart from the nominal runs). Writes `runs.csv`;
  `oat_summary.csv` (per conn level, axis and value: number of runs, median,
  min and max of the mean rate, medians of the burst rate and of the active
  fraction); `prior_summary.txt` (quantiles of the mean rate; shares
  of runs that are silent, active fraction < 0.1; bursting, burst rate > 0;
  above 20 Hz; these cut-offs are reporting conventions, set in the config,
  not targets); `prior_spearman.csv` (Spearman rank correlation, with
  p-value, of each free theta axis and of conn_prob with log10(mean rate +
  0.01 Hz) and with the burst rate; `scipy.stats.spearmanr`).
- Plots (`range_plots.py DIR`): `oat_curves.png` (one panel per free axis:
  median mean rate against the axis value, one line per conn level, min-max
  band; log x on `LOG_PARAMS` axes), `prior_rates.png`, `prior_spearman.png`.
- Test oracles (`smoke_test_range_check.py`): the oat design (|free| x K
  points, each differing from nominal on exactly one axis, endpoints on the
  theta box, frozen axes absent, padding count); `--rc-like` (parsing the
  rebuilt list reproduces every job_args value); end to end on a tiny
  network with the repository pipeline (each design point found in the
  sidecars exactly R times; the prior phase writes T x n_workers x k runs);
  the analysis on synthetic Poisson trains of known rate (rate recovered
  within the Poisson tolerance; a silent run flagged); bytes; the PBS job
  with stub `module`, `conda` and `python` (call sequence and arguments per
  setup, an unknown setup refused). The stubs are shell functions and the
  job's environment is built from scratch (only PATH kept), so nothing from
  the caller's login shell reaches the job, and a noexec /tmp does not stop
  the stubs.
- Confirmed: yes 2026-10-30 for the goal, the two setups, the design (axis
  sweeps and a prior sample), 20 s, a reduced run here and a 48-core job per
  setup on the cluster; inferred for K = 7, R = 2, the three conn levels (the
  box's ends and middle), t_skip = 2 s, T = 8, k = 1 and the reporting
  cut-offs.
- Status: smoke-tested 2026-10-30 in the sandbox (36-D root pipeline,
  freeze-patched copy, 37-D `Giulia/` copy; skipped: davinci, the cluster's
  gate version). davinci 2026-10-30, clone at 1a0932a, `--no-sim`: R8 found
  0 stub calls and then crashed on an unguarded index. The cause is the
  caller's environment reaching the job (an inherited `module` function, or
  BASH_ENV) or a noexec /tmp: each reproduced the failure here; which one
  applied on davinci is not known. R8 rebuilt as above, with guarded checks
  and the job's output printed after a failed check; passes here under
  both causes; rerun on davinci pending. With a noexec /tmp, R5 here and
  S4 of `smoke_test_clamp_cadex_copies.py` also fail (they run programs
  from the temporary folder); not changed.

### 5.1 Projects and where their model runs

The tester can check only what is in the repository. The cluster folders
below are separate copies (not git checkouts, `HPC_Repository_Hierarchy.md`
in project knowledge); Block 5 clamps them and its report, pasted by the
author, is the record for them.

| project | campaign family (job name) | model file it runs | in the repository | how it gets the clamp |
|---|---|---|---|---|
| Giulia's simulations | `campaign_cadex_hhgap_*` (`asn_campaign`) | `Phenomenological/Main/Giulia_Astro/ASD_fun_BD_cpp.py` (worker `lib_dir` = launch folder) (inferred) | `hpc/Phenomenological_finalv1/Giulia/` (relationship to `Giulia_Astro/` unresolved) | repository copy: Block 1; cluster copy: Block 5 |
| rho1300 family, the 100-electrode project (confirmed 2026-10-30; 637 neurons, `C_MAX` 700 um in `submit_sweep_mixed.sh`) | `campaign_cadex_rho1300v*` (`asn_campaign`) | `Phenomenological/Main/ASD_fun_BD_cpp.py` (inferred) | root pipeline `hpc/Phenomenological_finalv1/` (the cluster copy may drift) | repository copy: Block 1; cluster copy: Block 5 |
| MEAMA | `campaign_meama_rho1600_*` (`meama_campaign`) | `Phenomenological/MEAMA/ASD_fun_BD_cpp.py`, 37-D | launch code only (`hpc/Phenomenological_finalv1/MEAMA/`); the 37-D model is not in the repository | Block 5 |
| LindaColloDUP | `campaign_lcdup15_*` (`lcdup_campaign`) | `Phenomenological/LindaColloDUP/ASD_fun_BD_cpp.py`, copied from `MEAMA/` | launch package (Block 4); the model file is not in the repository | inherits it from a clamped `MEAMA/` copy, or Block 4's `apply_refractory_clamp.py` (step 5a); its launcher refuses a mismatch with `REFRACTORY_CLAMP`. On a copy clamped by Block 5, step 5a prints `ALREADY CLAMPED` with exit 0 for `--check` and the patch, and the launcher reads the file as clamped (sandbox, 2026-10-30) |

After each relaunch, Block 3's CODE section, run on the new campaign
folders, must report CLAMPED for each (it reads the model file named by the
campaign's `job_args.json` `lib_dir`).

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_test_refractory_scan.py` | Blocks 2 and 3 | none (synthetic campaigns) | seconds |
| `bash check_refractory_nan.sh --sample 300` | audit of existing campaigns | a campaign folder with `job_args.json` and `topo_*/iter_*.npz` | minutes (not timed on davinci) |
| `qsub -q cpu submit_refractory_scan.sh` | per-run tables for every run | -- | up to 12 h |
| `python apply_refractory_clamp.py --self-test` | Block 4 clamp patcher | embedded fixture | seconds |
| `python apply_freeze_patch.py --self-test` | Block 4 freeze patcher | embedded fixture | seconds |
| `SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_LCDUP.sh` | Block 4 dry run | the LindaColloDUP folder on davinci | seconds |
| `python3 clamp_cadex_copies.py --self-test` and `python3 smoke_test_clamp_cadex_copies.py` | Block 5 | embedded fixtures; a repository clone for the mock tree | seconds |
| `python3 clamp_cadex_copies.py <repository root>` | the repository invariant of Block 1 | the repository | seconds |
| `python3 smoke_test_range_check.py` | Block 6 | the repository pipeline, Brian2 and a C++ compiler | minutes |
| `qsub -v SETUP=...,LIB_DIR=...,OUT=... submit_range_check.sh` | Block 6 on a setup | the setup's folder on davinci | hours (not timed on davinci) |

## 7. Not executable in the sandbox

- Anything on davinci: PBS submission, the 37-D pipeline files (`MEAMA/`,
  `LindaColloDUP/`), real campaign sizes, job-log naming, read speed.
  Substitutes used: 36-D `main` copies with no-op parser options for the
  37-D flags; a stub `qsub`; campaign trees built from real worker output.
- Runs at the campaigns' length (180-200 s); the checks used 5 s, and 30 s
  for false positives.

## 8. Open questions

- Confirm Block 2's thresholds (t_early = 1 s; the late rule) (raised 2026-10-30).
- LindaColloDUP: the VA/VR option and I_inj (raised 2026-10-29).
- MEAMA: a clamped campaign in a folder of its own, or not (raised 2026-10-30).
- Where this project's decision log lives (raised 2026-10-29).
- The cluster folder of each project in section 5.1 (inferred paths) and that
  Block 5's report lists them all as CLAMPED (raised 2026-10-30).
- The 37-D model files of `MEAMA/` (and so of LindaColloDUP) are not in the
  repository, so the tester cannot check them; committing them would bring
  them under this spec (raised 2026-10-29, still open).
- Astrocyte variables ending NaN in Full runs at nominal parameters: a
  separate failure, not investigated (raised 2026-10-29).
- Retune the prior boxes of MEAMA and of the rho1300 family for the clamped
  model, after Block 6 has run on them (decided 2026-10-30, not started).
- Bring the Giulia setup onto GitHub: its 37-D pipeline with the new axis,
  the probability that a synapse is contacted by an astrocyte (the contact
  gate), which today exists only in the cluster's `Giulia_Astro/` (decided
  2026-10-30; files to come from the cluster).
- Block 6 for the 37-D setups: the driver uses only `main`, `sample_theta`,
  `resolve_sweep_group`, `_apply_freeze`, `_resolve_n_workers`,
  `NOMINAL_PARAMS`, `PARAM_BOUNDS_THETA`, `natural_to_theta`,
  `build_parser`, `PARAM_NAMES` and `LOG_PARAMS` of the setup's module;
  checked on the 36-D root pipeline and the 37-D `Giulia/` copy, not on the
  cluster's gate version (raised 2026-10-30).
