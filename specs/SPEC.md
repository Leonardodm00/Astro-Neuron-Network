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
its GIT CLONE status and queued-job hold added on top of 4e12a40; Block 4's
g box patch, conn_prob box, VA/VR values and clone folder, and Block 6's
two-node split, added on top of 15f82c9; Block 7 and Block 1's GitHub-wide
check added on top of a811543; Block 6 rewritten as the standard prior range
calibration on top of dcb52ab)

## 0. Coverage

- `hpc/refractory_audit/` (the audit and its `checks/`)
- `hpc/Phenomenological_finalv1/LindaColloDUP/` (launch package and its `checks/`)
- `hpc/range_check/` (the standard prior range calibration, Block 6)
- the seed guard (Block 7): `hpc/Phenomenological_finalv1/seed_alloc.py`, its
  identical copies in `MEAMA/` and `LindaColloDUP/`, its smoke test
  `smoke_test_seed_allocation.py`, and the seed-allocation blocks of
  `launch_campaign.sh`, `MEAMA/launch_MEAMA.sh` and
  `LindaColloDUP/launch_LCDUP.sh` -- not the rest of those launchers
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
    [corrected 2026-10-30] Each setup now runs on two 48-core nodes (the
    range-check entry below).
  - 2026-10-30 [author]: every running campaign is stopped, and the reruns
    are launched from folders of the git clone on davinci,
    `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network`: everything under
    the clone is clamped, every other `campaign_*` folder under `ANN/` is
    old. Tags as proposed: `cadex_rho1300_clamp_v1`, `cadex_hhgap_clamp_v1`,
    `meama_rho1600_clamp_n01`; LindaColloDUP keeps
    `lcdup15_rho1300_neur_n01` (never launched, and launched only clamped;
    the one exception to the `clamp` stem). LindaColloDUP goes first, from
    `hpc/Phenomenological_finalv1/LindaColloDUP/` in the clone; its 37-D
    pipeline is copied in from the cluster's `MEAMA/` and kept out of git
    (`.gitignore`, as is the campaign output) until the author sends the
    files to be versioned.
  - 2026-10-30 [author]: LindaColloDUP values. VA = -48 mV and VR = -58.2 mV,
    the registry nominals (the cluster's `MEAMA/` registry has the root
    copy's neuron and synapse rows: the author's paste, 2026-10-30);
    I_inj = 4.0 pA; conn_prob ~ U[0.01, 0.5] per topology, to lighten the
    simulations, with the density (1300 neurons/mm^2) and the culture width
    (C_MAX 350 um, so Nn = 159) unchanged; g_ampa and g_nmda boxes
    [0.01, 10] nS, log-uniform, nominal 0.3 nS (registry box [0.01, 5] nS).
  - 2026-10-30 [author]: the range check of a setup runs on two 48-core
    nodes, split as proposed by Claude: the one-axis sweeps on one node, the
    prior sample (16 topologies) on the other, and the analysis as a short
    job once both have ended (Block 6). [corrected 2026-10-30] Superseded by
    the next entry.
  - 2026-10-30 [author]: the standard prior range calibration, for every
    setup: the corners of the prior box (every free axis and conn_prob at
    their minimum or maximum, all combinations; a two-level fraction keeping
    every main effect and two-axis interaction when the 2^d corners exceed
    16,384) and as many random cells of the 7-bin grid (bin midpoints), as
    two 48-core jobs with a 24 h walltime, then the analysis job (Block 6).
    It replaces the one-axis sweeps and the prior sample.
  - 2026-10-30 [author]: the seed guard covers every launch folder under the
    cluster's ANN folder, not only the launching one (Block 7): the reruns
    start in fresh folders of the git clone, which hold no seed history.
    Asked by the author ("double check the seed guard") under the standing
    rule of 2026-08 that every campaign launch gets seeds no other launch has
    used.
  - 2026-10-30 [author]: every CAdEx model on GitHub carries the refractory
    hold, the second branch `Inverse-Transform-Sampling` included (Block 1).
  - 2026-10-29 [author]: LindaColloDUP runs MODE Neuronal; freezes Cm,
    I_inj, VR, VA, DeltaA, EC50_ampa, EC50_nmda (and the inert alpha_syn,
    O_G, Omega_G); SIMTIME 180 s; RHO_TOTAL 1300 cells/mm^2; everything else
    as MEAMA. The frozen values of VA, VR and I_inj were decided on
    2026-10-30 (entry above).

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
- GitHub-wide check (2026-10-30, asked by the author): every voltage
  equation in every tracked file of both branches was read (`git grep` for
  `d v.../dt =` and for the exponential term in any spelling). `main`
  a811543: the two files above are the only runnable CAdEx models, both
  clamped; the other hits are Hodgkin-Huxley models (top-level legacy
  scripts, `hpc/ASD_fun_BD_cpp.py`), a toy LIF in `MEAMA/env_check.py`, test
  fixtures, detection code and documentation. Branch
  `Inverse-Transform-Sampling`: its CAdEx copy
  `hpc/Phenomenological main/ASD_fun_BD_cpp.py` (an older model, `I` in place
  of `I_inj*bias_unit`) got the same flag and comment lines in 2f5b260;
  Brian2 parses it, `clamp_cadex_copies.py` rates it CLAMPED, and its own
  sweep pipeline on identical draws (60 neurons, 3 s, 6 runs, seed 1000)
  logged 4 NaN warnings unclamped and 0 clamped. (That commit's message says
  the branch no longer holds its sweep pipeline; it does.) Not CAdEx and not
  changed: the Izhikevich model of the top-level `ASN_fun_pheno.py` (used
  only by `Main_code_pheno.py`) has `refractory=2*ms` and no hold, the same
  failure class (section 8).
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
  `submit_LCDUP.sh`, `apply_freeze_patch.py`, `apply_box_patch.py`,
  `smoke_test_freeze.py`, `apply_refractory_clamp.py`, `APPLY_LCDUP.txt`
  (procedure), `.gitignore`, `seed_alloc.py` (Block 7; tracked here since
  2026-10-30, not copied from `MEAMA/`, whose copy cannot scan other
  folders). Deployed on davinci in this same folder of the git clone
  (decided 2026-10-30, section 4), with the 7 pipeline files copied in from
  the cluster's 37-D `MEAMA/`, which is not in the repository. `.gitignore` (anchored patterns) keeps the copies, their
  backups (`*.bak-*`, `*.crlf_bak`), `campaign_*/` and `artifacts/` out of
  git; `git clean -x` would delete them.
- `launch_LCDUP.sh` settings (environment variable = default): MODE =
  Neuronal; SWEEP_GROUP = neuron_synapse; RHO_TOTAL = 1300; ASTRO_FRAC = 0;
  C_MAX = 350 um; SIMTIME = 180 s; FREEZE =
  Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G
  (VA and VR written out at the registry nominals, so a changed registry
  cannot move them); CONN_RULE = flat; CONN_PERIODIC = 0; CONN_PROB_LO/HI =
  0.01/0.5; STOA_GATE_LO/HI = 1.0/1.0; N_TOPOLOGIES_WORKER = 32; K_PARAMS = 1;
  WALLTIME = 24:00:00; TARGET = 1000000; HEADROOM_PCT = 200; CAMPAIGN_TAG =
  lcdup15_rho1300_neur_n01; CONDA_ENV = brian_final; DRYRUN = 0. The g boxes
  are not launcher settings: they live in the folder's `HPC_main_sweep.py`
  after `apply_box_patch.py`; the launcher prints them (`g boxes : ...`) and
  writes them into the PROVENANCE file.
- Guards of `launch_LCDUP.sh`: exit 2 missing pipeline or package files, a
  `HPC_main_sweep.py` without the freeze patch, or one without both box
  markers; exit 3 SIZING_CONFIRMED not set, or REFRACTORY_CLAMP unset or not
  yes/no; exit 4 REFRACTORY_CLAMP disagreeing with the file, MODE,
  ASTRO_FRAC, gate, conn_prob bounds, FREEZE syntax, existing campaign tag;
  exit 5 seed allocator missing, an allocator without `--scan-tree`, a
  missing SEED_SCAN_TREE, or no free seed range (Block 7). `submit_LCDUP.sh`
  repeats the freeze and box guards on the compute node (exit 2).
- `apply_freeze_patch.py`: adds `--freeze` to the folder's own
  `HPC_main_sweep.py` (3 insertions, marker `[LindaColloDUP patch 2026-10-29]`);
  `--check`, `--self-test`; refuses CR bytes, partial or foreign patches,
  missing or duplicate anchors.
- `apply_box_patch.py` (decided 2026-10-30): in the folder's own
  `HPC_main_sweep.py`, rewrites the two rows of the `PARAM_BOUNDS` array
  whose comments name g_ampa and g_nmda from the registry box (0.01, 5.0) to
  (0.01, 10.0) nS and appends the marker `[LindaColloDUP box 2026-10-30]` to
  each; nothing else in the file changes. Everything derived from
  `PARAM_BOUNDS` follows at import: both axes stay in `LOG_PARAMS` (both
  bounds > 0 and log10(hi/lo) >= 1), the theta box becomes
  [ln 0.01, ln 10], and `job_args.json` `_axis_declaration` records
  low 0.01, high 10, log true. Refuses, writing nothing (exit 1): CR bytes,
  not UTF-8, no `PARAM_BOUNDS` block, a row missing or found twice, a box
  other than (0.01, 5.0) or (0.01, 10.0), the new box without the marker, a
  partial patch. Exit 0 patched or already patched. Backup
  `HPC_main_sweep.py.bak-box-<YYYYmmdd_HHMMSS>`; atomic write keeping the
  permissions; re-read (must classify as patched) and `py_compile`; restores
  the backup on failure. `--check`, `--self-test`, `--file`.
- `apply_refractory_clamp.py`: Block 1's flag on the folder's own
  `ASD_fun_BD_cpp.py` (marker `[LindaColloDUP clamp 2026-10-29]`); `--check`,
  `--self-test`; exit 0 patched or already clamped, 1 refused.
- Custom code: the two patchers and the box rule; no library edits a
  registry in place.
- Test oracles: `smoke_test_freeze.py` (40 checks: T1-T8; T4 also sets a
  value off the nominal to prove NAME=VALUE is applied, since the campaign's
  values equal the nominals; T7 checks the g boxes in `job_args.json`; T8
  the box patch: both markers, box, log flag and theta box of each g axis);
  the three patchers' `--self-test` (the box patcher's 20 checks include the
  refusals, the backup, idempotence and file permissions; the clamp's
  includes a Brian2 parse of the patched equations); the dry run (Nn = 159,
  `conn_prob : U[0.01, 0.5] per topology`, `g boxes : g_ampa [0.01, 10.0]
  nS; g_nmda [0.01, 10.0] nS`, two qsub lines `-J 0-651%2`); the worker
  probe of `APPLY_LCDUP.txt` step 6 (NaN warnings 0 when clamped; its
  `job_args.json` records 13 free axes, conn_prob 0.01-0.5, both g boxes,
  VA -48, VR -58.2, I_inj 4.0).
- Data flow: launches campaigns from the patched folder copies.
- Confirmed: yes 2026-10-29 for the settings the author decided (freeze
  list, SIMTIME, RHO_TOTAL, the rest as MEAMA); yes 2026-10-30 for
  REFRACTORY_CLAMP = yes, VA = -48 mV and VR = -58.2 mV, I_inj = 4.0 pA,
  conn_prob U[0.01, 0.5], the g boxes [0.01, 10] nS and the clone folder.
- Status: smoke-tested 2026-10-30 in the sandbox on a fresh folder built
  from the package files and a stand-in for the `MEAMA/` copy (the
  repository's root 36-D files, whose registry rows equal `MEAMA/`'s; the
  unclamped model of d0ced88; the three 37-D flags as no-op options):
  `APPLY_LCDUP.txt` steps 2-7 as written, the guards above, the probe.
  Skipped: the 37-D files and davinci.

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

### Block 6: range_check, the standard prior range calibration

- Purpose (decided by the author on 2026-10-30, the standard for every
  setup): before a campaign is launched, measure the activity its prior box
  produces, through the setup's own sweep code, at the corners of the box and
  at random cells of its B-bin grid; 20 s per run.
- Modules (`hpc/range_check/`): `range_design.py` (the designs; pure
  functions, numpy only), `range_check.py` (driver), `range_analysis.py`
  (metrics, labels, summaries), `range_plots.py` (figures only),
  `submit_range_check.sh` (the PBS job), `launch_range_check.sh` (login node:
  one setup as three PBS jobs on two 48-core nodes),
  `smoke_test_range_check.py`, `README_RANGE_CHECK.txt`.
- Notation. For a setup with sweep group G and freeze list Phi:
  F = (j_1, ..., j_dr), the indices of G minus Phi (`resolve_sweep_group`,
  the module's `_apply_freeze`) whose theta box has lo_j < hi_j, is the set of
  free axes; theta_j is the sweep's coordinate of axis j (`PARAM_BOUNDS_THETA`:
  ln on `LOG_PARAMS` axes, natural units otherwise), box [lo_j, hi_j].
  conn_prob is a design dimension when the setup's conn_rule is flat and its
  box [c_lo, c_hi] (`--conn_prob_lo/hi`) has c_lo < c_hi, in natural units.
  d = dr + 1 with conn_prob, d = dr without. theta0 =
  natural_to_theta(NOMINAL_PARAMS) after the freeze. N_w = n_workers (the
  module's `_resolve_n_workers`). seed = `--rc-seed` (default 20261030).
- Corners: X in {-1, +1}^(N_c x d), columns F then conn_prob.
  - If 2^d <= N_max (`--rc-corner-budget`, default 16,384): the full
    factorial, N_c = 2^d, row r column k = +1 if bit k of r is 0, else -1.
  - Else a regular 2^(d-p) fraction: b = floor(log2 N_max), N_c = 2^b,
    p = d - b; columns 0..b-1 the full factorial of b; column b+i the product
    of the base columns set in the generator mask g_i. The masks come from the
    greedy search `range_design.fraction_generators` (random orders from
    `default_rng(seed + 20000)`), trying the resolution R = 8, 7, 6, 5 in turn
    (8 orders each) and keeping the first found; `min_word_length` verifies
    that every word of the defining relation, |G| + popcount(xor of the masks
    in G) for a nonempty generator set G, is >= R. R >= 5: no main effect or
    two-axis interaction is aliased with another. None found: error.
  - A run with codes x: theta_j = lo_j where x_j = -1, hi_j where +1 (j in
    F), every other axis at theta0; conn_prob = c_lo where x_conn = -1, c_hi
    where +1.
  - Level L in {0, 1} (conn_prob at c_lo, c_hi; level 0 only when conn_prob
    is not a dimension): the n_L rows with x_conn = 2L - 1, permuted by
    `default_rng(seed + 20100 + L)`, split over T_c topologies
    (`--rc-corner-topologies`, default 8) of N_w k_c consecutive slots,
    k_c = ceil(n_L / (T_c N_w)); the T_c N_w k_c - n_L last slots hold the box
    centre, theta_j = (lo_j + hi_j) / 2 for j in F (centre points).
- Interior: B bins per dimension (`--rc-bins`, default 7) of equal width in
  the dimension's own coordinate, so equal prior mass under a uniform or
  log-uniform prior; midpoints m_(j,i) = lo_j + (i + 1/2)(hi_j - lo_j)/B,
  i = 0..B-1, and the same over [c_lo, c_hi] for conn_prob. Level L = the
  conn_prob bin, L = 0..B-1 (level 0 only when conn_prob is not a
  dimension): conn_prob = m_(conn,L); n = T_i N_w k_i cells,
  k_i = ceil(ceil(N_i / n_levels) / (T_i N_w)), N_i = `--rc-interior-runs`
  (default 16,384), T_i = `--rc-interior-topologies` (default 7); for each
  cell and each j in F independently, i_j ~ U{0, ..., B-1}
  (`default_rng(seed + 30000 + L)`), theta_j = m_(j,i_j); every other axis
  at theta0.
- Driver: `python3 range_check.py --rc-phase {corners,interior} --rc-level L
  --rc-out DIR [--rc-like JOB_ARGS] [--rc-bins B] [--rc-corner-budget N]
  [--rc-corner-topologies T] [--rc-interior-runs N] [--rc-interior-topologies
  T] [--rc-simtime T] [--rc-seed S] [--rc-subset N] [--rc-design-only]
  [--rc-allow-unclamped] <HPC_main_sweep.py arguments>`. It imports
  `HPC_main_sweep` from the `--lib_dir` among the pass-through arguments and
  calls its `main()` with them followed by `--out_dir DIR/corners_c<L>` or
  `DIR/interior_b<L>`, `--simtime` (default 20 s), `--n_topologies T`,
  `--n_params_per_worker k`, `--seed_master` (seed + 2000 + L for corners,
  seed + 3000 + L for the interior) and `--conn_prob_lo P --conn_prob_hi P`
  with P the level's conn_prob. Later arguments win. In the parent process
  only, `sample_theta` is replaced by a function that returns the padded
  design in order, so successive topologies take successive chunks of N_w k
  slots: the only change to the production path. `--rc-subset N`: the first
  N design points of the level, on one topology (tests). `--rc-like` rebuilds
  the base argument list from a campaign's `job_args.json` through the
  module's own parser (each key that is a parser destination; store_true
  options as flags when true; None values skipped; never out_dir,
  scratch_root, lib_dir, seed_master, n_workers, n_topologies,
  n_params_per_worker, simtime, rebuild_manifest_only); explicit arguments
  after it override. Refuses (exit non-zero, nothing run) an unclamped model
  unless `--rc-allow-unclamped`, an output folder that already holds runs,
  and a level outside the phase's range. Exit 1 when not every slot was drawn
  (a topology failed to build); the runs made stay usable.
- `design.json` (one per level folder): phase, level, free_axes, theta_lo,
  theta_hi, has_conn, conn_box, conn_prob, nominal_theta, n_workers,
  n_topologies, k, n_slots, n_design, n_padding; corners: corner_design (kind,
  n_runs, resolution, base, generators, d, n_levels, n_level_corners), codes
  (n_design x dr), centre_theta; interior: bins, n_levels, midpoints, cells
  (n x dr); and simtime, seed_master, lib_dir, module, model_line,
  param_names, log_params, argv.
- Analysis (`range_analysis.py DIR`): per run, the neuron spikes in the
  window [t_skip, T] (t_skip default 2 s, shifted to start at 0) go through
  `burst_metrics.compute_all` for the burst rate (log-ISI detector,
  `li_burst_rate_per_min`), CV of ISI and adaptation index; mean rate =
  spikes in the window / (Nn (T - t_skip)); active fraction = neurons with a
  window rate > 0.1 Hz. Labels come from the run's stored theta and its
  folder's design.json, independent of the draw order: corners, per free
  axis, code -1 or +1 where theta equals lo or hi and 0 at the box centre
  (tolerance 1e-9 of the box width), 'corner' when every code is +-1,
  'centre' when every code is 0; interior, per free axis the bin whose
  midpoint theta equals, 'interior' when every axis has one; else
  'unmatched' (exit 1). conn_prob: code 2L - 1 (corners), bin L (interior).
  Outcomes: y = log10(rate + 0.01 Hz); silent = 1[active fraction < 0.1];
  high = 1[rate > 20 Hz]; bursting = 1[burst rate > 0] (reporting
  conventions, not targets). Writes `runs.csv`; `corner_effects.csv`: for
  every dimension j, over the corner runs, E_j(f) = mean(f | x_j = +1) -
  mean(f | x_j = -1) for f = y, silent, high, bursting, with n(+) and n(-);
  `corner_interactions.csv`: for every pair (j, k), E_jk(y) =
  mean(y | x_j x_k = +1) - mean(y | x_j x_k = -1), the two-level interaction
  contrast; `interior_bins.csv`: per dimension and bin, n, rate quartiles,
  shares silent / high / bursting, median burst rate; `interior_pairs.csv`:
  per pair of dimensions and pair of bins, n, share silent, share high,
  median y; `interior_spearman.csv`: Spearman rho (`scipy.stats.spearmanr`)
  of each dimension (theta; the run's conn_prob) with y and with the burst
  rate; `calibration_summary.txt`: corner runs and the distinct corners of
  the design they cover, centre points, shares, the curvature mean y(centre)
  - mean y(corners), the main effects, the 20 largest interactions, the five
  lowest- and highest-rate corners, the interior bins where more than half
  the runs are silent or above 20 Hz, the Spearman list.
- Plots (`range_plots.py DIR`): `corner_effects.png` (main effects on y and
  on the shares), `interior_bins.png` (one panel per dimension: median rate
  per bin with the interquartile band on one shared scale, shares per bin),
  `rate_distributions.png` (log rate of corners, centre points, interior).
- Job script (`submit_range_check.sh`, 48 cores, walltime 24 h, `-k eo`):
  SETUP = lcdup | giulia (else exit 2). lcdup: LIB_DIR default
  `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP`
  (Block 4's folder); the settings of `launch_LCDUP.sh` written out (FREEZE
  default `Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G`
  or FREEZE_ENC decoded, `:` for `,` and `@` for `=`; `--conn_prob_lo 0.01
  --conn_prob_hi 0.5`; `--Nn 115` overridden by the density, so Nn = 159);
  refuses a LIB_DIR without both box markers. giulia: LIB_DIR default
  `Phenomenological/Main/Giulia_Astro`, LIKE = the first
  `campaign_cadex_hhgap_v5/sweep_*/job_args.json` there (none: exit 2).
  EXTRA (space-separated) ends every call. PHASES = all (default) | corners |
  interior | none: corners at levels CORNER_LEVELS (default `0 1`) with
  `--rc-corner-topologies CORNER_TOPOS` (8) `--rc-corner-budget
  CORNER_BUDGET` (16384); interior at levels INTERIOR_LEVELS (default
  0..BINS-1, BINS 7) with `--rc-bins BINS --rc-interior-runs INTERIOR_RUNS`
  (16384) `--rc-interior-topologies INTERIOR_TOPOS` (7); ANALYZE = 1
  (default) | 0: `range_analysis.py OUT --workers <ncpus>` then
  `range_plots.py OUT`. Exit 2 before conda: PHASES or ANALYZE invalid,
  PHASES=none with ANALYZE=0, `range_check.py` or `range_design.py` missing
  from the submission folder, a LIB_DIR without `HPC_main_sweep.py`, the
  lcdup box guard, PHASES=none with no `OUT/*/design.json`. CHECK_ONLY=1
  runs those guards (all but the last), prints `[range_check] precheck OK:
  ...` and exits 0. Exit 2 also when `conda activate` or the environment
  check fails; otherwise 1 if any call failed, else 0.
- Launcher (`launch_range_check.sh`, login node): refuses (exit 2) an
  unknown SETUP and a comma in OUT or in a passed variable (`qsub -v` splits
  on commas); runs the job script once with CHECK_ONLY=1 (failure: exit 2,
  nothing submitted); then submits into one OUT (default
  `~/ANN/range_check/<setup>_<UTC>`): job 1 `rc_corners` (Q_CORNERS, default
  intel, PHASES=corners ANALYZE=0), job 2 `rc_interior` (Q_INTERIOR, default
  cpu, PHASES=interior ANALYZE=0), both `select=1:ncpus=NCPUS` (48) and
  WALLTIME (24 h); job 3 `rc_analysis` (Q_ANALYSIS cpu, ANALYSIS_NCPUS 8,
  ANALYSIS_WALLTIME 2 h, `-W depend=afterany:<job 1>:<job 2>`, PHASES=none
  ANALYZE=1). Passed on when set: LIB_DIR, LIKE, BINS, CORNER_LEVELS,
  CORNER_TOPOS, CORNER_BUDGET, INTERIOR_LEVELS, INTERIOR_TOPOS,
  INTERIOR_RUNS, RC_SIMTIME, EXTRA, FREEZE_ENC, CONDA_ENV. A refused job 1:
  exit 3, nothing else submitted; job 2: exit 3 and the `qdel` line for job
  1; job 3: exit 3 and the manual command. DRYRUN=1 prints the three qsub
  lines and submits nothing.
- LindaColloDUP sizes: dr = 13, d = 14, so the full factorial: per conn
  level 8,192 corners + 256 centre points = 8 x 48 x 22 = 8,448 runs (16,896
  in all, conn_prob 0.01 and 0.5); interior 7 levels of 2,352 cells
  (7 x 48 x 7), 16,464 runs, conn_prob = 0.045 + 0.07 L.
- Custom code: the fraction's generator search (`range_design.py`
  docstring: pyDOE3 1.6.2's `fracfact_by_res(n, 5)` is not resolution V for
  n = 14 or 23, checked 2026-10-30), the labels and the effect contrasts.
- Test oracles (`smoke_test_range_check.py`, R1-R10; 63 checks with Brian2,
  56 and 1 skipped with `--no-sim`): R1 the fraction for the 36-D root
  pipeline (resolution 8 here, floor 7; no aliased main/two-axis pair in a
  Gram check; balanced; the two levels together are the design; every vector
  at the bounds its codes name; centre points; inactive axes at nominal); R2
  the full factorial with the LindaColloDUP freeze (each level all 2^13
  corners once; VA, VR held, also off the nominal); R3 `--rc-like`; R4 the
  37-D Giulia copy; R5 the interior (sizes, conn midpoints, every value a bin
  midpoint, chi-square uniformity over 91 tests, seed determinism); R6 end to
  end on a tiny network (labels equal the design's codes and bins, tables,
  figures, refusals); R7 a planted effect recovered (E_A = 0.5985 for a
  planted 0.6004; inert axes |E| < 0.003; interior medians doubling per bin;
  an off-grid run unmatched, exit 1); R8 bytes; R9 the job script and R10
  the launcher with stubs in a sealed environment, each probed first. Ten
  planted faults (2026-10-30) were each caught by a named check.
- Confirmed: yes 2026-10-30 for the design (corners at every free axis's
  bounds in all combinations, the same number of random cells of the 7-bin
  grid, a fraction keeping every main effect and two-axis interaction when
  2^d > 16,384, two 48-core jobs with a 24 h walltime then the analysis job,
  the standard for every setup), the two setups and the 20 s runs; inferred
  for T_c = 8, T_i = 7, the centre points, the seeds, t_skip = 2 s, the
  reporting cut-offs and the queue assignment.
- Status: smoke-tested 2026-10-30 in the sandbox (36-D root pipeline,
  freeze-patched copy, 37-D `Giulia/` copy; both phases with the exact
  LindaColloDUP settings on 159 neurons: corners at conn_prob 0.5, interior
  at 0.465, labels and analysis correct). Skipped: davinci, the cluster's
  gate version. History: until 2026-10-30 (a3878d2) this block specified
  one-axis sweeps at three conn_prob levels and a prior sample, replaced by
  this design at the author's request. The R8 failure on davinci of
  2026-10-30 (an exported `module` function reaching the job) was fixed by
  the sealed environment, now used by R9 and R10; it passed on davinci at
  1757d3e.

### Block 7: the seed guard across ~/ANN

- Module: `hpc/Phenomenological_finalv1/seed_alloc.py` (standard library
  only), byte-identical copies in `MEAMA/` and `LindaColloDUP/`; callers: the
  seed blocks of `launch_campaign.sh` (rho1300 family), `MEAMA/launch_MEAMA.sh`
  and `LindaColloDUP/launch_LCDUP.sh`.
- Invariant (author, 2026-08, extended 2026-10-30): a launch's seed span
  [base, base + span) contains no `_resolved_seed_master` recorded by any
  campaign and overlaps no span reserved by any launch, in its own folder and
  in every launch folder under SEED_SCAN_TREE (default
  `/davinci-1/home/ldellamea/ANN`).
- `scan_tree(tree, max_depth=6)`: `os.walk` of `tree`, symbolic links to
  folders not followed. In every folder at most `max_depth` levels below
  `tree` (`tree` is level 0): the child `campaign_*` folders are read as one
  launch folder (`used_seeds_from_job_args`: `campaign_*/sweep_*/job_args.json`,
  truncated files counted and skipped) and not entered; `artifacts/seed_ledger.tsv`,
  if present, is read as a ledger. Not entered: hidden folders, names starting
  `campaign_`, `scratch`, `mea_out`, `topo_`, and `__pycache__`, `artifacts`.
  Returns the seeds, the spans and counts (campaign folders, readable and
  unreadable job_args.json, ledgers, unreadable ledgers and folders). A tree
  that is not a directory raises ValueError.
- `allocate(..., scan_trees, scan_depth)`: the own folder's seeds and ledger
  (unchanged) plus every scan; spans read twice (the own ledger lies inside
  the tree) are de-duplicated; a random base is redrawn until the span is
  free (256 attempts, else RuntimeError, exit 3). Only the own `--ledger` is
  written.
- CLI: `--scan-tree DIR` (repeatable), `--scan-depth N` (default 6); a
  missing tree exits 2 with no base on stdout; `--report` prints each scan
  (campaign folders, job_args.json, seeds, ledgers with their span counts)
  and the totals. Without `--scan-tree` the behaviour is unchanged.
- Launchers: `SEED_SCAN_TREE` (default the ANN folder) passed as
  `--scan-tree`; refuse (exit 5) an allocator without `--scan-tree`, a
  missing tree and a failed allocation, with nothing submitted; the header
  prints `Seed guard : this folder and every launch folder under ...`;
  `launch_LCDUP.sh` also writes it, and the sha256 of `seed_alloc.py`, into
  the PROVENANCE file. A dry run still reserves its span (note
  `DRYRUN-not-submitted`). `launch_campaign.sh` and `launch_MEAMA.sh` keep
  their existing fallback when no allocator is found (raw OS entropy with a
  warning); the LindaColloDUP launcher refuses.
- Not covered by the scan, by design: the range check's runs (fixed seeds,
  outside any `campaign_*` folder, never training data); launchers outside
  the three above (`Burst Tests/`, `Multiple Sweeps/` with its hard-coded
  bases, the MEAMA benchmark's fixed seed).
- Test oracles: `smoke_test_seed_allocation.py` T7 (10 checks): the scan
  reads the campaigns and ledgers of launch folders at depths 1 and 2 and
  skips hidden, `scratch*`, `topo_*`, inside-campaign and too-deep folders;
  `--scan-depth 7` reaches depth 7; a missing tree is refused; from one
  folder, 40 allocations in a small seed space overlap a neighbouring
  folder's span without the scan (16 of 40 in the run of 2026-10-30) and
  never with it; only the own ledger is written; `--report` lists the scan.
  Groups T1-T6 unchanged (47 checks in all).
- Confirmed: yes 2026-10-30 for the scope (all of ANN) and the refusal of a
  missing tree; inferred for the depth 6, the skipped folder names and the
  default path.
- Status: smoke-tested 2026-10-30 in the sandbox; the three launchers run
  against a mock ANN tree (an old MEAMA-like campaign and ledger beside the
  clone): each allocated clear of it and refused (exit 5, no qsub call) an
  old allocator; `launch_LCDUP.sh` also refused a missing tree and a tree
  whose ledgers reserve the whole seed space. A git pull replaces an ignored
  copy of MEAMA's allocator left by an earlier step 0 (git 2.43). Skipped:
  davinci (the real ANN tree, its size and its git 2.31).

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
| LindaColloDUP | `campaign_lcdup15_*` (`lcdup_campaign`) | `Phenomenological/LindaColloDUP/ASD_fun_BD_cpp.py`, copied from `MEAMA/` [corrected 2026-10-30: `Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/ASD_fun_BD_cpp.py` in the git clone, copied from `MEAMA/` and ignored by git; Block 5 sees it as an ordinary file, not GIT CLONE, since it is untracked] | launch package (Block 4); the model file is not in the repository | inherits it from a clamped `MEAMA/` copy, or Block 4's `apply_refractory_clamp.py` (step 5a); its launcher refuses a mismatch with `REFRACTORY_CLAMP`. On a copy clamped by Block 5, step 5a prints `ALREADY CLAMPED` with exit 0 for `--check` and the patch, and the launcher reads the file as clamped (sandbox, 2026-10-30) |

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
| `python apply_box_patch.py --self-test` | Block 4 g box patcher | embedded fixture | seconds |
| `python smoke_test_freeze.py` | Block 4, both patches and the axis set | the LindaColloDUP folder after both patches (numpy, matplotlib) | seconds |
| `SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_LCDUP.sh` | Block 4 dry run | the LindaColloDUP folder on davinci | seconds |
| `python3 clamp_cadex_copies.py --self-test` and `python3 smoke_test_clamp_cadex_copies.py` | Block 5 | embedded fixtures; a repository clone for the mock tree | seconds |
| `python3 clamp_cadex_copies.py <repository root>` | the repository invariant of Block 1 | the repository | seconds |
| `python3 smoke_test_seed_allocation.py` | Block 7 (and the allocator's older groups) | numpy, the root 36-D pipeline | seconds |
| `python3 seed_alloc.py --root . --ledger artifacts/seed_ledger.tsv --scan-tree /davinci-1/home/ldellamea/ANN --report` | Block 7 on davinci: what the guard sees | a launch folder | seconds (not timed on davinci) |
| `python3 smoke_test_range_check.py` | Block 6 | the repository pipeline, Brian2 and a C++ compiler | minutes |
| `qsub -v SETUP=...,LIB_DIR=...,OUT=... submit_range_check.sh` | Block 6 on a setup, one node | the setup's folder on davinci | hours (not timed on davinci) |
| `SETUP=lcdup bash launch_range_check.sh` (first with `DRYRUN=1`) | Block 6 on a setup: corners and interior on two nodes, then the analysis job | the setup's folder on davinci; `qsub` | seconds to submit; the jobs: 352 (corners) and 343 (interior) runs per worker, about 11 h per node at the sandbox core's speed (not timed on davinci; `claude/Prior_Range_Calibration_Standard.md` section 6) |
| `python3 range_check.py --rc-phase corners --rc-level 0 --rc-out X --rc-design-only <settings>` | Block 6: a level's design, nothing run | the setup's folder | seconds |

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
  [closed 2026-10-30] VA -48 mV, VR -58.2 mV, I_inj 4.0 pA (section 4).
- MEAMA: a clamped campaign in a folder of its own, or not (raised 2026-10-30).
  [closed 2026-10-30] Superseded: the reruns are launched from folders of the
  git clone (section 4); MEAMA's folder there is chosen when it is prepared.
- LindaColloDUP at the low end of its conn_prob box (noted 2026-10-30, for
  the range check to answer). Under the flat rule each directed pair is
  wired independently, so a neuron's in-degree is Binomial(Nn - 1,
  conn_prob). At conn_prob 0.01 and Nn = 159: mean in-degree 1.58,
  20.4 % of the neurons (32.5 per network) receive no synapse, and the
  largest strongly connected group holds 37 % of the neurons (weakly
  connected: 95 %); at 0.02: 4.1 % without input, 91 % strongly connected;
  at 0.05: 0.05 neurons per network without input, 99.9 %. 8.2 % of the
  topologies drawn from U[0.01, 0.5] lie below 0.05. (Exact binomial values
  and a Monte Carlo of the flat rule, 2000 networks per value, sandbox.)
  Those networks may be driven mostly by the per-neuron bias and the noise
  rather than by recurrence; the oat level 0.01 shows it.
- The pipeline files of the LindaColloDUP folder in the clone are copies
  outside git until the author sends them; versioning them in a separate
  repository folder, matched to a campaign through the sha256 values in its
  PROVENANCE file, would bring them under this spec (raised 2026-10-30).
- Where this project's decision log lives (raised 2026-10-29).
- The cluster folder of each project in section 5.1 (inferred paths) and that
  Block 5's report lists them all as CLAMPED (raised 2026-10-30).
- The 37-D model files of `MEAMA/` (and so of LindaColloDUP) are not in the
  repository, so the tester cannot check them; committing them would bring
  them under this spec (raised 2026-10-29, still open).
- Astrocyte variables ending NaN in Full runs at nominal parameters: a
  separate failure, not investigated (raised 2026-10-29).
- The legacy Izhikevich model in the top-level `ASN_fun_pheno.py` has a 2 ms
  refractory period and no hold: during it the quadratic term can carry v
  past the threshold with no reset, as in the CAdEx runaway. Not used by any
  campaign; fix only if that code is revived (hold v at c, or drop the
  refractory period, as Izhikevich 2003 has none) (raised 2026-10-30).
- `launch_campaign.sh` and `launch_MEAMA.sh` still fall back to raw OS
  entropy, with a warning, when no seed allocator is found; the
  LindaColloDUP launcher refuses instead. Make them refuse too when their
  reruns are prepared (raised 2026-10-30).
- Retune the prior boxes of MEAMA and of the rho1300 family for the clamped
  model, after Block 6 has run on them (decided 2026-10-30, not started).
- Bring the Giulia setup onto GitHub: its 37-D pipeline with the new axis,
  the probability that a synapse is contacted by an astrocyte (the contact
  gate), which today exists only in the cluster's `Giulia_Astro/` (decided
  2026-10-30; files to come from the cluster).
- Topology-level axes other than conn_prob (the contact gate of the new
  Giulia setup) are not design dimensions of Block 6 yet: with a gate box in
  EXTRA each topology draws its gate from the prior (raised 2026-10-30).
- Block 6 for the 37-D setups: the driver uses only `main`, `sample_theta`,
  `resolve_sweep_group`, `_apply_freeze`, `_resolve_n_workers`,
  `NOMINAL_PARAMS`, `PARAM_BOUNDS_THETA`, `natural_to_theta`,
  `theta_to_natural`, `build_parser` (and its `conn_prob_lo`, `conn_prob_hi`,
  `conn_rule`), `PARAM_NAMES`, `LOG_PARAMS` and `N_DIMS` of the setup's module;
  checked on the 36-D root pipeline and the 37-D `Giulia/` copy, not on the
  cluster's gate version (raised 2026-10-30).
